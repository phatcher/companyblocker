//! Shared dump-opening plumbing: opens the gzip-compressed dump file through `rapidgzip-core`'s
//! parallel decoder rather than decompressing it on the caller's thread. The dump is one
//! trivial stub member followed by one continuous member carrying essentially all the content, so
//! there is no free multi-member parallelism to exploit; `rapidgzip-core` instead builds a
//! speculative index of DEFLATE block boundaries within that one member and decompresses the
//! indexed blocks in parallel, on its own worker threads, handing the caller's thread only the
//! ordered decoded bytes through `std::io::Read`.
//!
//! Called by both `extract::InputSource::open` and `profile::run_profile`, so every dump read
//! (`extract`, `--resume`, and `profile`) shares the one parallel reader.
//!
//! `rapidgzip_core::Decoder::open` takes the file path itself and reads it positionally from its
//! own worker threads, so no caller-side read-ahead buffer applies here.

use rapidgzip_core::{DecodeError, Decoder, DecoderReader};
use std::io;
use std::path::Path;

/// Opens `path` as a gzip-compressed dump file and returns a `Read + Send` decompressed stream
/// backed by `rapidgzip-core`'s parallel decoder.
///
/// # Errors
/// Returns an error if `path` can't be opened, or if the gzip framing is invalid.
pub(crate) fn open_parallel_gzip_source(path: &Path) -> io::Result<DecoderReader> {
    let decoder = Decoder::builder()
        .build()
        .map_err(|error| io::Error::other(error.to_string()))?;
    decoder
        .open(path)
        .map_err(|error| decode_error_to_io_error(&error))
}

/// Converts a `rapidgzip-core` decode failure into an `io::Error`, preserving the original
/// `io::ErrorKind` (e.g. `NotFound`) when the failure came from an underlying I/O error rather
/// than a framing problem.
fn decode_error_to_io_error(error: &DecodeError) -> io::Error {
    if let DecodeError::Io { source, .. } = error {
        return io::Error::new(source.kind(), error.to_string());
    }
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ScopedTempDir;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::fs::File;
    use std::io::{Read, Write};

    /// A multi-member gzip file: two independently-finished gzip streams concatenated one after
    /// the other -- the shape the real dump's trivial stub member plus its one content member
    /// takes, and what `rapidgzip-core` (like the `flate2::read::MultiGzDecoder` it replaced)
    /// exists to handle. Nothing here depends on the real dump file actually being multi-member;
    /// the point is that `open_parallel_gzip_source` decodes one correctly regardless.
    fn write_multi_member_gzip(path: &Path, first: &[u8], second: &[u8]) {
        let mut file = File::create(path).unwrap();
        for chunk in [first, second] {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(chunk).unwrap();
            let member = encoder.finish().unwrap();
            file.write_all(&member).unwrap();
        }
    }

    #[test]
    fn open_parallel_gzip_source_decodes_a_multi_member_file_byte_for_byte() {
        let dir = ScopedTempDir::new("dump-source-multi-member");
        let path = dir.path().join("multi-member.gz");
        let first = b"first member payload\n".repeat(100);
        let second = b"second member payload\n".repeat(200);
        write_multi_member_gzip(&path, &first, &second);

        let mut reader = open_parallel_gzip_source(&path).unwrap();
        let mut output = Vec::new();
        reader.read_to_end(&mut output).unwrap();

        let mut expected = first;
        expected.extend_from_slice(&second);
        assert_eq!(output, expected);
    }

    #[test]
    fn open_parallel_gzip_source_returns_an_error_for_a_missing_file() {
        let dir = ScopedTempDir::new("dump-source-missing");
        let result = open_parallel_gzip_source(&dir.path().join("does-not-exist.gz"));
        let error = result.err().expect("expected an error for a missing file");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    /// A framing failure (not the file-open `Io` variant above) takes
    /// `decode_error_to_io_error`'s other branch, which has no underlying `io::Error` kind to
    /// preserve and so reports `InvalidData`.
    #[test]
    fn open_parallel_gzip_source_returns_invalid_data_for_a_non_gzip_file() {
        let dir = ScopedTempDir::new("dump-source-not-gzip");
        let path = dir.path().join("not-gzip.gz");
        File::create(&path)
            .unwrap()
            .write_all(b"not a gzip file")
            .unwrap();

        let result = open_parallel_gzip_source(&path);
        let error = result.err().expect("expected an error for invalid gzip framing");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}
