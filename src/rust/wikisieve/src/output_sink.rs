//! Output sink -- ported from `src/rust/wikidata/main.rs`'s already-fixed `OutputSink`/
//! `StreamChunkWriter` design. `File`/`Stdout` are today's plain, non-resumable sinks
//! (used whenever `--resume` isn't passed); `Stream` is the atomic write-then-rename chunked
//! sink used when `--resume` is passed (requires `--chunk-dir`).

use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::extract::Config;
use crate::resume::ResumeState;

/// Chunk file-name prefix shared by the extract mode that writes chunks and the `merge-chunks`
/// mode that reads them back. One constant so a later change can't move one side only, leaving
/// `merge-chunks` silently unable to see the very chunks this binary just wrote.
pub(crate) const DEFAULT_CHUNK_PREFIX: &str = "wikisieve-part-";

pub(crate) enum ProjectedWriter {
    File(BufWriter<File>),
    Stdout(BufWriter<io::Stdout>),
    Stream(StreamChunkWriter),
}

/// The projected JSONL sink, plus -- when `--raw-candidate-output` asked for it -- the raw dump
/// line behind each emitted row (see `FlatRawSink` and `RawChunkStream` for why the raw half sits
/// in two places rather than one).
pub(crate) struct OutputSink {
    writer: ProjectedWriter,
    /// Flat-mode raw capture only. In chunked mode the raw companion lives inside
    /// `StreamChunkWriter` instead, because it has to be sealed by the same rename that seals the
    /// projected chunk it belongs to, under that chunk's own index.
    flat_raw: Option<FlatRawSink>,
}

impl OutputSink {
    pub(crate) fn open(output_path: &Path, raw_candidate_output: Option<&Path>) -> io::Result<Self> {
        let flat_raw = match raw_candidate_output {
            Some(path) => Some(FlatRawSink::create(path)?),
            None => None,
        };
        if output_path.as_os_str() == "-" {
            return Ok(Self {
                writer: ProjectedWriter::Stdout(BufWriter::with_capacity(
                    1024 * 1024,
                    io::stdout(),
                )),
                flat_raw,
            });
        }
        create_parent_dir(output_path)?;
        Ok(Self {
            writer: ProjectedWriter::File(BufWriter::with_capacity(
                1024 * 1024,
                File::create(output_path)?,
            )),
            flat_raw,
        })
    }

    /// The chunked sink, resuming against `resume_state`: the chunks it covers are kept and
    /// counted, and any written after it was saved are dropped (see `StreamChunkWriter::new`).
    pub(crate) fn stream_mode(
        config: &Config,
        resume_state: &ResumeState,
    ) -> io::Result<(Self, usize)> {
        let chunk_dir = config.chunk_dir.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "--chunk-dir is required when --resume is used",
            )
        })?;
        let (writer, restored_rows) = StreamChunkWriter::new(
            chunk_dir,
            config.chunk_prefix.as_str(),
            config.resume_enabled,
            resume_state.next_chunk_index,
            config.raw_candidate_output.is_some(),
        )?;
        Ok((
            Self {
                writer: ProjectedWriter::Stream(writer),
                flat_raw: None,
            },
            restored_rows,
        ))
    }

    /// Writes one emitted row: the projected record, and -- when capture is on -- the raw dump
    /// line it was projected from. One call, so the two artifacts cannot drift apart by a row.
    pub(crate) fn write_row(&mut self, projected_line: &str, raw_line: &[u8]) -> io::Result<()> {
        match &mut self.writer {
            ProjectedWriter::File(writer) => {
                writer.write_all(projected_line.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            ProjectedWriter::Stdout(writer) => {
                writer.write_all(projected_line.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            ProjectedWriter::Stream(writer) => writer.write_row(projected_line, raw_line)?,
        }
        if let Some(raw) = self.flat_raw.as_mut() {
            raw.write_line(raw_line)?;
        }
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> io::Result<()> {
        match &mut self.writer {
            ProjectedWriter::File(writer) => writer.flush()?,
            ProjectedWriter::Stdout(writer) => writer.flush()?,
            ProjectedWriter::Stream(writer) => writer.finish()?,
        }
        if let Some(raw) = self.flat_raw.as_mut() {
            raw.finish()?;
        }
        Ok(())
    }

    /// Finalizes the current chunk (if any rows are pending) so a completed batch's output is
    /// durable on disk before its resume checkpoint is persisted -- see the call site in
    /// `process_dump`. No-op for the non-chunked sinks.
    pub(crate) fn finalize_pending_chunk(&mut self) -> io::Result<()> {
        match &mut self.writer {
            ProjectedWriter::File(_) | ProjectedWriter::Stdout(_) => Ok(()),
            ProjectedWriter::Stream(writer) => writer.finalize_current_chunk(),
        }
    }

    pub(crate) fn next_chunk_index(&self) -> usize {
        match &self.writer {
            ProjectedWriter::File(_) | ProjectedWriter::Stdout(_) => 0,
            ProjectedWriter::Stream(writer) => writer.next_chunk_index,
        }
    }
}

pub(crate) fn create_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Fsyncs `path`'s parent directory after a rename has landed a new name in it, so the rename
/// itself -- not just the renamed file's bytes -- survives a power loss. A file's own `sync_all`
/// guarantees its content is durable before the rename commits to it; without also syncing the
/// directory entry, a crash can still roll the rename back on some filesystems while the target
/// file's bytes remain on disk under its old, discarded name.
///
/// Windows has no equivalent: `File::open` on a directory fails there, and NTFS gives no portable
/// directory-fsync call through `std`, so this is a deliberate no-op off Unix.
#[cfg(unix)]
pub(crate) fn sync_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            File::open(parent)?.sync_all()?;
        }
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn sync_parent_dir(_path: &Path) -> io::Result<()> {
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Raw-candidate capture -- the dump line behind each emitted row, written in the same
// pass so a later field addition never costs a second scan of the 145GB source.
//
// Always gzipped, and always one entity per line with no array delimiters or trailing commas
// (`trim_wikidata_line` has already stripped those by the time a line reaches here). That is not
// a storage choice alone: it makes the artifact a valid `--input` to this same binary, so
// re-deriving a new `projected_fields` entry from the cache is an ordinary run against a smaller
// file rather than a replay mode that would have to be written and kept correct separately.
// ---------------------------------------------------------------------------------------------

/// The gzip level used for every raw-candidate artifact.
///
/// `fast` rather than `default`, and not because the size difference is negligible -- measured
/// over 19,372 real matched candidates (257 MB of entity JSON), deflate level 1 gives 6.1:1 and
/// level 6 gives 7.7:1, so a slower level would take roughly a fifth off the artifact. It is a
/// deliberate trade the other way: this compresses on the extract run's critical path, which is
/// the only run that can produce the cache at all and already takes about an hour on the full
/// dump. Capture at `fast` cost 4-5% of that run's wall time in a bounded real-dump measurement;
/// level 6 roughly doubles the compression work to save ~0.4 GB on a ~2.2 GB artifact.
const RAW_CANDIDATE_COMPRESSION: Compression = Compression::fast();

/// Flat-mode raw capture: one gzip stream at `--raw-candidate-output`.
struct FlatRawSink {
    encoder: GzEncoder<BufWriter<File>>,
}

impl FlatRawSink {
    fn create(path: &Path) -> io::Result<Self> {
        create_parent_dir(path)?;
        Ok(Self {
            encoder: GzEncoder::new(
                BufWriter::with_capacity(1024 * 1024, File::create(path)?),
                RAW_CANDIDATE_COMPRESSION,
            ),
        })
    }

    fn write_line(&mut self, raw_line: &[u8]) -> io::Result<()> {
        self.encoder.write_all(raw_line)?;
        self.encoder.write_all(b"\n")
    }

    /// Writes the gzip trailer and flushes it through to the file. Without the trailer the
    /// artifact decompresses as a truncated stream, so this is the difference between a cache and
    /// an unreadable file -- `run_extract` calls it via `OutputSink::finish`.
    fn finish(&mut self) -> io::Result<()> {
        self.encoder.try_finish()?;
        self.encoder.get_mut().flush()
    }
}

/// Chunked raw capture: the gzip stream companion to the chunk currently being written, sealed
/// by the same `finalize_current_chunk` call that seals that chunk.
struct RawChunkStream {
    encoder: GzEncoder<BufWriter<File>>,
    temp_path: PathBuf,
}

pub(crate) struct StreamChunkWriter {
    chunk_dir: PathBuf,
    chunk_prefix: String,
    /// Whether each chunk gets a raw-candidate companion (`--raw-candidate-output`).
    capture_raw: bool,
    next_chunk_index: usize,
    current_rows: usize,
    current_writer: Option<BufWriter<File>>,
    current_temp_path: Option<PathBuf>,
    current_raw: Option<RawChunkStream>,
}

impl StreamChunkWriter {
    /// Opens the chunk directory for a fresh run, or for a resume against a checkpoint whose
    /// `next_chunk_index` is `checkpoint_next_chunk_index`.
    ///
    /// Each batch seals its chunk and then saves the checkpoint, so a run killed between the two
    /// leaves a chunk the checkpoint does not cover; its lines are rescanned from the checkpoint,
    /// and keeping the chunk would emit its rows twice. A resume therefore keeps exactly the chunks
    /// below the checkpoint's `next_chunk_index` and drops the rest with their companions. With no
    /// checkpoint (index 0) no chunk is covered, since the rescan starts at the dump's first line.
    fn new(
        chunk_dir: &Path,
        chunk_prefix: &str,
        resume_enabled: bool,
        checkpoint_next_chunk_index: usize,
        capture_raw: bool,
    ) -> io::Result<(Self, usize)> {
        fs::create_dir_all(chunk_dir)?;
        let mut existing = iter_existing_chunk_files(chunk_dir, chunk_prefix)?;

        if !resume_enabled {
            for (index, path) in &existing {
                fs::remove_file(path)?;
                remove_raw_companion(chunk_dir, chunk_prefix, *index)?;
            }
            for temp_path in iter_existing_temp_files(chunk_dir, chunk_prefix)? {
                fs::remove_file(temp_path)?;
            }
            // A companion whose own chunk is already gone: an interrupted run can leave one
            // behind, since the companion is renamed into place first (see
            // `finalize_current_chunk`). A fresh start owns this prefix outright, so nothing
            // here is another run's to keep.
            for (_, path) in iter_existing_raw_chunk_files(chunk_dir, chunk_prefix)? {
                fs::remove_file(path)?;
            }
            existing.clear();
        }

        let mut restored_rows = 0usize;
        if resume_enabled {
            let (covered, uncovered): (Vec<_>, Vec<_>) = existing
                .into_iter()
                .partition(|(index, _)| *index < checkpoint_next_chunk_index);
            for (index, path) in &uncovered {
                eprintln!(
                    "[wikisieve] resume dropping {}: written after the last checkpoint \
                     (next_chunk_index={checkpoint_next_chunk_index})",
                    path.display()
                );
                fs::remove_file(path)?;
                remove_raw_companion(chunk_dir, chunk_prefix, *index)?;
            }
            existing = covered;
            for (_, path) in &existing {
                restored_rows += count_non_empty_lines(path)?;
            }
        }

        let next_chunk_index = existing.last().map_or(1usize, |(idx, _)| idx + 1);
        Ok((
            Self {
                chunk_dir: chunk_dir.to_path_buf(),
                chunk_prefix: chunk_prefix.to_string(),
                capture_raw,
                next_chunk_index,
                current_rows: 0,
                current_writer: None,
                current_temp_path: None,
                current_raw: None,
            },
            restored_rows,
        ))
    }

    fn write_row(&mut self, projected_line: &str, raw_line: &[u8]) -> io::Result<()> {
        self.ensure_chunk_open()?;
        let writer = self
            .current_writer
            .as_mut()
            .ok_or_else(|| io::Error::other("stream chunk writer missing active chunk handle"))?;
        writer.write_all(projected_line.as_bytes())?;
        writer.write_all(b"\n")?;
        if let Some(raw) = self.current_raw.as_mut() {
            raw.encoder.write_all(raw_line)?;
            raw.encoder.write_all(b"\n")?;
        }
        self.current_rows += 1;
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        self.finalize_current_chunk()
    }

    fn ensure_chunk_open(&mut self) -> io::Result<()> {
        if self.current_writer.is_some() {
            return Ok(());
        }
        let temp_name = format!("{}{:06}.tmp", self.chunk_prefix, self.next_chunk_index);
        let temp_path = self.chunk_dir.join(temp_name);
        let file = File::create(&temp_path)?;
        self.current_writer = Some(BufWriter::with_capacity(1024 * 1024, file));
        self.current_temp_path = Some(temp_path);
        if self.capture_raw {
            let raw_temp_path =
                raw_chunk_temp_path(&self.chunk_dir, &self.chunk_prefix, self.next_chunk_index);
            let raw_file = File::create(&raw_temp_path)?;
            self.current_raw = Some(RawChunkStream {
                encoder: GzEncoder::new(
                    BufWriter::with_capacity(1024 * 1024, raw_file),
                    RAW_CANDIDATE_COMPRESSION,
                ),
                temp_path: raw_temp_path,
            });
        }
        self.current_rows = 0;
        Ok(())
    }

    fn finalize_current_chunk(&mut self) -> io::Result<()> {
        let Some(mut writer) = self.current_writer.take() else {
            return Ok(());
        };
        writer.flush()?;
        // Synced before anything is renamed onto the chunk's final name: that rename is what a
        // resume counts and what `merge-chunks` consumes, so it must never commit to a name whose
        // bytes are still only in the OS page cache -- a power loss right after the rename would
        // otherwise leave a resume state (or a merge) counting a chunk that a crash can still
        // shorten.
        writer.get_ref().sync_all()?;
        let raw_temp_path = self.finish_current_raw_stream()?;

        let Some(temp_path) = self.current_temp_path.take() else {
            return Ok(());
        };

        if self.current_rows == 0 {
            fs::remove_file(temp_path)?;
            if let Some(raw_temp_path) = raw_temp_path {
                fs::remove_file(raw_temp_path)?;
            }
            return Ok(());
        }

        // The companion is renamed into place *before* the chunk it belongs to. The chunk's final
        // name is what a resume counts and what `merge-chunks` consumes, so it is the commit
        // point: a kill between the two renames can leave a companion with no chunk (harmless,
        // and swept by the next fresh start or overwritten by the retried chunk index) but never
        // a chunk with no companion, which is the direction that would silently shorten the
        // raw-candidate cache.
        if let Some(raw_temp_path) = raw_temp_path {
            let raw_final_path =
                raw_chunk_path(&self.chunk_dir, &self.chunk_prefix, self.next_chunk_index);
            fs::rename(raw_temp_path, &raw_final_path)?;
            sync_parent_dir(&raw_final_path)?;
        }

        let final_name = format!("{}{:06}.jsonl", self.chunk_prefix, self.next_chunk_index);
        let final_path = self.chunk_dir.join(final_name);
        fs::rename(temp_path, &final_path)?;
        sync_parent_dir(&final_path)?;
        self.next_chunk_index += 1;
        self.current_rows = 0;
        Ok(())
    }

    /// Writes the current companion's gzip trailer, flushes it and syncs it to disk before
    /// returning its temp path so the caller can rename or discard it -- the same durability
    /// requirement as the chunk it belongs beside. `None` when capture is off.
    fn finish_current_raw_stream(&mut self) -> io::Result<Option<PathBuf>> {
        let Some(mut raw) = self.current_raw.take() else {
            return Ok(None);
        };
        raw.encoder.try_finish()?;
        raw.encoder.get_mut().flush()?;
        raw.encoder.get_mut().get_ref().sync_all()?;
        Ok(Some(raw.temp_path))
    }
}

/// The raw-candidate companion to chunk `index`: `<prefix><index>.raw.jsonl.gz`.
///
/// A distinct extension, not a distinct prefix, so `iter_existing_chunk_files` (which selects on
/// `.jsonl`) keeps seeing exactly the projected chunks, and so the pairing between a chunk and its
/// companion is derivable from the index alone rather than stored anywhere.
pub(crate) fn raw_chunk_path(chunk_dir: &Path, chunk_prefix: &str, index: usize) -> PathBuf {
    chunk_dir.join(format!("{chunk_prefix}{index:06}.raw.jsonl.gz"))
}

/// The in-progress companion's name. Ends in `.tmp`, so `iter_existing_temp_files` sweeps an
/// abandoned one on the same paths that already sweep an abandoned chunk.
fn raw_chunk_temp_path(chunk_dir: &Path, chunk_prefix: &str, index: usize) -> PathBuf {
    chunk_dir.join(format!("{chunk_prefix}{index:06}.raw.jsonl.gz.tmp"))
}

fn remove_raw_companion(chunk_dir: &Path, chunk_prefix: &str, index: usize) -> io::Result<()> {
    let path = raw_chunk_path(chunk_dir, chunk_prefix, index);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        // A run that captured nothing has no companions to remove, and neither does a chunk
        // written before capture was switched on.
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

pub(crate) fn iter_existing_chunk_files(
    chunk_dir: &Path,
    chunk_prefix: &str,
) -> io::Result<Vec<(usize, PathBuf)>> {
    let mut paths: Vec<(usize, PathBuf)> = Vec::new();
    for entry in fs::read_dir(chunk_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with(chunk_prefix)
            || !Path::new(name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("jsonl"))
        {
            continue;
        }
        let index_text = &name[chunk_prefix.len()..name.len() - ".jsonl".len()];
        let Ok(index) = index_text.parse::<usize>() else {
            continue;
        };
        paths.push((index, path));
    }
    paths.sort_by_key(|(index, _)| *index);
    Ok(paths)
}

/// The raw-candidate companions present in `chunk_dir`, by chunk index. The counterpart to
/// `iter_existing_chunk_files`, selecting the `.raw.jsonl.gz` names that one deliberately skips.
pub(crate) fn iter_existing_raw_chunk_files(
    chunk_dir: &Path,
    chunk_prefix: &str,
) -> io::Result<Vec<(usize, PathBuf)>> {
    const RAW_SUFFIX: &str = ".raw.jsonl.gz";

    let mut paths: Vec<(usize, PathBuf)> = Vec::new();
    for entry in fs::read_dir(chunk_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with(chunk_prefix) || !name.ends_with(RAW_SUFFIX) {
            continue;
        }
        let index_text = &name[chunk_prefix.len()..name.len() - RAW_SUFFIX.len()];
        let Ok(index) = index_text.parse::<usize>() else {
            continue;
        };
        paths.push((index, path));
    }
    paths.sort_by_key(|(index, _)| *index);
    Ok(paths)
}

pub(crate) fn iter_existing_temp_files(
    chunk_dir: &Path,
    chunk_prefix: &str,
) -> io::Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(chunk_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if name.starts_with(chunk_prefix)
            && Path::new(name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("tmp"))
        {
            paths.push(path);
        }
    }
    Ok(paths)
}

fn count_non_empty_lines(path: &Path) -> io::Result<usize> {
    let file = File::open(path)?;
    count_non_empty_lines_in(BufReader::new(file))
}

/// `count_non_empty_lines` through a gzip decoder -- what a raw-candidate artifact has to be read
/// through to be counted at all, since its row count is not recoverable from its byte length.
pub(crate) fn count_non_empty_gz_lines(path: &Path) -> io::Result<usize> {
    let file = File::open(path)?;
    count_non_empty_lines_in(BufReader::with_capacity(
        1024 * 1024,
        flate2::read::MultiGzDecoder::new(file),
    ))
}

fn count_non_empty_lines_in(reader: impl BufRead) -> io::Result<usize> {
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        chunk_path, config_for, entry_names, path_arg, read_file, read_gz_file, seed_chunk,
        seed_raw_companion, temp_chunk_path, ScopedTempDir, TEST_PREFIX,
    };

    /// Writes a projected row with no raw-candidate line behind it. Most tests here are about
    /// chunk or sink lifecycle rather than raw capture, and with capture off the raw argument is
    /// never read -- naming that once keeps those tests reading as they did before
    /// `--raw-candidate-output` gave `write_row` its second argument.
    impl StreamChunkWriter {
        fn write_line(&mut self, projected_line: &str) -> io::Result<()> {
            self.write_row(projected_line, b"")
        }
    }

    impl OutputSink {
        fn write_line(&mut self, projected_line: &str) -> io::Result<()> {
            self.write_row(projected_line, b"")
        }
    }

    // -----------------------------------------------------------------------------------------
    // `sync_parent_dir`: the directory fsync every rename in this crate's chunk, merge
    // and resume paths follows. A unit test can't observe an fsync's effect on a real power loss,
    // but it can pin the one behaviour that would otherwise regress silently: the call succeeds
    // for a rename's actual destination, on whichever platform runs the suite.
    // -----------------------------------------------------------------------------------------

    #[test]
    fn sync_parent_dir_succeeds_for_a_file_with_a_real_parent_directory() {
        let dir = ScopedTempDir::new("sync-parent-dir");
        let path = dir.path().join("some-file.txt");
        fs::write(&path, b"content").unwrap();

        sync_parent_dir(&path).unwrap();
    }

    #[test]
    fn sync_parent_dir_is_a_no_op_for_a_bare_file_name_with_no_parent() {
        sync_parent_dir(Path::new("bare-file-name.txt")).unwrap();
    }

    // -----------------------------------------------------------------------------------------
    // StreamChunkWriter lifecycle: open -> write -> finalize-and-rename.
    //
    // Every assertion below reads the chunk file back and checks its bytes. "The call returned
    // Ok" is exactly what the earlier resume incident would have passed -- three compounding bugs in this
    // class of resume/chunk logic, none of which surfaced as an error.
    // -----------------------------------------------------------------------------------------

    #[test]
    fn stream_chunk_writer_finalizes_a_chunk_with_exact_written_content() {
        let dir = ScopedTempDir::new("chunk-finalize");
        let (mut writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, false).unwrap();
        assert_eq!(restored_rows, 0);
        assert_eq!(writer.next_chunk_index, 1);

        writer.write_line(r#"{"id":"Q1"}"#).unwrap();
        writer.write_line(r#"{"id":"Q2"}"#).unwrap();

        // Before finalization the rows live under the `.tmp` name only: nothing is visible under
        // the final name a resume would count, which is what makes the rename the commit point.
        assert_eq!(entry_names(dir.path()), vec!["part-000001.tmp"]);

        writer.finish().unwrap();

        assert_eq!(entry_names(dir.path()), vec!["part-000001.jsonl"]);
        assert_eq!(
            read_file(&chunk_path(dir.path(), 1)),
            "{\"id\":\"Q1\"}\n{\"id\":\"Q2\"}\n"
        );
        assert_eq!(writer.next_chunk_index, 2);
    }

    #[test]
    fn stream_chunk_writer_numbers_successive_chunks_and_leaves_earlier_ones_intact() {
        let dir = ScopedTempDir::new("chunk-sequence");
        let (mut writer, _) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, false).unwrap();

        writer.write_line("a").unwrap();
        writer.write_line("b").unwrap();
        writer.finalize_current_chunk().unwrap();
        writer.write_line("c").unwrap();
        writer.finish().unwrap();

        assert_eq!(
            entry_names(dir.path()),
            vec!["part-000001.jsonl", "part-000002.jsonl"]
        );
        assert_eq!(read_file(&chunk_path(dir.path(), 1)), "a\nb\n");
        assert_eq!(read_file(&chunk_path(dir.path(), 2)), "c\n");
        assert_eq!(writer.next_chunk_index, 3);
    }

    #[test]
    fn stream_chunk_writer_discards_an_opened_chunk_that_took_no_rows() {
        // The `current_rows == 0` branch: `process_dump` calls `finalize_pending_chunk` once per
        // batch, so a batch that emitted nothing must leave no zero-length chunk behind and must
        // not consume a chunk index.
        let dir = ScopedTempDir::new("chunk-empty");
        let (mut writer, _) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, false).unwrap();

        writer.ensure_chunk_open().unwrap();
        assert_eq!(entry_names(dir.path()), vec!["part-000001.tmp"]);

        writer.finalize_current_chunk().unwrap();

        assert!(entry_names(dir.path()).is_empty());
        assert_eq!(writer.next_chunk_index, 1);
    }

    #[test]
    fn stream_chunk_writer_finalize_without_an_open_chunk_is_a_no_op() {
        let dir = ScopedTempDir::new("chunk-idempotent-finalize");
        let (mut writer, _) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, false).unwrap();

        writer.write_line("a").unwrap();
        writer.finalize_current_chunk().unwrap();
        // Repeated finalization -- the per-batch call in `process_dump` when a batch emitted
        // nothing, then `finish()` at the end of the run -- must not touch the sealed chunk.
        writer.finalize_current_chunk().unwrap();
        writer.finish().unwrap();

        assert_eq!(entry_names(dir.path()), vec!["part-000001.jsonl"]);
        assert_eq!(read_file(&chunk_path(dir.path(), 1)), "a\n");
        assert_eq!(writer.next_chunk_index, 2);
    }

    #[test]
    fn stream_chunk_writer_reopens_a_chunk_after_finalizing_an_empty_one() {
        let dir = ScopedTempDir::new("chunk-empty-then-rows");
        let (mut writer, _) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, false).unwrap();

        writer.ensure_chunk_open().unwrap();
        writer.finalize_current_chunk().unwrap();
        writer.write_line("a").unwrap();
        writer.finish().unwrap();

        // The discarded empty chunk released index 1, so the first chunk with rows takes it.
        assert_eq!(entry_names(dir.path()), vec!["part-000001.jsonl"]);
        assert_eq!(read_file(&chunk_path(dir.path(), 1)), "a\n");
    }

    // -----------------------------------------------------------------------------------------
    // Startup: fresh-start cleanup, resume-restore row counting, and dropping the chunks a
    // checkpoint does not cover.
    //
    // These are the paths that incident ran through in the sibling crate: what a run decides
    // about chunks it did not write is what destroyed a durable cache there, so each test below
    // asserts the surviving files' content, not just their presence.
    // -----------------------------------------------------------------------------------------

    #[test]
    fn stream_chunk_writer_fresh_start_removes_existing_chunks_and_stale_temp_files() {
        let dir = ScopedTempDir::new("chunk-fresh-start");
        seed_chunk(dir.path(), 1, "stale-a\n");
        seed_chunk(dir.path(), 2, "stale-b\n");
        fs::write(temp_chunk_path(dir.path(), 3), "half-written row").unwrap();
        // Neither of these carries the chunk prefix, so a fresh start must leave them alone --
        // a chunk directory is not necessarily the run's exclusive property.
        fs::write(dir.path().join("other-000001.jsonl"), "not ours\n").unwrap();
        fs::write(dir.path().join("notes.txt"), "not ours either\n").unwrap();

        let (mut writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, false).unwrap();

        assert_eq!(restored_rows, 0);
        assert_eq!(writer.next_chunk_index, 1);
        assert_eq!(
            entry_names(dir.path()),
            vec!["notes.txt", "other-000001.jsonl"]
        );

        writer.write_line("fresh").unwrap();
        writer.finish().unwrap();

        // Index 1 is reused and the stale content is gone, not appended to.
        assert_eq!(read_file(&chunk_path(dir.path(), 1)), "fresh\n");
    }

    #[test]
    fn stream_chunk_writer_resume_counts_existing_rows_and_continues_numbering() {
        let dir = ScopedTempDir::new("chunk-resume-restore");
        seed_chunk(dir.path(), 1, "a\nb\n");
        seed_chunk(dir.path(), 2, "c\n");

        let (mut writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, true, 3, false).unwrap();

        assert_eq!(restored_rows, 3);
        assert_eq!(writer.next_chunk_index, 3);

        writer.write_line("d").unwrap();
        writer.finish().unwrap();

        // The two restored chunks keep their exact content: a resume adds a chunk, it never
        // rewrites or truncates one it counted.
        assert_eq!(read_file(&chunk_path(dir.path(), 1)), "a\nb\n");
        assert_eq!(read_file(&chunk_path(dir.path(), 2)), "c\n");
        assert_eq!(read_file(&chunk_path(dir.path(), 3)), "d\n");
    }

    #[test]
    fn stream_chunk_writer_resume_row_count_ignores_blank_lines() {
        let dir = ScopedTempDir::new("chunk-resume-blank-lines");
        seed_chunk(dir.path(), 1, "a\n\n   \nb\n");

        let (_writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, true, 2, false).unwrap();

        assert_eq!(restored_rows, 2);
    }

    #[test]
    fn stream_chunk_writer_resume_with_no_existing_chunks_starts_from_index_one() {
        let dir = ScopedTempDir::new("chunk-resume-empty-dir");

        let (writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, true, 1, false).unwrap();

        assert_eq!(restored_rows, 0);
        assert_eq!(writer.next_chunk_index, 1);
    }

    #[test]
    fn stream_chunk_writer_resume_numbering_follows_the_highest_existing_index() {
        let dir = ScopedTempDir::new("chunk-resume-gap");
        seed_chunk(dir.path(), 1, "a\n");
        seed_chunk(dir.path(), 5, "b\n");
        // Resume, unlike a fresh start, does not sweep stale `.tmp` files. That is harmless --
        // only `.jsonl` files are ever counted or renamed over -- but assert it so the split in
        // behaviour between the two startup paths is deliberate rather than incidental.
        fs::write(temp_chunk_path(dir.path(), 6), "abandoned").unwrap();

        let (writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, true, 6, false).unwrap();

        assert_eq!(restored_rows, 2);
        assert_eq!(writer.next_chunk_index, 6);
        assert!(temp_chunk_path(dir.path(), 6).exists());
    }

    #[test]
    fn stream_chunk_writer_resume_drops_a_chunk_sealed_after_the_checkpoint() {
        // A run killed after sealing chunk 2 and before saving the checkpoint that covers it: the
        // checkpoint still says chunk 2 is next, so batch 2 is rescanned and chunk 2 must go, or
        // its rows would be emitted twice.
        let dir = ScopedTempDir::new("chunk-uncovered");
        seed_chunk(dir.path(), 1, "a\nb\n");
        seed_chunk(dir.path(), 2, "c\n");

        let (mut writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, true, 2, false).unwrap();

        assert_eq!(restored_rows, 2);
        assert_eq!(writer.next_chunk_index, 2);
        assert_eq!(entry_names(dir.path()), vec!["part-000001.jsonl"]);

        writer.write_line("c").unwrap();
        writer.finish().unwrap();

        assert_eq!(read_file(&chunk_path(dir.path(), 1)), "a\nb\n");
        assert_eq!(read_file(&chunk_path(dir.path(), 2)), "c\n");
    }

    #[test]
    fn stream_chunk_writer_resume_with_no_checkpoint_drops_every_chunk() {
        // No checkpoint means the rescan starts at the dump's first line, so no chunk is covered.
        let dir = ScopedTempDir::new("chunk-no-checkpoint");
        seed_chunk(dir.path(), 1, "a\n");

        let (writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, true, 0, false).unwrap();

        assert_eq!(restored_rows, 0);
        assert_eq!(writer.next_chunk_index, 1);
        assert!(entry_names(dir.path()).is_empty());
    }

    // -----------------------------------------------------------------------------------------
    // Raw-candidate capture: the companion written beside each sealed chunk.
    //
    // The companion is the cache that replaces a full-dump rescan, so the tests that matter are
    // the ones about what happens to it when a run is interrupted, restarted or replayed -- a
    // companion silently missing a chunk's worth of rows produces a cache that looks complete and
    // reprojects short.
    // -----------------------------------------------------------------------------------------

    #[test]
    fn stream_chunk_writer_seals_a_raw_companion_beside_each_chunk() {
        let dir = ScopedTempDir::new("raw-companion-seal");
        let (mut writer, _) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, true).unwrap();

        writer.write_row("projected-a", b"raw-a").unwrap();
        writer.write_row("projected-b", b"raw-b").unwrap();

        // Both halves are still under `.tmp` names: the companion is not visible under its final
        // name until the chunk it belongs to is sealed.
        assert_eq!(
            entry_names(dir.path()),
            vec!["part-000001.raw.jsonl.gz.tmp", "part-000001.tmp"]
        );

        writer.finalize_current_chunk().unwrap();
        writer.write_row("projected-c", b"raw-c").unwrap();
        writer.finish().unwrap();

        assert_eq!(
            entry_names(dir.path()),
            vec![
                "part-000001.jsonl",
                "part-000001.raw.jsonl.gz",
                "part-000002.jsonl",
                "part-000002.raw.jsonl.gz",
            ]
        );
        assert_eq!(
            read_file(&chunk_path(dir.path(), 1)),
            "projected-a\nprojected-b\n"
        );
        assert_eq!(
            read_gz_file(&raw_chunk_path(dir.path(), TEST_PREFIX, 1)),
            "raw-a\nraw-b\n"
        );
        assert_eq!(read_file(&chunk_path(dir.path(), 2)), "projected-c\n");
        assert_eq!(
            read_gz_file(&raw_chunk_path(dir.path(), TEST_PREFIX, 2)),
            "raw-c\n"
        );
    }

    #[test]
    fn stream_chunk_writer_writes_no_companion_when_capture_is_off() {
        let dir = ScopedTempDir::new("raw-companion-off");
        let (mut writer, _) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, false).unwrap();

        // The raw line is still supplied by `process_dump` on every row; capture being off is
        // what decides whether it is written, not the caller withholding it.
        writer.write_row("projected-a", b"raw-a").unwrap();
        writer.finish().unwrap();

        assert_eq!(entry_names(dir.path()), vec!["part-000001.jsonl"]);
    }

    #[test]
    fn stream_chunk_writer_discards_the_companion_of_a_chunk_that_took_no_rows() {
        let dir = ScopedTempDir::new("raw-companion-empty");
        let (mut writer, _) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, true).unwrap();

        writer.ensure_chunk_open().unwrap();
        writer.finalize_current_chunk().unwrap();

        // Neither half survives, and neither consumes the index: `process_dump` opens a chunk per
        // batch, and a batch that emitted nothing must leave the directory as it found it.
        assert!(entry_names(dir.path()).is_empty());
        assert_eq!(writer.next_chunk_index, 1);
    }

    #[test]
    fn stream_chunk_writer_fresh_start_removes_companions_including_an_orphaned_one() {
        let dir = ScopedTempDir::new("raw-companion-fresh-start");
        seed_chunk(dir.path(), 1, "stale\n");
        seed_raw_companion(dir.path(), 1, "stale-raw\n");
        // A companion whose chunk never got renamed -- the window between the two renames in
        // `finalize_current_chunk`. It has no chunk to be found through, so the sweep has to find
        // it directly or it outlives every later run.
        seed_raw_companion(dir.path(), 2, "orphan-raw\n");
        fs::write(dir.path().join("other-000001.raw.jsonl.gz"), "not ours").unwrap();

        let (mut writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, false, 0, true).unwrap();

        assert_eq!(restored_rows, 0);
        assert_eq!(entry_names(dir.path()), vec!["other-000001.raw.jsonl.gz"]);

        writer.write_row("fresh", b"fresh-raw").unwrap();
        writer.finish().unwrap();

        assert_eq!(
            read_gz_file(&raw_chunk_path(dir.path(), TEST_PREFIX, 1)),
            "fresh-raw\n"
        );
    }

    #[test]
    fn stream_chunk_writer_resume_keeps_the_companions_of_the_chunks_it_counted() {
        let dir = ScopedTempDir::new("raw-companion-resume");
        seed_chunk(dir.path(), 1, "a\nb\n");
        seed_raw_companion(dir.path(), 1, "raw-a\nraw-b\n");

        let (mut writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, true, 2, true).unwrap();

        assert_eq!(restored_rows, 2);
        writer.write_row("c", b"raw-c").unwrap();
        writer.finish().unwrap();

        // The restored companion is untouched and the resumed one is added beside it: the cache
        // grows across a resume exactly as the projected chunks do.
        assert_eq!(
            read_gz_file(&raw_chunk_path(dir.path(), TEST_PREFIX, 1)),
            "raw-a\nraw-b\n"
        );
        assert_eq!(
            read_gz_file(&raw_chunk_path(dir.path(), TEST_PREFIX, 2)),
            "raw-c\n"
        );
    }

    #[test]
    fn stream_chunk_writer_resume_drops_the_companion_of_an_uncovered_chunk() {
        let dir = ScopedTempDir::new("raw-companion-uncovered");
        seed_chunk(dir.path(), 1, "a\n");
        seed_raw_companion(dir.path(), 1, "raw-a\n");
        seed_chunk(dir.path(), 2, "b\n");
        seed_raw_companion(dir.path(), 2, "raw-b\n");

        let (mut writer, restored_rows) =
            StreamChunkWriter::new(dir.path(), TEST_PREFIX, true, 2, true).unwrap();

        assert_eq!(restored_rows, 1);
        assert_eq!(
            entry_names(dir.path()),
            vec!["part-000001.jsonl", "part-000001.raw.jsonl.gz"]
        );

        writer.write_row("b", b"raw-b").unwrap();
        writer.finish().unwrap();

        // The rescanned batch's rows land in both halves once, not twice.
        assert_eq!(
            read_gz_file(&raw_chunk_path(dir.path(), TEST_PREFIX, 2)),
            "raw-b\n"
        );
    }

    #[test]
    fn iter_existing_raw_chunk_files_selects_only_prefixed_numeric_companions() {
        let dir = ScopedTempDir::new("iter-raw-chunks-filter");
        seed_raw_companion(dir.path(), 1, "a\n");
        seed_raw_companion(dir.path(), 2, "b\n");
        seed_chunk(dir.path(), 3, "projected, not a companion\n");
        fs::write(
            dir.path().join("part-000004.raw.jsonl.gz.tmp"),
            "in progress",
        )
        .unwrap();
        fs::write(dir.path().join("part-abc.raw.jsonl.gz"), "unnumbered").unwrap();
        fs::write(dir.path().join("other-000001.raw.jsonl.gz"), "wrong prefix").unwrap();

        let found = iter_existing_raw_chunk_files(dir.path(), TEST_PREFIX).unwrap();

        assert_eq!(
            found,
            vec![
                (1, raw_chunk_path(dir.path(), TEST_PREFIX, 1)),
                (2, raw_chunk_path(dir.path(), TEST_PREFIX, 2)),
            ]
        );
    }

    #[test]
    fn count_non_empty_gz_lines_counts_rows_through_the_decoder() {
        let dir = ScopedTempDir::new("count-gz-lines");
        seed_raw_companion(dir.path(), 1, "a\n\n  \t \nb\n");

        assert_eq!(
            count_non_empty_gz_lines(&raw_chunk_path(dir.path(), TEST_PREFIX, 1)).unwrap(),
            2
        );
    }

    #[test]
    fn iter_existing_chunk_files_selects_only_prefixed_numeric_jsonl_files() {
        let dir = ScopedTempDir::new("iter-chunks-filter");
        seed_chunk(dir.path(), 1, "a\n");
        seed_chunk(dir.path(), 2, "b\n");
        fs::write(temp_chunk_path(dir.path(), 3), "in progress").unwrap();
        fs::write(dir.path().join("part-abc.jsonl"), "unnumbered\n").unwrap();
        fs::write(dir.path().join("other-000001.jsonl"), "wrong prefix\n").unwrap();
        // A directory that would otherwise match the name pattern: the `is_file` guard.
        fs::create_dir(dir.path().join("part-000004.jsonl")).unwrap();

        let found = iter_existing_chunk_files(dir.path(), TEST_PREFIX).unwrap();

        assert_eq!(
            found,
            vec![
                (1, chunk_path(dir.path(), 1)),
                (2, chunk_path(dir.path(), 2)),
            ]
        );
    }

    #[test]
    fn iter_existing_chunk_files_orders_by_numeric_index_not_file_name() {
        // `next_chunk_index` comes from the *last* entry, so the ordering has to be numeric.
        // These unpadded names sort as "10" < "7" by name and 7 < 10 by index.
        let dir = ScopedTempDir::new("iter-chunks-order");
        fs::write(dir.path().join("part-10.jsonl"), "ten\n").unwrap();
        fs::write(dir.path().join("part-7.jsonl"), "seven\n").unwrap();

        let found = iter_existing_chunk_files(dir.path(), TEST_PREFIX).unwrap();

        assert_eq!(
            found.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![7, 10]
        );
    }

    #[test]
    fn iter_existing_temp_files_selects_only_prefixed_temp_files() {
        let dir = ScopedTempDir::new("iter-temps-filter");
        fs::write(temp_chunk_path(dir.path(), 1), "a").unwrap();
        fs::write(temp_chunk_path(dir.path(), 2), "b").unwrap();
        seed_chunk(dir.path(), 3, "finalized\n");
        fs::write(dir.path().join("other-000001.tmp"), "wrong prefix").unwrap();

        let mut found = iter_existing_temp_files(dir.path(), TEST_PREFIX).unwrap();
        found.sort();

        assert_eq!(
            found,
            vec![
                temp_chunk_path(dir.path(), 1),
                temp_chunk_path(dir.path(), 2)
            ]
        );
    }

    #[test]
    fn count_non_empty_lines_counts_rows_without_a_trailing_newline() {
        let dir = ScopedTempDir::new("count-lines");
        let path = dir.path().join("rows.jsonl");
        fs::write(&path, "a\n\n  \t \nb").unwrap();

        assert_eq!(count_non_empty_lines(&path).unwrap(), 2);
    }

    #[test]
    fn count_non_empty_lines_of_an_empty_file_is_zero() {
        let dir = ScopedTempDir::new("count-lines-empty");
        let path = dir.path().join("rows.jsonl");
        fs::write(&path, "").unwrap();

        assert_eq!(count_non_empty_lines(&path).unwrap(), 0);
    }

    // -----------------------------------------------------------------------------------------
    // OutputSink: what each variant actually puts on disk.
    // -----------------------------------------------------------------------------------------

    #[test]
    fn output_sink_file_mode_writes_newline_terminated_lines_into_a_created_directory() {
        let dir = ScopedTempDir::new("sink-file");
        let path = dir.path().join("nested").join("deep").join("out.jsonl");

        let mut sink = OutputSink::open(&path, None).unwrap();
        sink.write_line(r#"{"id":"Q1"}"#).unwrap();
        // Chunk finalization is a no-op for a non-chunked sink: `process_dump` calls it once per
        // batch regardless of which sink it holds.
        sink.finalize_pending_chunk().unwrap();
        sink.write_line(r#"{"id":"Q2"}"#).unwrap();
        sink.finish().unwrap();

        assert_eq!(read_file(&path), "{\"id\":\"Q1\"}\n{\"id\":\"Q2\"}\n");
        assert_eq!(sink.next_chunk_index(), 0);
    }

    #[test]
    fn output_sink_stream_mode_restores_existing_rows_and_appends_a_new_chunk() {
        let dir = ScopedTempDir::new("sink-stream");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "already-durable\n");

        let chunk_arg = path_arg(&chunk_dir);
        let state_arg = path_arg(&dir.path().join("resume-state.json"));
        let config = config_for(&[
            "--input",
            "unused.jsonl.gz",
            "--spec",
            "unused-spec.json",
            "--output",
            "unused.jsonl",
            "--resume",
            "--chunk-dir",
            chunk_arg.as_str(),
            "--state-path",
            state_arg.as_str(),
            "--chunk-prefix",
            TEST_PREFIX,
        ]);

        let checkpoint = ResumeState {
            next_chunk_index: 2,
            ..ResumeState::default()
        };
        let (mut sink, restored_rows) = OutputSink::stream_mode(&config, &checkpoint).unwrap();
        assert_eq!(restored_rows, 1);

        sink.write_line("new-row").unwrap();
        sink.finalize_pending_chunk().unwrap();

        assert_eq!(read_file(&chunk_path(&chunk_dir, 1)), "already-durable\n");
        assert_eq!(read_file(&chunk_path(&chunk_dir, 2)), "new-row\n");
        assert_eq!(sink.next_chunk_index(), 3);
    }

}
