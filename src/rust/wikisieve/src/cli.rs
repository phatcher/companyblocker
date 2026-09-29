//! Small argument-parsing helpers shared by every subcommand's own `Config::from_args`, plus the
//! combined `--help` text. Each subcommand still owns its own flag loop and its own `Config`
//! type; only the bits that would otherwise be copied three times live here.

use std::io;
use std::path::PathBuf;

pub(crate) fn require_arg(value: Option<PathBuf>, flag: &str) -> io::Result<PathBuf> {
    value.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("missing required argument: {flag}"),
        )
    })
}

pub(crate) fn parse_usize_arg(raw_value: &str, flag: &str) -> io::Result<usize> {
    raw_value.parse::<usize>().map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid {flag} value '{raw_value}': {error}"),
        )
    })
}

pub(crate) fn next_arg_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> io::Result<String> {
    args.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("missing value for {flag}"),
        )
    })
}

pub(crate) fn print_usage() {
    eprintln!(
        "Usage: wikisieve --input <path>|- --spec <path> --output <path> [--summary-json <path>] \
         [--max-rows N] [--max-records N] [--output-mode ids|jsonl] [--resume] \
         [--state-path <path>] [--chunk-dir <path>] \
         [--chunk-prefix <prefix>] [--raw-candidate-output <path>]\n\
         \n\
         \x20      wikisieve merge-chunks --chunk-dir <path> --output <path> \
         [--chunk-prefix <prefix>] [--summary-json <path>] [--state-path <path>] \
         [--expected-rows N] [--raw-candidate-output <path>]\n\
         \n\
         merge-chunks assembles a completed --resume run's chunk directory into one flat file at \
         --output, verifying the merged row count against the expected total first. That total \
         comes from --expected-rows, else --summary-json's records_emitted, else --state-path's; \
         at least one is required.\n\
         \n\
         --raw-candidate-output turns on the raw-candidate cache: the dump line behind each \
         emitted row, gzipped, written in the same pass. Without --resume it lands at that path \
         directly; with --resume each chunk gets a companion and merge-chunks assembles them, \
         the same split --output already follows. The artifact is itself a valid --input, so a \
         later spec can re-derive new projected_fields from it instead of rescanning the dump.\n\
         \n\
         \x20      wikisieve profile --input <path> --spec <path> --output <report.json> \
         [--max-rows N] [--gap-min-count N]\n\
         \n\
         profile is a schema-discovery/diagnostic pass, deliberately separate from and slower \
         than extract: for every matched candidate it classifies *every* present claim property \
         (not just the ones --spec projects) into one of the four projected_fields shapes, or \
         'complex' for anything else (quantity, globe-coordinate, novalue/somevalue snaks, or a \
         property whose claims disagree on a shape). --output gets a JSON report of per-property \
         frequency+shape counts, plus a gap report: simple-shaped properties seen on at least \
         --gap-min-count candidates (default 1) that no projected_fields entry names yet -- \
         directly answering 'what am I not extracting' instead of diffing two lists by hand."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_usize_arg_accepts_valid_numbers() {
        assert_eq!(parse_usize_arg("0", "--max-rows").unwrap(), 0);
        assert_eq!(parse_usize_arg("42", "--max-rows").unwrap(), 42);
    }

    #[test]
    fn parse_usize_arg_rejects_non_numeric_value() {
        let error = parse_usize_arg("abc", "--max-rows").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--max-rows"));
        assert!(error.to_string().contains("abc"));
    }

    #[test]
    fn parse_usize_arg_rejects_negative_value() {
        assert!(parse_usize_arg("-1", "--max-records").is_err());
    }
}
