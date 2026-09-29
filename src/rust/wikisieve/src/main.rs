use std::env;
use std::io;

use wikisieve::extract::{self, Config};
use wikisieve::merge_chunks::{self, MergeConfig};
use wikisieve::profile::{self, ProfileConfig};
use wikisieve::version;

/// The one subcommand this binary recognizes. Anything else in first position falls through to
/// the extract mode's flat flag parsing, so every pre-existing `wikisieve --input ...` invocation
/// keeps working unchanged and an unknown flag still reports itself as an unknown flag rather
/// than as an unknown subcommand.
const MERGE_CHUNKS_SUBCOMMAND: &str = "merge-chunks";

/// The `profile` subcommand: the schema-discovery/diagnostic pass, see
/// `wikisieve::profile::run_profile` and the README's "Schema discovery: `profile`".
const PROFILE_SUBCOMMAND: &str = "profile";

/// Reports this binary's build provenance (crate version, git commit, dirty flag) and exits --
/// see `wikisieve::version`. Checked in first position only, same
/// as the two subcommand names, so it never shadows an extract-mode flag of the same name.
const VERSION_FLAG: &str = "--version";

fn is_merge_chunks_invocation(args: &[String]) -> bool {
    args.first()
        .is_some_and(|arg| arg == MERGE_CHUNKS_SUBCOMMAND)
}

fn is_profile_invocation(args: &[String]) -> bool {
    args.first().is_some_and(|arg| arg == PROFILE_SUBCOMMAND)
}

fn is_version_invocation(args: &[String]) -> bool {
    args.first().is_some_and(|arg| arg == VERSION_FLAG)
}

fn main() -> io::Result<()> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if is_version_invocation(&args) {
        println!("{}", version::build_info().version_line());
        return Ok(());
    }
    if is_merge_chunks_invocation(&args) {
        args.remove(0);
        let merge_config = MergeConfig::from_args(args.into_iter())?;
        return merge_chunks::run_merge_chunks(&merge_config);
    }
    if is_profile_invocation(&args) {
        args.remove(0);
        let profile_config = ProfileConfig::from_args(args.into_iter())?;
        return profile::run_profile(&profile_config);
    }

    let config = Config::from_args(args.into_iter())?;
    extract::run_extract(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn merge_chunks_subcommand_is_recognized_only_in_first_position() {
        assert!(is_merge_chunks_invocation(&owned(&[
            "merge-chunks",
            "--chunk-dir",
            "chunks"
        ])));
        // Every pre-existing invocation still routes to the extract mode: the subcommand name is
        // only ever consulted in first position, and a flag there is not a subcommand.
        assert!(!is_merge_chunks_invocation(&owned(&[
            "--input",
            "in.jsonl.gz",
            "--output",
            "merge-chunks"
        ])));
        assert!(!is_merge_chunks_invocation(&[]));
    }

    #[test]
    fn version_flag_is_recognized_only_in_first_position() {
        assert!(is_version_invocation(&owned(&["--version"])));
        // Same rule as the subcommand names: only first position counts, so an extract-mode
        // invocation that merely names a file called "--version" is never misrouted.
        assert!(!is_version_invocation(&owned(&[
            "--input",
            "in.jsonl.gz",
            "--output",
            "--version"
        ])));
        assert!(!is_version_invocation(&[]));
    }
}
