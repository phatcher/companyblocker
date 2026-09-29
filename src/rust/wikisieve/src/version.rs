//! Build-time provenance embedded by `build.rs`: which crate version and git commit (with a
//! dirty flag) produced this binary. `wikisieve --version` and every `--summary-json` report the
//! same values (`main::is_version_invocation`, `extract::write_summary`), so a run's output can
//! always be tied back to the exact source that built the binary that made it. Before this, nothing recorded which commit built a `tools/bin` binary copied there
//! by hand.

/// One build's provenance: the crate version from `Cargo.toml`, the git commit `build.rs`
/// captured at compile time, and whether the working tree carried uncommitted changes then.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildInfo {
    pub version: &'static str,
    pub commit: &'static str,
    pub dirty: bool,
}

impl BuildInfo {
    /// The line `wikisieve --version` prints: `wikisieve <version> (commit <commit>[, dirty])`.
    #[must_use]
    pub fn version_line(&self) -> String {
        if self.dirty {
            format!("wikisieve {} (commit {}, dirty)", self.version, self.commit)
        } else {
            format!("wikisieve {} (commit {})", self.version, self.commit)
        }
    }
}

/// This binary's build provenance, embedded by `build.rs` via `cargo:rustc-env`.
#[must_use]
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        commit: env!("WIKISIEVE_GIT_COMMIT"),
        dirty: env!("WIKISIEVE_GIT_DIRTY") == "true",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_line_reports_clean_and_dirty_builds() {
        let clean = BuildInfo {
            version: "1.2.3",
            commit: "abcdef",
            dirty: false,
        };
        assert_eq!(clean.version_line(), "wikisieve 1.2.3 (commit abcdef)");

        let dirty = BuildInfo {
            version: "1.2.3",
            commit: "abcdef",
            dirty: true,
        };
        assert_eq!(
            dirty.version_line(),
            "wikisieve 1.2.3 (commit abcdef, dirty)"
        );
    }

    #[test]
    fn build_info_reports_the_crate_version_and_a_non_empty_commit() {
        let info = build_info();
        assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
        assert!(!info.commit.is_empty());
    }

    #[test]
    fn build_info_reports_the_commit_the_checkout_is_at() {
        // A build that missed a commit keeps reporting the one before it, which is how every
        // build for ten days reported one stale commit. Skipped where there is no git to ask,
        // as `build.rs` itself falls back to "unknown" there.
        let Ok(output) = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
        else {
            return;
        };
        if !output.status.success() || build_info().commit == "unknown" {
            return;
        }
        let head = String::from_utf8(output.stdout).unwrap();
        assert_eq!(build_info().commit, head.trim());
    }
}
