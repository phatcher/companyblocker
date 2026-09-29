//! Embeds this build's git commit and dirty flag as compile-time env vars, consumed by
//! `crate::version::build_info`. Before this, nothing recorded which commit
//! built a `tools/bin` binary copied there by hand, so whether a deployed binary matches the
//! crate's current source was unknowable from any artefact.
//!
//! Best-effort: a checkout without `git` on `PATH`, or without a `.git` directory at all (a
//! source tarball with no history), still compiles -- it just reports `"unknown"`/clean rather
//! than failing the build over a provenance nicety.

use std::process::Command;

fn main() {
    let commit = run_git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = git_is_dirty();

    println!("cargo:rustc-env=WIKISIEVE_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=WIKISIEVE_GIT_DIRTY={dirty}");

    // Rerun whenever HEAD moves. On a branch `HEAD` only names the branch and a commit leaves it
    // unchanged; the commit moves the branch's ref file, or `packed-refs` once refs are packed.
    // Watching `HEAD` alone kept one commit stamped on every build for ten days. `--git-path`
    // resolves each file for this checkout, a worktree included, and absolute because a build
    // script runs from the crate directory, three below the repository root.
    for path in watched_git_paths() {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rerun-if-changed=build.rs");
}

/// `HEAD`, the ref it points at when on a branch, and `packed-refs`, as absolute paths.
fn watched_git_paths() -> Vec<String> {
    let mut names = vec!["HEAD".to_string(), "packed-refs".to_string()];
    if let Some(branch_ref) = run_git(&["symbolic-ref", "-q", "HEAD"]) {
        names.push(branch_ref);
    }
    names
        .iter()
        .filter_map(|name| {
            run_git(&["rev-parse", "--path-format=absolute", "--git-path", name.as_str()])
        })
        .collect()
}

fn run_git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// `true` when `git status --porcelain` reports anything at all -- any non-empty output means
/// an uncommitted change, so this build's commit alone would not reproduce it.
fn git_is_dirty() -> bool {
    run_git(&["status", "--porcelain"]).is_some_and(|status| !status.is_empty())
}
