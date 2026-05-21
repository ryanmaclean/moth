//! Jujutsu (jj) branch strategies — implements [`git::BranchStrategy`] so an
//! agent can edit a repo through jj instead of git.
//!
//! Three strategies mirror the git crate:
//!
//! * [`JjHeadStrategy`] — agent works in `repo_root` directly. Refuses when
//!   the working-copy change `@` has uncommitted content (jj's analog of a
//!   "dirty tree").
//! * [`JjWorkspaceStrategy`] — uses jj's native *workspace* feature (the
//!   analog of a git worktree). On `Success`, the source bookmark is moved
//!   forward to include the agent's work, then the workspace is forgotten
//!   and the directory removed. On `Failure`, the workspace is left in
//!   place for inspection.
//! * [`JjBookmarkStrategy`] — named bookmark (jj's renamed-from-"branch"
//!   concept). A workspace is created pointing at the bookmark; the
//!   bookmark is the persistence point. Worktree removed only on success.
//!
//! Shells out to `jj(1)`. No `libjj`, no async, no external deps beyond the
//! `git` crate (for the trait + [`git::GitError`]).
//!
//! Workspaces live at `<repo_root>/.sandcastle/workspaces/<name>` — mirroring
//! the git crate's `.sandcastle/worktrees/` layout.
//!
//! ## JJ vs git — mapping notes
//!
//! jj has no "branches" by default; it has *bookmarks* (renamed in jj 0.23+).
//! It has no "dirty tree" — every working copy is itself a real commit (`@`)
//! that jj auto-snapshots. So we map:
//!
//! * "dirty tree" => `@` is non-empty (`jj log -r @ -T empty` is `false`).
//! * "branch" / "worktree" => `bookmark` / `workspace`.
//! * "merge back" => move source bookmark to the workspace's `@` and forget
//!   the workspace. No merge commit; jj has no `--no-ff` concept. The
//!   workspace's change becomes reachable from the source bookmark.
//!
//! The `Workspace::source_branch` field stores the *workspace name* (which
//! doubles as a unique identifier for the agent's session); `target_branch`
//! stores the *source bookmark name* the work should merge into, or `None`
//! when no bookmark is at `@-` (jj allows anonymous "branches").

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

pub use git::{AgentStatus, BranchStrategy, GitError, Workspace};

// -- jj invocation ------------------------------------------------------------

/// Run `jj <args>` with cwd `cwd`. Returns the [`Output`] on success,
/// [`GitError::Git`] on non-zero exit, [`GitError::Io`] if jj can't be spawned.
///
/// We reuse [`GitError`] rather than introducing a parallel `JjError` — the
/// `BranchStrategy` trait already returns it and a second error type would
/// just force callers to convert. The `command` field's value is the literal
/// arg list, prefixed with `jj` is left implicit.
pub fn jj_cmd<I, S>(args: I, cwd: &Path) -> Result<Output, GitError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args: Vec<_> = args.into_iter().map(|s| s.as_ref().to_owned()).collect();
    let out = Command::new("jj").args(&args).current_dir(cwd).output()?;
    if out.status.success() {
        Ok(out)
    } else {
        let cmd = args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ");
        Err(GitError::Git {
            command: cmd,
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            exit: out.status.code().unwrap_or(-1),
        })
    }
}

fn stdout_trimmed(out: Output) -> String {
    String::from_utf8_lossy(&out.stdout).trim_end().to_owned()
}

/// Is `repo_root` inside a jj repo? Uses `jj root`, which prints the workspace
/// root or errors. We swallow the stderr and map to [`GitError::NotARepo`].
fn ensure_repo(repo_root: &Path) -> Result<(), GitError> {
    if !repo_root.exists() {
        return Err(GitError::NotARepo(repo_root.to_owned()));
    }
    match jj_cmd(["root"], repo_root) {
        Ok(_) => Ok(()),
        Err(_) => Err(GitError::NotARepo(repo_root.to_owned())),
    }
}

/// Is the working copy change `@` non-empty? jj's analog of `git status
/// --porcelain` — the template `empty` returns `true` when the change has no
/// diff from its parents.
fn working_copy_is_dirty(repo_root: &Path) -> Result<bool, GitError> {
    let out = jj_cmd(["log", "--no-graph", "-r", "@", "-T", "empty"], repo_root)?;
    let s = stdout_trimmed(out);
    // `empty` template returns "true" / "false". Anything else => treat as dirty
    // (safer to refuse than to clobber).
    Ok(s != "true")
}

/// The closest bookmark on the ancestor chain of `@-`, or `None`.
/// Multiple bookmarks may exist; we take the first one jj reports.
fn source_bookmark(repo_root: &Path) -> Result<Option<String>, GitError> {
    // Template emits "<name>\n" per bookmark on @-.
    let out = jj_cmd(
        [
            "log",
            "--no-graph",
            "-r",
            "@-",
            "-T",
            r#"bookmarks ++ "\n""#,
        ],
        repo_root,
    )?;
    let raw = stdout_trimmed(out);
    // The template prints space-separated bookmarks per matched commit.
    let first = raw
        .split_whitespace()
        .map(|s| s.trim_end_matches('*')) // local-only marker, just in case
        .find(|s| !s.is_empty())
        .map(|s| s.to_owned());
    Ok(first)
}

/// Does a local bookmark by this name exist?
fn bookmark_exists(repo_root: &Path, name: &str) -> Result<bool, GitError> {
    // `jj bookmark list <name>` prints nothing and exits 0 if absent, or the
    // bookmark line if present. We grep stdout.
    let out = Command::new("jj")
        .args(["bookmark", "list", name])
        .current_dir(repo_root)
        .output()?;
    if !out.status.success() {
        // Non-zero from `bookmark list <name>` means jj itself failed (e.g.
        // not a repo). Bubble up.
        return Err(GitError::Git {
            command: format!("bookmark list {name}"),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            exit: out.status.code().unwrap_or(-1),
        });
    }
    Ok(out
        .stdout
        .iter()
        .any(|b| !b.is_ascii_whitespace()))
}

fn workspace_root(repo_root: &Path) -> PathBuf {
    repo_root.join(".sandcastle").join("workspaces")
}

/// `agent-YYYYMMDD-HHMMSS-XXXX`. Mirrors the git crate's name format so both
/// strategies leave the same shape of directories on disk.
fn gen_workspace_name() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    let (y, mo, d, h, mi, s) = unix_to_ymdhms(secs);
    let pid = std::process::id();
    let suffix = (pid.wrapping_mul(2_654_435_761).wrapping_add(nanos)) & 0xFFFF;
    format!("agent-{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}-{suffix:04x}")
}

fn unix_to_ymdhms(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let s = (secs % 60) as u32;
    let m = ((secs / 60) % 60) as u32;
    let h = ((secs / 3600) % 24) as u32;
    let mut days = (secs / 86_400) as i64;
    let mut year: i64 = 1970;
    loop {
        let leap = is_leap(year);
        let yd = if leap { 366 } else { 365 };
        if days < yd {
            break;
        }
        days -= yd;
        year += 1;
    }
    let leap = is_leap(year);
    let dim = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 0usize;
    while month < 12 && days >= dim[month] {
        days -= dim[month];
        month += 1;
    }
    (year as u32, (month + 1) as u32, (days + 1) as u32, h, m, s)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// -- JjHeadStrategy -----------------------------------------------------------

/// Agent works directly in `repo_root`. Refuses to start when the working-copy
/// change `@` has uncommitted content — analogous to git's dirty-tree check.
pub struct JjHeadStrategy;

impl BranchStrategy for JjHeadStrategy {
    fn prepare(&self, repo_root: &Path) -> Result<Workspace, GitError> {
        ensure_repo(repo_root)?;
        if working_copy_is_dirty(repo_root)? {
            return Err(GitError::DirtyTree);
        }
        let target = source_bookmark(repo_root)?;
        Ok(Workspace {
            path: repo_root.to_owned(),
            // For the head strategy the "source" is the working-copy change.
            // We use the target bookmark name (if any), else "@" as a literal.
            source_branch: target.clone().unwrap_or_else(|| "@".to_owned()),
            target_branch: target,
        })
    }

    fn finish(&self, _ws: Workspace, _status: AgentStatus) -> Result<(), GitError> {
        Ok(())
    }

    fn name(&self) -> &str {
        "jj-head"
    }
}

// -- JjWorkspaceStrategy ------------------------------------------------------

/// Native jj workspace (the analog of a git worktree). On `Success`, the
/// source bookmark — if any — is advanced to include the agent's work, then
/// the workspace is forgotten and its directory removed. On `Failure`, the
/// workspace is left in place and its path is printed to stderr.
pub struct JjWorkspaceStrategy;

impl BranchStrategy for JjWorkspaceStrategy {
    fn prepare(&self, repo_root: &Path) -> Result<Workspace, GitError> {
        ensure_repo(repo_root)?;
        let target = source_bookmark(repo_root)?;
        let ws_name = gen_workspace_name();
        let ws_dir = workspace_root(repo_root);
        std::fs::create_dir_all(&ws_dir)?;
        let ws_path = ws_dir.join(&ws_name);

        jj_cmd(
            [
                OsStr::new("workspace"),
                OsStr::new("add"),
                OsStr::new("--name"),
                OsStr::new(&ws_name),
                ws_path.as_os_str(),
            ],
            repo_root,
        )?;
        Ok(Workspace {
            path: ws_path,
            source_branch: ws_name,
            target_branch: target,
        })
    }

    fn finish(&self, ws: Workspace, status: AgentStatus) -> Result<(), GitError> {
        // Resolve repo_root from any workspace path via `jj root` run *from*
        // the source repo dir. We stored the ws under <repo_root>/.sandcastle/
        // workspaces/<name>, so walk up the path.
        let repo_root = repo_root_from_workspace(&ws.path)?;

        match status {
            AgentStatus::Success => {
                // Advance the source bookmark — if any — to include the agent's
                // work. Setting a bookmark to the workspace's `@` makes the
                // agent's accumulated changes reachable from it. If `@` is
                // empty (agent did nothing), jj will refuse to move backwards
                // unless --allow-backwards is passed; we tolerate that error.
                if let Some(target) = &ws.target_branch {
                    let rev = format!("{}@", ws.source_branch);
                    let _ = jj_cmd(
                        [
                            "bookmark",
                            "set",
                            "--allow-backwards",
                            target,
                            "-r",
                            &rev,
                        ],
                        &repo_root,
                    )?;
                }
                jj_cmd(
                    ["workspace", "forget", &ws.source_branch],
                    &repo_root,
                )?;
                // jj does not delete the workspace dir for us; do it ourselves.
                let _ = std::fs::remove_dir_all(&ws.path);
                Ok(())
            }
            AgentStatus::Failure(reason) => {
                eprintln!(
                    "jj-workspace: agent failed ({reason}); workspace preserved at {}",
                    ws.path.display()
                );
                Ok(())
            }
        }
    }

    fn name(&self) -> &str {
        "jj-workspace"
    }
}

// -- JjBookmarkStrategy -------------------------------------------------------

/// Commits land on a caller-named bookmark. A workspace is created pointing at
/// it. The bookmark is the persistence point — no implicit merge into the
/// surrounding repo's `@-` bookmark. Worktree removed only on success.
pub struct JjBookmarkStrategy {
    pub name: String,
}

impl JjBookmarkStrategy {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl BranchStrategy for JjBookmarkStrategy {
    fn prepare(&self, repo_root: &Path) -> Result<Workspace, GitError> {
        ensure_repo(repo_root)?;
        let ws_dir = workspace_root(repo_root);
        std::fs::create_dir_all(&ws_dir)?;
        let ws_path = ws_dir.join(&self.name);

        // If the bookmark already exists, point the new workspace at it.
        // Otherwise the workspace inherits the source repo's @- (default).
        let existed = bookmark_exists(repo_root, &self.name)?;
        if existed {
            jj_cmd(
                [
                    OsStr::new("workspace"),
                    OsStr::new("add"),
                    OsStr::new("--name"),
                    OsStr::new(&self.name),
                    OsStr::new("-r"),
                    OsStr::new(&self.name),
                    ws_path.as_os_str(),
                ],
                repo_root,
            )?;
        } else {
            jj_cmd(
                [
                    OsStr::new("workspace"),
                    OsStr::new("add"),
                    OsStr::new("--name"),
                    OsStr::new(&self.name),
                    ws_path.as_os_str(),
                ],
                repo_root,
            )?;
            // Create the bookmark at the new workspace's @ so future moves
            // have a target. `bookmark create` is invoked from the source
            // repo, with `-r <name>@` to scope to the new workspace.
            let rev = format!("{}@", self.name);
            jj_cmd(["bookmark", "create", &self.name, "-r", &rev], repo_root)?;
        }
        Ok(Workspace {
            path: ws_path,
            source_branch: self.name.clone(),
            target_branch: None,
        })
    }

    fn finish(&self, ws: Workspace, status: AgentStatus) -> Result<(), GitError> {
        let repo_root = repo_root_from_workspace(&ws.path)?;
        match status {
            AgentStatus::Success => {
                // Advance the bookmark to the workspace's @ so the agent's
                // accumulated work is captured under it.
                let rev = format!("{}@", self.name);
                let _ = jj_cmd(
                    [
                        "bookmark",
                        "set",
                        "--allow-backwards",
                        &self.name,
                        "-r",
                        &rev,
                    ],
                    &repo_root,
                )?;
                jj_cmd(["workspace", "forget", &self.name], &repo_root)?;
                let _ = std::fs::remove_dir_all(&ws.path);
                Ok(())
            }
            AgentStatus::Failure(reason) => {
                eprintln!(
                    "jj-bookmark[{}]: agent failed ({reason}); workspace preserved at {}",
                    self.name,
                    ws.path.display()
                );
                Ok(())
            }
        }
    }

    fn name(&self) -> &str {
        "jj-bookmark"
    }
}

/// Given a workspace path of the form `<repo_root>/.sandcastle/workspaces/<n>`,
/// recover `<repo_root>` by walking up two components. If the workspace dir
/// has already been removed (Success path) the path is still a `PathBuf` we
/// can manipulate purely lexically.
fn repo_root_from_workspace(ws_path: &Path) -> Result<PathBuf, GitError> {
    ws_path
        .parent() // <repo>/.sandcastle/workspaces
        .and_then(Path::parent) // <repo>/.sandcastle
        .and_then(Path::parent) // <repo>
        .map(Path::to_owned)
        .ok_or_else(|| GitError::Other(format!("can't infer repo root from {}", ws_path.display())))
}

// -- tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Returns `true` and prints a skip notice if `jj` isn't installed.
    fn jj_missing() -> bool {
        match Command::new("jj").arg("--version").output() {
            Ok(o) if o.status.success() => false,
            _ => {
                eprintln!("jj not installed, skipping");
                true
            }
        }
    }

    struct TempRepo {
        path: PathBuf,
    }

    impl TempRepo {
        fn new() -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "sandcastle-jj-test-{}-{}-{}",
                std::process::id(),
                nanos,
                n
            ));
            std::fs::create_dir_all(&path).unwrap();
            jj_cmd(["git", "init"], &path).unwrap();
            // user.name/email get jj past the "won't push" warning; doesn't
            // affect functionality but keeps test stderr clean.
            jj_cmd(["config", "set", "--repo", "user.name", "t"], &path).unwrap();
            jj_cmd(
                ["config", "set", "--repo", "user.email", "t@t"],
                &path,
            )
            .unwrap();
            // Make sure @ is empty and a `main` bookmark exists at @-, so the
            // tests have a stable starting state.
            jj_cmd(["describe", "-m", "initial"], &path).unwrap();
            jj_cmd(["new", "-m", ""], &path).unwrap();
            jj_cmd(["bookmark", "create", "main", "-r", "@-"], &path).unwrap();
            TempRepo { path }
        }

        fn write(&self, rel: &str, contents: &str) {
            let p = self.path.join(rel);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(p, contents).unwrap();
        }
    }

    impl Drop for TempRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn bookmark_target(repo: &Path, name: &str) -> Option<String> {
        // `jj log -r <bm> -T 'change_id.short()' --no-graph` => change id, or
        // empty if the bookmark doesn't resolve.
        let out = Command::new("jj")
            .args([
                "log",
                "--no-graph",
                "-r",
                name,
                "-T",
                "change_id.short()",
                "--ignore-working-copy",
            ])
            .current_dir(repo)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&out.stdout).trim_end().to_owned();
        if s.is_empty() { None } else { Some(s) }
    }

    // ---- JjHeadStrategy ----

    #[test]
    fn head_clean_repo_returns_repo_root() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        let ws = JjHeadStrategy.prepare(&r.path).unwrap();
        assert_eq!(ws.path, r.path);
        assert_eq!(ws.target_branch.as_deref(), Some("main"));
        JjHeadStrategy.finish(ws, AgentStatus::Success).unwrap();
    }

    #[test]
    fn head_refuses_dirty_working_copy() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        r.write("dirty.txt", "uncommitted");
        let err = JjHeadStrategy.prepare(&r.path).unwrap_err();
        assert!(matches!(err, GitError::DirtyTree), "got {err:?}");
    }

    #[test]
    fn head_finish_is_noop_on_failure() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        let ws = JjHeadStrategy.prepare(&r.path).unwrap();
        JjHeadStrategy
            .finish(ws, AgentStatus::Failure("boom".into()))
            .unwrap();
        assert!(r.path.join(".jj").exists());
    }

    // ---- JjWorkspaceStrategy ----

    #[test]
    fn workspace_happy_path_advances_source_bookmark() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        let before = bookmark_target(&r.path, "main").unwrap();

        let ws = JjWorkspaceStrategy.prepare(&r.path).unwrap();
        assert!(ws.path.starts_with(r.path.join(".sandcastle/workspaces")));
        assert!(ws.path.exists());
        assert_eq!(ws.target_branch.as_deref(), Some("main"));

        // Agent does work in the workspace.
        std::fs::write(ws.path.join("agent.txt"), "hi").unwrap();
        jj_cmd(["describe", "-m", "agent work"], &ws.path).unwrap();
        // Move to a fresh empty @ so the described commit becomes a parent.
        jj_cmd(["new"], &ws.path).unwrap();

        let wt_path = ws.path.clone();
        JjWorkspaceStrategy
            .finish(ws, AgentStatus::Success)
            .unwrap();

        assert!(!wt_path.exists(), "workspace dir should be removed on success");
        let after = bookmark_target(&r.path, "main").unwrap();
        assert_ne!(before, after, "main should have moved forward");
    }

    #[test]
    fn workspace_failure_preserves_dir() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        let ws = JjWorkspaceStrategy.prepare(&r.path).unwrap();
        let wt_path = ws.path.clone();
        std::fs::write(wt_path.join("wip.txt"), "wip").unwrap();
        JjWorkspaceStrategy
            .finish(ws, AgentStatus::Failure("nope".into()))
            .unwrap();
        assert!(wt_path.exists(), "workspace should be kept on failure");
        // The workspace still resolves in `jj workspace list`.
        let out = jj_cmd(["workspace", "list"], &r.path).unwrap();
        let listing = String::from_utf8_lossy(&out.stdout);
        assert!(listing.contains("agent-"), "expected agent- in: {listing}");
    }

    #[test]
    fn workspace_path_layout() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        let ws = JjWorkspaceStrategy.prepare(&r.path).unwrap();
        let parent = ws.path.parent().unwrap();
        assert!(parent.ends_with(".sandcastle/workspaces"));
        assert!(ws.source_branch.starts_with("agent-"));
        JjWorkspaceStrategy
            .finish(ws, AgentStatus::Success)
            .unwrap();
    }

    // ---- JjBookmarkStrategy ----

    #[test]
    fn bookmark_new_name_creates_bookmark_and_workspace() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        assert!(!bookmark_exists(&r.path, "agent-x").unwrap());
        let s = JjBookmarkStrategy::new("agent-x");
        let ws = s.prepare(&r.path).unwrap();
        assert!(ws.path.exists());
        assert_eq!(ws.source_branch, "agent-x");
        assert!(bookmark_exists(&r.path, "agent-x").unwrap());
        s.finish(ws, AgentStatus::Success).unwrap();
    }

    #[test]
    fn bookmark_existing_name_is_reused() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        jj_cmd(["bookmark", "create", "agent-work", "-r", "@-"], &r.path).unwrap();
        let before = bookmark_target(&r.path, "agent-work").unwrap();

        let s = JjBookmarkStrategy::new("agent-work");
        let ws = s.prepare(&r.path).unwrap();
        std::fs::write(ws.path.join("x.txt"), "x").unwrap();
        jj_cmd(["describe", "-m", "on existing"], &ws.path).unwrap();
        jj_cmd(["new"], &ws.path).unwrap();
        s.finish(ws, AgentStatus::Success).unwrap();

        let after = bookmark_target(&r.path, "agent-work").unwrap();
        assert_ne!(before, after, "agent-work should have advanced");
        // main should not have moved.
        let main_log = jj_cmd(
            ["log", "--no-graph", "-r", "main", "-T", "description"],
            &r.path,
        )
        .unwrap();
        let desc = String::from_utf8_lossy(&main_log.stdout);
        assert!(!desc.contains("on existing"), "main contains: {desc}");
    }

    #[test]
    fn bookmark_failure_preserves_workspace() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        let s = JjBookmarkStrategy::new("inspect-me");
        let ws = s.prepare(&r.path).unwrap();
        let wt = ws.path.clone();
        s.finish(ws, AgentStatus::Failure("nope".into())).unwrap();
        assert!(wt.exists());
        assert!(bookmark_exists(&r.path, "inspect-me").unwrap());
    }

    // ---- general ----

    #[test]
    fn not_a_jj_repo_errors() {
        if jj_missing() {
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "sandcastle-jj-not-a-repo-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let err = JjHeadStrategy.prepare(&dir).unwrap_err();
        assert!(matches!(err, GitError::NotARepo(_)), "got {err:?}");
        let err = JjWorkspaceStrategy.prepare(&dir).unwrap_err();
        assert!(matches!(err, GitError::NotARepo(_)), "got {err:?}");
        let err = JjBookmarkStrategy::new("x").prepare(&dir).unwrap_err();
        assert!(matches!(err, GitError::NotARepo(_)), "got {err:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn strategy_names() {
        assert_eq!(JjHeadStrategy.name(), "jj-head");
        assert_eq!(JjWorkspaceStrategy.name(), "jj-workspace");
        assert_eq!(JjBookmarkStrategy::new("x").name(), "jj-bookmark");
    }

    #[test]
    fn gen_workspace_name_format() {
        let n = gen_workspace_name();
        assert!(n.starts_with("agent-"));
        assert_eq!(n.len(), 26, "unexpected: {n}");
        let parts: Vec<&str> = n.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "agent");
        assert_eq!(parts[1].len(), 8);
        assert_eq!(parts[2].len(), 6);
        assert_eq!(parts[3].len(), 4);
    }

    #[test]
    fn jj_cmd_propagates_failure() {
        if jj_missing() {
            return;
        }
        let r = TempRepo::new();
        let err = jj_cmd(["this-is-not-a-jj-subcommand"], &r.path).unwrap_err();
        match err {
            GitError::Git { exit, .. } => assert!(exit != 0),
            other => panic!("expected Git, got {other:?}"),
        }
    }

    #[test]
    fn unix_to_ymdhms_known_epochs() {
        assert_eq!(unix_to_ymdhms(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(unix_to_ymdhms(946_684_800), (2000, 1, 1, 0, 0, 0));
        assert_eq!(unix_to_ymdhms(1_709_209_496), (2024, 2, 29, 12, 24, 56));
    }

    #[test]
    fn repo_root_from_workspace_walks_up_two_dirs() {
        let p = Path::new("/tmp/foo/.sandcastle/workspaces/agent-x");
        assert_eq!(
            repo_root_from_workspace(p).unwrap(),
            PathBuf::from("/tmp/foo")
        );
    }
}
