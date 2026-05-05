//! LLM session-id capture and resolution.
//!
//! The Claude Code / Gemini / Cursor / Copilot hosts all pass a per-session UUID
//! in the JSON payload sent to PreToolUse hooks. RTK persists that id in the
//! `commands.llm_session_id` column so external tools (e.g. monitor-ccu) can
//! correlate executed commands with the originating LLM conversation.
//!
//! Four capture vectors, applied in priority order at INSERT time:
//!
//! 1. CLI flag `--llm-session-id <uuid>` — explicit override.
//! 2. Environment variable `RTK_LLM_SESSION_ID` — picked up automatically.
//! 3. Hook handlers prefix the rewritten command with the env var (covered by 2).
//! 4. Process-tree walk + on-disk cache populated by `rtk hook session-start`
//!    — last-resort fallback when none of the above is set.
//!
//! Vector 3 produces vector 2 at exec time, so the only resolution paths the
//! Tracker actually needs to handle are 1, 2, and 4.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Environment variable consulted by every code path that opens a tracker.
pub const ENV_LLM_SESSION_ID: &str = "RTK_LLM_SESSION_ID";

/// Maximum age of a session cache file before it is treated as stale and
/// removed during housekeeping. Caps unbounded growth if SessionEnd never fires.
const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Maximum number of parent levels walked when discovering a session id via
/// the process tree. Protects against pathological cycles or runaway loops.
const MAX_PROC_TREE_DEPTH: usize = 20;

/// Persisted layout of a single active-session file under
/// `~/.cache/rtk/active-sessions/<claude_pid>.json`.
#[derive(Debug, Serialize, Deserialize)]
struct SessionFile {
    session_id: String,
    started_at: String,
    claude_pid: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transcript_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
}

/// Resolve the LLM session id at INSERT time.
///
/// Honors precedence: explicit CLI arg (1) → env var (2) → process-tree
/// fallback (4). Returns `None` if no vector yields a value, in which case the
/// row is recorded with `llm_session_id = NULL`.
pub fn resolve_llm_session_id(cli_arg: Option<String>) -> Option<String> {
    cli_arg
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var(ENV_LLM_SESSION_ID)
                .ok()
                .filter(|s| !s.is_empty())
        })
        .or_else(discover_via_proctree)
}

/// Extract a session id from a hook JSON payload, trying every field name used
/// in the wild across Claude Code, Gemini CLI, Cursor and Copilot.
///
/// Returns `None` if no matching field is present or if the value is empty.
pub fn extract_session_id_from_payload(v: &Value) -> Option<String> {
    const KEYS: &[&str] = &[
        "session_id",
        "sessionId",
        "conversation_id",
        "conversationId",
    ];
    for key in KEYS {
        if let Some(s) = v.get(*key).and_then(|x| x.as_str()) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

/// Build a shell prefix that the LLM host will execute verbatim. The prefix
/// scopes `RTK_LLM_SESSION_ID` to the rewritten command only, leaving the
/// caller's environment unchanged.
///
/// The session id is shell-quoted defensively even though hosts only emit
/// UUID-shaped strings, to keep this safe if upstream payloads ever change.
pub fn prefix_command_with_session_id(rewritten: &str, session_id: &str) -> String {
    format!(
        "{}={} {}",
        ENV_LLM_SESSION_ID,
        shell_quote(session_id),
        rewritten
    )
}

/// POSIX-safe single-quote escape: every literal single quote becomes `'\''`.
fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Directory holding active-session marker files. Created on first write.
fn cache_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".cache").join("rtk").join("active-sessions"))
}

/// Ensure the cache directory exists. Best-effort `chmod 0700` on Unix to keep
/// session ids out of reach of other local users.
fn ensure_cache_dir(dir: &PathBuf) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("create_dir_all {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(dir)?.permissions();
        perms.set_mode(0o700);
        let _ = fs::set_permissions(dir, perms);
    }
    Ok(())
}

/// Drop cache files older than `CACHE_TTL`. Best-effort: any I/O error on a
/// single entry is ignored so housekeeping never blocks a real session.
fn prune_stale_cache(dir: &PathBuf) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        let Ok(modified) = meta.modified() else {
            continue;
        };
        if now
            .duration_since(modified)
            .map(|age| age > CACHE_TTL)
            .unwrap_or(false)
        {
            let _ = fs::remove_file(&path);
        }
    }
}

/// Persist a session marker for the current Claude/Gemini/Cursor/Copilot host
/// process so future RTK invocations spawned beneath it can recover the id via
/// proctree walk.
///
/// Reads a SessionStart-style JSON payload from stdin. Looks for a session id
/// (any of the canonical key names) plus optional `transcript_path`/`cwd`
/// hints. Writes the result keyed by `$PPID` (the host that invoked the hook).
pub fn run_session_start() -> Result<()> {
    let mut input = String::new();
    io::stdin()
        .take(1_048_576)
        .read_to_string(&mut input)
        .context("read stdin")?;
    let input = input.trim();
    if input.is_empty() {
        return Ok(());
    }

    let v: Value = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };

    let session_id = match extract_session_id_from_payload(&v) {
        Some(s) => s,
        None => return Ok(()),
    };

    let parent_pid = parent_pid_of_self().unwrap_or(0);
    let transcript_path = v
        .get("transcript_path")
        .and_then(|x| x.as_str())
        .map(String::from);
    let cwd = v.get("cwd").and_then(|x| x.as_str()).map(String::from);

    let dir = match cache_dir() {
        Some(d) => d,
        None => return Ok(()),
    };
    ensure_cache_dir(&dir)?;
    prune_stale_cache(&dir);

    let payload = SessionFile {
        session_id,
        started_at: chrono::Utc::now().to_rfc3339(),
        claude_pid: parent_pid,
        transcript_path,
        cwd,
    };

    let path = dir.join(format!("{parent_pid}.json"));
    let body = serde_json::to_string(&payload).context("serialize session file")?;
    fs::write(&path, body).with_context(|| format!("write {}", path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
    }

    Ok(())
}

/// Remove the session marker for the current host process. Best-effort:
/// missing files are not an error (SessionEnd may fire without a SessionStart
/// in tooling that re-attaches to an existing session).
pub fn run_session_end() -> Result<()> {
    let parent_pid = match parent_pid_of_self() {
        Some(pid) => pid,
        None => return Ok(()),
    };
    let dir = match cache_dir() {
        Some(d) => d,
        None => return Ok(()),
    };
    let path = dir.join(format!("{parent_pid}.json"));
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Walk parent PIDs up to `MAX_PROC_TREE_DEPTH` levels looking for a cached
/// session marker. Returns `None` when no ancestor matches or when the OS
/// prevents the walk.
fn discover_via_proctree() -> Option<String> {
    let dir = cache_dir()?;
    if !dir.exists() {
        return None;
    }

    use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_processes(ProcessRefreshKind::new()));
    sys.refresh_processes();

    let mut current_pid = std::process::id();
    for _ in 0..MAX_PROC_TREE_DEPTH {
        let candidate = dir.join(format!("{current_pid}.json"));
        if candidate.exists() {
            if let Ok(body) = fs::read_to_string(&candidate) {
                if let Ok(parsed) = serde_json::from_str::<SessionFile>(&body) {
                    if !parsed.session_id.is_empty() {
                        return Some(parsed.session_id);
                    }
                }
            }
        }
        let proc = sys.process(Pid::from_u32(current_pid))?;
        let parent = proc.parent()?;
        let parent_u32 = parent.as_u32();
        if parent_u32 == 0 || parent_u32 == current_pid {
            return None;
        }
        current_pid = parent_u32;
    }
    None
}

/// PID of the process that spawned the current `rtk` invocation.
///
/// Used by SessionStart/SessionEnd: the LLM host invokes the hook as a child,
/// so `$PPID` from rtk's perspective is the host's PID.
fn parent_pid_of_self() -> Option<u32> {
    use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_processes(ProcessRefreshKind::new()));
    sys.refresh_processes();
    let me = sys.process(Pid::from_u32(std::process::id()))?;
    me.parent().map(|p| p.as_u32())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_session_id_handles_canonical_keys() {
        assert_eq!(
            extract_session_id_from_payload(&json!({"session_id": "abc"})),
            Some("abc".into())
        );
        assert_eq!(
            extract_session_id_from_payload(&json!({"sessionId": "def"})),
            Some("def".into())
        );
        assert_eq!(
            extract_session_id_from_payload(&json!({"conversation_id": "ghi"})),
            Some("ghi".into())
        );
        assert_eq!(
            extract_session_id_from_payload(&json!({"conversationId": "jkl"})),
            Some("jkl".into())
        );
    }

    #[test]
    fn extract_session_id_returns_none_when_absent_or_empty() {
        assert_eq!(extract_session_id_from_payload(&json!({})), None);
        assert_eq!(
            extract_session_id_from_payload(&json!({"session_id": ""})),
            None
        );
    }

    #[test]
    fn prefix_command_quotes_session_id() {
        let prefixed = prefix_command_with_session_id("rtk git status", "abc-123");
        assert_eq!(prefixed, "RTK_LLM_SESSION_ID='abc-123' rtk git status");
    }

    #[test]
    fn prefix_command_escapes_single_quotes() {
        let prefixed = prefix_command_with_session_id("rtk git status", "a'b");
        assert_eq!(prefixed, "RTK_LLM_SESSION_ID='a'\\''b' rtk git status");
    }

    #[test]
    fn cli_arg_takes_precedence_over_env() {
        // SAFETY: this test only sets and removes its own private env var.
        std::env::set_var(ENV_LLM_SESSION_ID, "from-env");
        let resolved = resolve_llm_session_id(Some("from-cli".into()));
        std::env::remove_var(ENV_LLM_SESSION_ID);
        assert_eq!(resolved, Some("from-cli".into()));
    }

    #[test]
    fn empty_cli_arg_falls_back_to_env() {
        std::env::set_var(ENV_LLM_SESSION_ID, "from-env-fallback");
        let resolved = resolve_llm_session_id(Some(String::new()));
        std::env::remove_var(ENV_LLM_SESSION_ID);
        assert_eq!(resolved, Some("from-env-fallback".into()));
    }
}
