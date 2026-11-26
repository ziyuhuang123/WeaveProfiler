#!/usr/bin/env python3
"""WeaveProfiler (`wp`) unified hard-enforcement hook for Claude Code, Codex CLI, Qwen Code, and Antigravity.

Wired by `agent_installer.py` into:
  * Claude Code:  `.claude/settings.json`
  * Codex CLI:    `.codex/hooks.json`
  * Qwen Code:    `.qwen/settings.json` (`--protocol qwen`)
  * Antigravity:  `.agents/plugins/weaveprofiler/hooks.json` (`--protocol agy`)

Invocation:

    python -B <skill_root>/weaveprof/guard_hook.py --event PreToolUse|PostToolUse|PostInvocation|Stop|SubagentStop \
        --protocol auto|claude|codex|qwen|agy --workspace <ws> --session-dir <session_dir>

The hook reads one JSON event from stdin (supporting both snake_case Claude/Codex/Qwen
payloads and camelCase Antigravity `toolCall` / `continuationActive` payloads) and emits
the target agent's native allow/block response:
  * Claude Code / Codex CLI: exit 0 to allow; stderr reason + exit 2 to block.
  * Qwen Code: JSON decision on stdout + exit 0 (`permissionDecision: "deny"` for
    `PreToolUse`; `decision: "block"` + `stopReason` for `Stop` / `SubagentStop`).
  * Antigravity (`agy`): JSON decision on stdout + exit 0 (`decision: "allow"|"deny"` for
    `PreToolUse`; `{}` for `PostToolUse` + `PostInvocation` alert injection;
    `decision: "allow"|"continue"` for `Stop`).

This module also hosts the shared session primitives (session location, HMAC seal
verification, seal registration, byte-exact backup restore, completion verdict) used by
`mcp_server.py` and `agent_installer.py`.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import contextlib
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import struct
import time
import uuid
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
SKILL_ROOT = _HERE.parent
try:
    if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == _HERE:
        sys.path.pop(0)
except OSError:
    pass
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

SESSION_SCHEMA = "weaveprof.agent-session/v1"
SESSION_FILE = "session.json"
KEY_FILE = ".supervisor.key"
BACKUP_DIRNAME = "backups"
CONFIG_BACKUP_DIRNAME = "config_backups"
GLOBAL_GUARDS_DIRNAME = "_global_guards"
STOP_BLOCK_LIMIT = 3
EXIT_ALLOW = 0
EXIT_BLOCK = 2
TAG = "[WeaveProfiler wp guard]"
SEALING_TOOLS_HINT = "`weaveprof_megakernel_audit` (fast isolated-copy megakernel audit) or `weaveprof_profile`"

FORBIDDEN_GZ_SUFFIXES: Tuple[str, ...] = (
    ".perfetto.json.gz",
    ".trace.json.gz",
    ".perfetto-trace.gz",
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_SCAN_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".pytest_cache", ".mypy_cache", "site-packages", "third_party", "3rdparty",
})
_GZ_CLEANUP_ONLY_RE = re.compile(
    r"^\s*(?:rm|del|erase|unlink|Remove-Item)\b[^\n;|&]*(?:\.perfetto\.json\.gz|\.trace\.json\.gz|\.perfetto-trace\.gz)\s*$",
    re.I,
)
_FORBIDDEN_GZ_CMD_RE = re.compile(
    r"(?:(?:--write-gz\b|write_perfetto_gz\s*=\s*True\b|export_perfetto_native\.py\b)"
    r"|(?:(?:-o|--out|output|out_path|gzip\.open|write_json|>{1,2}|tee|cp|mv)\s*[^;\n|&]{0,160}"
    r"(?:\.perfetto\.json\.gz|\.trace\.json\.gz|\.perfetto-trace\.gz)\b)"
    r"|(?:\b(?:wp_decode|fi_lite_decode)\.py\b[^\n;|&]*(?:\.perfetto\.json\.gz|\.trace\.json\.gz|\.perfetto-trace\.gz)\b))",
    re.I,
)
_WP_PROFILING_ACTIVITY_RE = re.compile(
    r"(?:\bwp_decode\.py\b|\bfi_lite_decode\.py\b|\bwp_triton(?:_amd)?\b|\bwp_cute\b|\bwp_lite(?:_hip)?\.(?:cuh|h)\b"
    r"|\bprobe-inject\b|\bprobe-place\b|\bcapture\.(?:fi02|wp02)\.bin\b|\bfi_merge_perfetto\b"
    r"|\bweaveprof_(?:megakernel_audit|profile)\b|\brun_megakernel_audit\b|\bwp\.(?:profile|verify)\b"
    r"|\bweaveprof\s+(?:profile|verify|render)\b|\brender_swhw_overview_png\b"
    r"|\.(?:perfetto\.json\.gz|trace\.json\.gz|perfetto-trace\.gz)\b)",
    re.I,
)
_HELP_ONLY_CMD_RE = re.compile(r"^\s*[^;&|\n]*\s(?:--help|-h|--version)\s*$", re.I)
_WSL_MNT_RE = re.compile(r"/mnt/([a-zA-Z])(/[^\s\"'`;|&)>]*)")
_WIN_PATH_RE = re.compile(r"([A-Za-z]:[\\/][^\s\"'`;|&)>]+)")

# Legacy `fi_*` tokens are intentionally retained alongside `wp_*` so agents cannot bypass
# probe-pollution checks via older header/macro spellings.
PROBE_TOKENS: Tuple[str, ...] = (
    "wp_lite.cuh", "wp_lite_hip.h", "wp_cute", "wp_triton",
    "WP_OPEN", "WP_KERNEL_SCOPE", "WP_DEVICE_SCOPE", "WP_B(", "WP_E(",
    "fi_lite.cuh", "fi_lite_hip.h", "fi_cute", "fi_triton",
    "FI_OPEN", "FI_KERNEL_SCOPE", "FI_B(", "FI_E(",
    "%globaltimer", "s_memrealtime", "prof_interval_start",
)
SECRET_NEEDLES: Tuple[str, ...] = (
    ".supervisor.key", ".supervisor.", "weaveprof/sessions",
    "weaveprof_supervisor_hmac_key", "weaveprof_supervisor_key_file",
    "weaveprof_sessions_dir", "guard_hook.py", "mcp_server.py",
)

EDIT_TOOLS = {
    "edit", "write", "multiedit", "notebookedit", "notebook_edit", "str_replace_based_edit_tool", "str_replace_editor",
    "create_file", "write_file", "write_to_file", "edit_file", "replace", "replace_file_content",
    "multi_replace_file_content", "apply_patch", "applypatch", "patch",
}
SHELL_TOOLS = {
    "bash", "shell", "local_shell", "exec", "exec_command", "container.exec", "run_shell_command",
    "run_command", "terminal", "powershell", "unified_exec", "run_terminal_cmd",
    "write_stdin", "manage_task",
}
READ_ONLY_TOOLS = {
    "read", "read_file", "grep", "grep_search", "glob", "ls", "list_directory", "list_dir", "find_by_name",
    "websearch", "web_search", "webfetch", "web_fetch", "todowrite", "todo_write", "todoread",
    "view", "view_file", "view_file_outline", "view_code_item",
}

# Allowlist of normal Antigravity stop reasons that must pass the Stop gate; any other
# reason (USER_CANCELED, ERROR, MAX_INVOCATIONS, MAX_STEPS_EXCEEDED, etc.) exits cleanly.
_AGY_GATE_STOP_REASONS = {"", "NO_TOOL_CALL", "MODEL_STOP"}

_SAMPLING_RE = re.compile(
    r"(?:\b(?:FI_SAMPLE|WP_SAMPLE|WEAVEPROF_SAMPLE|--fi-sample|--wp-sample|--sample-stride)\s*(?:=|:|\s)\s*['\"]?[+]?(?:0x[0-9a-f]+|[2-9]|\d{2,})"
    r"|['\"](?:FI_SAMPLE|WP_SAMPLE|WEAVEPROF_SAMPLE)['\"]\s*[\]:,]\s*=?\s*['\"]?[+]?(?:0x[0-9a-f]+|[2-9]|\d{2,}))",
    re.I,
)
_INSTALLER_RE = re.compile(
    r"(?:\bagent_installer\b|\binstall[-_]agent\b|\bguard[-_]exec\b|run\.py['\"]?\s+(?:install|agent|guard)\w*)", re.I
)
_PATCH_HDR_RE = re.compile(r"^\*\*\* (Add|Update|Delete) File:\s*(.+?)\s*$")
_PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to:\s*(.+?)\s*$")


def _shell_write_re(target: str) -> "re.Pattern[str]":
    """Heuristic: shell command that writes/deletes/moves onto a path matching `target` (regex)."""
    tgt = r"[^\s\"';|&]*(?:" + target + r")"
    return re.compile(
        r"(?:(?<![0-9&])>{1,2}\s*[\"']?" + tgt
        + r"|\b(?:tee|truncate|rm|rmdir|rd|del|erase|unlink|shred|touch|ln|Remove-Item|Set-Content|Add-Content|Out-File|New-Item)\b[^\n;|&]*" + tgt
        + r"|\bdd\b[^\n;|&]*\bof=[\"']?" + tgt
        + r"|\b(?:sed|perl)\s+(?:-\w+\s+)*-\w*i\w*\b[^\n;|&]*" + tgt
        + r"|\b(?:cp|mv|rsync|install|Copy-Item|Move-Item|Rename-Item)\b[^\n;|&]*" + tgt
        + r"|(?:write_text|write_bytes|json\.dump|os\.remove|os\.unlink|os\.replace|os\.rename|shutil\.\w+|IO\.File\]::Write\w*|open\s*\([^)]*['\"][wa]b?\+?['\"])[^\n]*" + tgt
        + r"|" + tgt + r"[^\n]*(?:write_text|write_bytes|['\"][wa]b?\+?['\"]\s*\)))",
        re.I | re.M,
    )


_EVIDENCE_WRITE_RE = _shell_write_re(r"seal\.json|verify\.json|episode_ledger\.json")
_CONFIG_DIR_OR_FILE = (
    r"\.claude(?:[/\\][^\s\"';|&]*)?|"
    r"\.codex(?:[/\\][^\s\"';|&]*)?|"
    r"\.qwen(?:[/\\][^\s\"';|&]*)?|"
    r"\.agents(?:[/\\](?:hooks\.json|plugins[/\\]weaveprofiler\b[^\s\"';|&]*|skills[/\\]weaveprofiler\b[^\s\"';|&]*))?|"
    r"\.mcp\.json"
)
_CONFIG_WRITE_RE = re.compile(
    _shell_write_re(_CONFIG_DIR_OR_FILE).pattern
    + r"|\bgit\s+clean\b[^\n;|&]*-[^\s]*[dx]"
    + r"|\b(?:cd|Set-Location|sl)\b[^\n;|&]*(?:\.claude|\.codex|\.qwen|\.agents)\b[^\n]*(?:rm|del|mv|Remove-Item|Set-Content|truncate|>|write_text|write_bytes|unlink|shutil)",
    re.I | re.M,
)


# ---------------------------------------------------------------------------
# Session location & persistence
# ---------------------------------------------------------------------------

def sessions_root(override: Optional[os.PathLike | str] = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("WEAVEPROF_SESSIONS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return (Path(base) / "weaveprof" / "sessions").resolve()


def workspace_id(workspace: os.PathLike | str) -> str:
    resolved = os.path.normcase(str(Path(workspace).expanduser().resolve()))
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def session_dir_for(workspace: os.PathLike | str, root: Optional[os.PathLike | str] = None) -> Path:
    return sessions_root(root) / workspace_id(workspace)


def resolve_session_dir(
    *, session_dir: Optional[str] = None, workspace: Optional[str] = None,
    cwd: Optional[str] = None, root: Optional[str] = None,
) -> Path:
    if session_dir:
        return Path(session_dir).expanduser().resolve()
    if workspace:
        return session_dir_for(workspace, root)
    start = Path(cwd or os.getcwd()).resolve()
    for cand in (start, *start.parents):
        sd = session_dir_for(cand, root)
        if (sd / SESSION_FILE).is_file():
            return sd
    return session_dir_for(start, root)


def sha256_file(path: os.PathLike | str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _atomic_write_bytes(path: Path, data: bytes, mode: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.wp-{uuid.uuid4().hex[:12]}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        with contextlib.suppress(OSError):
            os.fsync(fh.fileno())
    if mode is not None:
        with contextlib.suppress(OSError):
            os.chmod(tmp, mode)
    last: Optional[BaseException] = None
    for _ in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # Windows: target briefly open by another process
            last = exc
            time.sleep(0.025)
    with contextlib.suppress(OSError):
        tmp.unlink()
    raise last if last else OSError(f"cannot replace {path}")


def write_json_atomic(path: os.PathLike | str, doc: Any) -> None:
    text = json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False, default=str) + "\n"
    _atomic_write_bytes(Path(path), text.encode("utf-8"), 0o600)


def read_session(session_dir: os.PathLike | str) -> Optional[Dict[str, Any]]:
    p = Path(session_dir) / SESSION_FILE
    try:
        doc = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


@contextlib.contextmanager
def session_lock(session_dir: os.PathLike | str, timeout: float = 10.0) -> Iterator[None]:
    lock = Path(session_dir) / ".session.lock"
    deadline = time.monotonic() + timeout
    fd: Optional[int] = None
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            with contextlib.suppress(OSError):
                if time.time() - lock.stat().st_mtime > 30.0:
                    lock.unlink()
                    continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out acquiring session lock {lock} after {timeout}s")
            time.sleep(0.02)
        except OSError as exc:
            raise OSError(f"Cannot acquire session lock {lock}: {exc}") from exc
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
            with contextlib.suppress(OSError):
                lock.unlink()


def update_session(session_dir: os.PathLike | str, fn) -> Optional[Dict[str, Any]]:
    """Read-modify-write session.json under an advisory lock. `fn(doc)` mutates in place."""
    sd = Path(session_dir)
    with session_lock(sd):
        doc = read_session(sd)
        if doc is None:
            return None
        fn(doc)
        doc["updated_at"] = round(time.time(), 3)
        write_json_atomic(sd / SESSION_FILE, doc)
        return doc


@contextlib.contextmanager
def supervisor_env(session_dir: os.PathLike | str) -> Iterator[None]:
    """Temporarily point harness HMAC sealing/verification at the session key."""
    saved = {k: os.environ.get(k) for k in ("WEAVEPROF_SUPERVISOR_KEY_FILE", "WEAVEPROF_SUPERVISOR_HMAC_KEY")}
    os.environ["WEAVEPROF_SUPERVISOR_KEY_FILE"] = str(Path(session_dir) / KEY_FILE)
    os.environ.pop("WEAVEPROF_SUPERVISOR_HMAC_KEY", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _read_key(session_dir: os.PathLike | str) -> Optional[bytes]:
    try:
        raw = (Path(session_dir) / KEY_FILE).read_bytes().strip()
    except OSError:
        return None
    return raw or None


_POLICY_SIGNED_FIELDS: Tuple[str, ...] = (
    "schema", "workspace", "protected", "config_files",
    "require_final_seal", "png_only", "max_overhead", "bench_cmd", "card", "arch", "skill_root",
)


_MIN_PNG_WIDTH = 320
_MIN_PNG_HEIGHT = 160


def _is_valid_png(path: os.PathLike | str) -> bool:
    try:
        p = Path(path)
        if not p.is_file():
            return False
        st = p.stat()
        if st.st_size < 67:
            return False
        with open(p, "rb") as fh:
            hdr = fh.read(24)
        if len(hdr) < 24 or hdr[:8] != _PNG_MAGIC or hdr[12:16] != b"IHDR":
            return False
        width, height = struct.unpack(">II", hdr[16:24])
        return width >= _MIN_PNG_WIDTH and height >= _MIN_PNG_HEIGHT
    except OSError:
        return False


def _is_forbidden_gz_path(path: os.PathLike | str) -> bool:
    low = str(path).replace("\\", "/").lower()
    return any(low.endswith(suf) for suf in FORBIDDEN_GZ_SUFFIXES)


def _allow_gz_env() -> bool:
    return os.environ.get("WEAVEPROF_ALLOW_GZ", "").strip() == "1"


def _session_png_only(session: Optional[Dict[str, Any]]) -> bool:
    if _allow_gz_env():
        return False
    if session is None:
        return True
    return bool(session.get("png_only", True))


def _scan_dir_artifacts(
    root: Path, *, since_ts: float = 0.0, max_depth: int = 3, max_files: int = 800,
) -> Tuple[List[Path], List[Path]]:
    """Fast bounded scan returning (forbidden_gz_files, valid_png_files) created/modified >= since_ts."""
    bad_gz: List[Path] = []
    good_png: List[Path] = []
    try:
        root = root.resolve()
        if not root.is_dir():
            return bad_gz, good_png
        # Never scan or delete files inside the WeaveProfiler skill package itself.
        if _under(root, SKILL_ROOT):
            return bad_gz, good_png
    except OSError:
        return bad_gz, good_png
    seen = 0
    base_depth = len(root.parts)
    for cur_root, dirs, files in os.walk(root):
        cur_p = Path(cur_root)
        depth = len(cur_p.parts) - base_depth
        if depth >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = [
                d for d in dirs
                if d not in _SCAN_SKIP_DIRS and not d.startswith(".")
                and not _under(cur_p / d, SKILL_ROOT)
            ]
        for fn in files:
            seen += 1
            if seen > max_files:
                return bad_gz, good_png
            low = fn.lower()
            is_gz = any(low.endswith(suf) for suf in FORBIDDEN_GZ_SUFFIXES)
            is_png = low.endswith(".png")
            if not (is_gz or is_png):
                continue
            fp = cur_p / fn
            try:
                st = fp.stat()
            except OSError:
                continue
            if since_ts > 0.0 and st.st_mtime < since_ts:
                continue
            if is_gz:
                bad_gz.append(fp)
            elif is_png and _is_valid_png(fp):
                good_png.append(fp)
    return bad_gz, good_png


def _purge_forbidden_gz(paths: Sequence[Path]) -> List[str]:
    removed: List[str] = []
    for p in paths:
        try:
            if _under(p, SKILL_ROOT):
                continue
            if p.is_file():
                p.unlink()
                removed.append(str(p))
        except OSError:
            pass
    return removed


def _extract_candidate_dirs(text: str, cwd: Path) -> List[Path]:
    """Extract existing local directories referenced in Windows or WSL (/mnt/<drive>/...) paths."""
    dirs: List[Path] = []
    seen: set[str] = set()
    home = Path.home().resolve()
    broad_skip = {
        home,
        (home / "work").resolve(),
        (home / ".cache").resolve(),
        (home / ".cache" / "weaveprof").resolve(),
        (home / ".gemini").resolve(),
        (home / ".qwen").resolve(),
        (home / ".claude").resolve(),
        (home / ".codex").resolve(),
    }

    def _add(p: Path) -> None:
        try:
            cand = p if p.is_dir() else p.parent
            if cand.is_dir():
                res = cand.resolve()
                if len(res.parts) <= 2 or res in broad_skip or _under(res, SKILL_ROOT):
                    return
                # Skip top-level .gemini/.cache/.qwen directories
                if any(_under(res, b) and res == b for b in broad_skip):
                    return
                k = os.path.normcase(str(res))
                if k not in seen:
                    seen.add(k)
                    dirs.append(res)
        except OSError:
            pass

    if os.name == "nt":
        for drv, rest in _WSL_MNT_RE.findall(text):
            win_str = f"{drv.upper()}:{rest.replace('/', os.sep)}"
            _add(Path(win_str))
        for raw_win in _WIN_PATH_RE.findall(text):
            _add(Path(raw_win))
    else:
        for m in re.finditer(r"(/[^\s\"'`;|&)>]+)", text):
            _add(Path(m.group(1)))
    _add(cwd)
    return dirs


def _seal_has_valid_png(seal_entry: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    sp = Path(str(seal_entry.get("seal_path") or ""))
    if sp.is_file():
        try:
            doc = json.loads(sp.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            doc = {}
        arts = doc.get("artifacts") if isinstance(doc.get("artifacts"), dict) else {}
        for k in ("overview_png", "preview_png", "trace"):
            a = arts.get(k)
            if isinstance(a, dict) and isinstance(a.get("path"), str):
                cand = Path(a["path"])
                if cand.suffix.lower() == ".png" and _is_valid_png(cand):
                    return True, str(cand)
        for png in sorted(sp.parent.glob("*.png")):
            if _is_valid_png(png):
                return True, str(png)
    return False, None


def compute_policy_hmac(session_dir: os.PathLike | str, session: Dict[str, Any]) -> Optional[str]:
    """Compute HMAC-SHA256 over immutable session policy & baseline fields."""
    key = _read_key(session_dir)
    if key is None:
        return None
    body = {k: session.get(k) for k in _POLICY_SIGNED_FIELDS}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hmac.new(key, raw, hashlib.sha256).hexdigest()


def verify_session_policy(session_dir: os.PathLike | str, session: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Fail-closed integrity check for session.json immutable policy fields."""
    if not isinstance(session, dict):
        return False, "Session JSON missing or invalid"
    claimed = session.get("policy_hmac")
    if not isinstance(claimed, str) or not claimed:
        return False, "Session policy_hmac missing (session.json tampered or not provisioned with integrity protection)"
    expected = compute_policy_hmac(session_dir, session)
    if expected is None:
        return False, "Session supervisor key missing; cannot verify session policy_hmac"
    if not hmac.compare_digest(claimed, expected):
        return False, "Session policy_hmac mismatch (immutable policy/baseline fields in session.json were tampered)"
    return True, ""


# ---------------------------------------------------------------------------
# Seal verification & registration
# ---------------------------------------------------------------------------

_VERIFY_CACHE: Dict[Tuple[Any, ...], Tuple[bool, List[str], Optional[Dict[str, Any]], Tuple[Tuple[str, int, int], ...]]] = {}


def _bound_artifact_stats(doc: Optional[Dict[str, Any]]) -> Optional[Tuple[Tuple[str, int, int], ...]]:
    if not isinstance(doc, dict):
        return None
    artifacts = doc.get("artifacts")
    if not isinstance(artifacts, dict):
        return ()
    stats: List[Tuple[str, int, int]] = []
    for name, entry in sorted(artifacts.items()):
        if name == "source" or not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        try:
            st = Path(entry["path"]).stat()
        except OSError:
            return None
        stats.append((entry["path"], st.st_mtime_ns, st.st_size))
    return tuple(stats)


def verify_seal_file(
    session_dir: os.PathLike | str, seal_path: os.PathLike | str, *,
    require_pass: bool = True, max_overhead_limit: Optional[float] = None,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """HMAC (session key) + harness `verify_seal` check.

    A later edit of the profiled source only invalidates the *source binding* of an older
    seal, not the fact that it was verified; that freshness question is answered by the
    `protected_shas` snapshot comparison (Stop gate), so source-binding drift is tolerated.
    """
    sp = Path(seal_path)
    try:
        st = sp.stat()
    except OSError:
        return False, [f"Seal file missing: {sp}"], None
    key = _read_key(session_dir)
    if key is None:
        return False, ["Session supervisor key missing; cannot verify HMAC (fail-closed)"], None
    key_fp = hashlib.sha256(key).hexdigest()[:16]
    lim_key = round(float(max_overhead_limit), 8) if max_overhead_limit is not None else None
    cache_key = (str(sp.resolve()), st.st_mtime_ns, st.st_size, int(bool(require_pass)), key_fp, lim_key)
    hit = _VERIFY_CACHE.get(cache_key)
    if hit is not None:
        cached_ok, cached_errs, cached_doc, cached_stats = hit
        live_stats = _bound_artifact_stats(cached_doc)
        if live_stats is not None and live_stats == cached_stats:
            return cached_ok, cached_errs, cached_doc
    from weaveprof.harness import verify_seal

    with supervisor_env(session_dir):
        kwargs: Dict[str, Any] = {"require_pass": require_pass, "require_hmac": True}
        if max_overhead_limit is not None:
            kwargs["max_overhead_limit"] = float(max_overhead_limit)
        res = verify_seal(sp, **kwargs)
    errors = [e for e in (res.get("errors") or []) if not str(e).startswith("Bound artifact 'source'")]
    doc = res.get("seal") if isinstance(res.get("seal"), dict) else None
    if doc is None:
        errors = errors or ["Seal JSON unreadable"]
    else:
        if doc.get("mode") == "precaptured":
            errors.append("Precaptured seals cannot unlock supervised agent edits or Stop gates")
        claimed = doc.get("seal_hmac")
        expected = hmac.new(key, str(doc.get("seal_sha256") or "").encode("utf-8"), hashlib.sha256).hexdigest()
        if not isinstance(claimed, str) or not hmac.compare_digest(claimed, expected):
            msg = "Seal supervisor HMAC mismatch (possible forgery)"
            if msg not in errors:
                errors.append(msg)
    out = (not errors, errors, doc)
    live_stats = _bound_artifact_stats(doc)
    if live_stats is not None:
        _VERIFY_CACHE[cache_key] = (*out, live_stats)
    return out


def live_protected_shas(session: Dict[str, Any]) -> Dict[str, Optional[str]]:
    return {str(e.get("rel")): sha256_file(e.get("path", "")) for e in session.get("protected") or []}


def register_seal(
    session_dir: os.PathLike | str, seal_path: os.PathLike | str, *,
    kernel_us: Optional[float] = None, tool: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify a supervisor seal and record it in session.json (PASS resets `stop_blocks`)."""
    sd = Path(session_dir)
    session = read_session(sd)
    if session is None:
        return {"registered": False, "errors": [f"No WeaveProfiler session at {sd}"]}
    pol_ok, pol_err = verify_session_policy(sd, session)
    if not pol_ok:
        return {"registered": False, "errors": [f"SESSION_POLICY_TAMPERED: {pol_err}"]}
    sp = Path(seal_path).expanduser().resolve()
    ok_pass, errs_pass, doc = verify_seal_file(sd, sp, require_pass=True,
                                               max_overhead_limit=session.get("max_overhead"))
    if ok_pass:
        ok, errs = True, []
    else:
        ok, errs, doc = verify_seal_file(sd, sp, require_pass=False,
                                         max_overhead_limit=session.get("max_overhead"))
    if not ok or doc is None:
        return {"registered": False, "errors": errs or errs_pass}
    status = "PASS" if ok_pass else str(doc.get("status") or "FAIL")
    src = (doc.get("artifacts") or {}).get("source") or {}
    live = live_protected_shas(session)
    source_sha = src.get("sha256")
    covers = bool(source_sha) and source_sha in set(v for v in live.values() if v)
    prior_valid = valid_pass_seals(sd, session)
    baseline_shas = {str(e.get("baseline_sha256")) for e in session.get("protected") or [] if e.get("baseline_sha256")}
    first_must_match_baseline = (status == "PASS" and covers and not prior_valid
                                 and bool(baseline_shas) and source_sha not in baseline_shas)
    if first_must_match_baseline:
        covers = False
    ovh = doc.get("overhead_ratio")
    try:
        overhead_pct = round(float(ovh) * 100.0, 4) if ovh is not None else None
    except (TypeError, ValueError):
        overhead_pct = None
    try:
        kus = float(kernel_us) if kernel_us is not None else None
    except (TypeError, ValueError):
        kus = None
    entry = {
        "seal_path": str(sp),
        "seal_sha256": doc.get("seal_sha256"),
        "source_path": src.get("path"),
        "source_sha256": source_sha,
        "protected_shas": live,
        "covers_protected": covers,
        "status": status,
        "overhead_pct": overhead_pct,
        "kernel_us": kus,
        "tool": tool,
        "timestamp": round(time.time(), 3),
    }

    def _apply(d: Dict[str, Any]) -> None:
        seals = [s for s in d.get("seals") or [] if s.get("seal_sha256") != entry["seal_sha256"]]
        seals.append(entry)
        d["seals"] = seals
        if status == "PASS" and covers:
            d["stop_blocks"] = 0

    update_session(sd, _apply)
    note = None
    if first_must_match_baseline:
        note = ("First PASS seal must match the unmodified baseline SHA-256 of a protected kernel file; "
                "restore the baseline and seal it before optimizing.")
    elif status == "PASS" and not covers:
        note = ("PASS seal recorded but its source SHA-256 matches no protected file's current bytes; "
                "it does not unlock edits/stop. Profile the protected kernel itself.")
    return {"registered": True, "status": status, "covers_protected": covers, "entry": entry, "note": note,
            "errors": errs_pass if status != "PASS" else []}


def valid_pass_seals(session_dir: os.PathLike | str, session: Dict[str, Any]) -> List[Dict[str, Any]]:
    pol_ok, _ = verify_session_policy(session_dir, session)
    if not pol_ok:
        return []
    out = []
    for entry in session.get("seals") or []:
        if entry.get("status") != "PASS" or not entry.get("covers_protected"):
            continue
        ok, _errs, doc = verify_seal_file(session_dir, entry.get("seal_path", ""), require_pass=True,
                                          max_overhead_limit=session.get("max_overhead"))
        if ok and doc is not None and doc.get("seal_sha256") == entry.get("seal_sha256"):
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Protected-file inspection & restore
# ---------------------------------------------------------------------------

def _read_text(path: os.PathLike | str) -> str:
    try:
        return Path(path).read_bytes().decode("utf-8", "replace")
    except OSError:
        return ""


def new_probe_tokens(entry: Dict[str, Any], text: Optional[str] = None) -> List[str]:
    live = _read_text(entry.get("path", "")) if text is None else text
    base = _read_text(entry.get("backup", ""))
    return [t for t in PROBE_TOKENS if t in live and t not in base]


def restore_protected(session: Dict[str, Any], rels: Optional[Sequence[str]] = None) -> List[str]:
    restored = []
    for e in session.get("protected") or []:
        if rels is not None and e.get("rel") not in rels:
            continue
        backup = Path(e.get("backup", ""))
        target = Path(e.get("path", ""))
        try:
            data = backup.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(data).hexdigest() != e.get("baseline_sha256"):
            continue  # never restore from a tampered backup
        if sha256_file(target) == e.get("baseline_sha256"):
            continue
        mode = None
        with contextlib.suppress(OSError):
            mode = backup.stat().st_mode & 0o777
        _atomic_write_bytes(target, data, mode)
        restored.append(str(e.get("rel")))
    return restored


def mutated_since_baseline(session: Dict[str, Any]) -> List[str]:
    return [str(e.get("rel")) for e in session.get("protected") or []
            if sha256_file(e.get("path", "")) != e.get("baseline_sha256")]


def mutated_config_files(session: Dict[str, Any]) -> List[str]:
    return [str(e.get("rel")) for e in session.get("config_files") or []
            if sha256_file(e.get("path", "")) != e.get("sha256")]


def restore_config_files(session: Dict[str, Any], rels: Optional[Sequence[str]] = None) -> List[str]:
    restored = []
    for e in session.get("config_files") or []:
        if rels is not None and e.get("rel") not in rels:
            continue
        backup = Path(e.get("backup", ""))
        target = Path(e.get("path", ""))
        try:
            data = backup.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(data).hexdigest() != e.get("sha256"):
            continue
        if sha256_file(target) == e.get("sha256"):
            continue
        _atomic_write_bytes(target, data, 0o644)
        restored.append(str(e.get("rel")))
    return restored


def incumbent(seals: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    timed = [s for s in seals if isinstance(s.get("kernel_us"), (int, float))]
    if not timed:
        return None
    best = min(timed, key=lambda s: float(s["kernel_us"]))
    return {"kernel_us": best["kernel_us"], "seal_sha256": best.get("seal_sha256"),
            "overhead_pct": best.get("overhead_pct"), "protected_shas": best.get("protected_shas")}


def verify_session_completion(
    workspace: Optional[str] = None, *, session_dir: Optional[str] = None, sessions_root_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Exact Stop-gate verdict: intact session policy, >=1 verified PASS seal, probe-free sources, final bytes sealed."""
    sd = resolve_session_dir(session_dir=session_dir, workspace=workspace, root=sessions_root_dir)
    session = read_session(sd)
    if session is None:
        return {"ok": False, "installed": False, "session_dir": str(sd),
                "blockers": [f"SESSION_MISSING: no WeaveProfiler session at {sd}; run the wp installer."]}
    pol_ok, pol_err = verify_session_policy(sd, session)
    if not pol_ok:
        return {"ok": False, "installed": True, "session_dir": str(sd), "workspace": session.get("workspace"),
                "blockers": [f"SESSION_POLICY_TAMPERED: {pol_err}"], "valid_pass_seals": 0}
    valid = valid_pass_seals(sd, session)
    live = live_protected_shas(session)
    blockers: List[str] = []
    cfg_mut = mutated_config_files(session)
    if cfg_mut:
        restored_cfg = restore_config_files(session, cfg_mut)
        blockers.append(
            f"SUPERVISOR_CONFIG_TAMPER: hook/MCP config files {cfg_mut} were modified or deleted "
            f"(restored: {restored_cfg})."
        )
    if not valid:
        blockers.append("NO_VERIFIED_PASS_SEAL: no HMAC-verified PASS seal of the protected kernel exists yet; call "
                        + SEALING_TOOLS_HINT + ".")
    polluted = {}
    for e in session.get("protected") or []:
        toks = new_probe_tokens(e)
        if toks:
            polluted[str(e.get("rel"))] = toks
    if polluted:
        blockers.append(f"PROBE_POLLUTION: protected files contain manual probe macros {polluted}; strip them "
                        "(production kernel source must stay probe-free).")
    matched = None
    if valid and session.get("require_final_seal", True):
        for entry in reversed(valid):
            snap = entry.get("protected_shas") or {}
            if snap and all(snap.get(rel) == sha for rel, sha in live.items()):
                matched = entry
                break
        if matched is None:
            latest = valid[-1].get("protected_shas") or {}
            changed = [rel for rel, sha in live.items() if latest.get(rel) != sha]
            blockers.append(
                f"UNSEALED_FINAL_KERNEL: protected kernel was modified after the last verified seal ({changed}); call "
                + SEALING_TOOLS_HINT + " to verify & seal the optimized kernel (or revert the regression to a "
                "sealed incumbent) before stopping.")
    elif valid:
        matched = valid[-1]
    png_only = _session_png_only(session)
    if png_only:
        ws_path = Path(str(session.get("workspace") or "")).resolve()
        since_ts = float(session.get("created_at") or 0.0) - 1.0
        bad_gz: List[Path] = []
        gz_ws, _ = _scan_dir_artifacts(ws_path, since_ts=since_ts, max_depth=3)
        bad_gz.extend(gz_ws)
        if matched and matched.get("seal_path"):
            gz_run, _ = _scan_dir_artifacts(Path(matched["seal_path"]).parent, since_ts=since_ts, max_depth=2)
            bad_gz.extend(gz_run)
        if bad_gz:
            removed = _purge_forbidden_gz(bad_gz)
            blockers.append(
                f"FORBIDDEN_GZ_ARTIFACT: PNG-only delivery policy forbids `.perfetto*.gz` / `.trace.json.gz` "
                f"files (removed {removed[:4]}); deliver a timeline PNG (`<model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png`) instead."
            )
        if matched is not None:
            has_png, _png_path = _seal_has_valid_png(matched)
            if not has_png:
                blockers.append(
                    "MISSING_OVERVIEW_PNG: PNG-only Stop gate requires a verified, non-empty `.png` deliverable "
                    "(`<model>_<op>_<shape>_<YYYYMMDD_HHMMSS>.png`) bound to the PASS seal; re-run profiling with "
                    "`render_overview_png=True` before stopping."
                )
    return {
        "ok": not blockers,
        "installed": True,
        "session_dir": str(sd),
        "workspace": session.get("workspace"),
        "blockers": blockers,
        "valid_pass_seals": len(valid),
        "matched_seal_sha256": matched.get("seal_sha256") if matched else None,
        "live_shas": live,
        "incumbent": incumbent(valid),
        "stop_blocks": int(session.get("stop_blocks") or 0),
    }


def status_report(session_dir: os.PathLike | str) -> Dict[str, Any]:
    sd = Path(session_dir)
    session = read_session(sd)
    if session is None:
        return {"installed": False, "session_id": sd.name,
                "hint": "Workspace not provisioned; run `python -B <skill_root>/run.py install-agent --workspace <ws>`."}
    valid = valid_pass_seals(sd, session)
    latest = valid[-1] if valid else None
    comp = verify_session_completion(session_dir=str(sd))
    files = []
    for e in session.get("protected") or []:
        live = sha256_file(e.get("path", ""))
        sealed = (latest.get("protected_shas") or {}).get(e.get("rel")) if latest else None
        files.append({
            "rel": e.get("rel"), "path": e.get("path"),
            "baseline_sha256": e.get("baseline_sha256"), "live_sha256": live,
            "latest_sealed_sha256": sealed,
            "modified_since_baseline": live != e.get("baseline_sha256"),
            "matches_latest_seal": bool(sealed) and live == sealed,
            "new_probe_tokens": new_probe_tokens(e),
        })
    return {
        "installed": True,
        "session_id": sd.name,
        "workspace": session.get("workspace"),
        "protected": files,
        "seals_recorded": len(session.get("seals") or []),
        "valid_pass_seals": len(valid),
        "edit_unlocked": bool(valid),
        "stop_unlocked": bool(comp.get("ok")),
        "stop_blockers": comp.get("blockers"),
        "require_final_seal": bool(session.get("require_final_seal", True)),
        "max_overhead": session.get("max_overhead"),
        "bench_cmd": session.get("bench_cmd"),
        "card": session.get("card"),
        "arch": session.get("arch"),
        "incumbent": incumbent(valid),
        "next_step": ("Call " + SEALING_TOOLS_HINT + " on the baseline kernel to unlock edits.") if not valid else (
            "Edits unlocked. After each optimization re-run " + SEALING_TOOLS_HINT + " so the final kernel is sealed."),
    }


# ---------------------------------------------------------------------------
# Tool-input parsing
# ---------------------------------------------------------------------------

def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _norm(s: str) -> str:
    return s.replace("\\", "/").lower()


def _tool_kind(tool_name: str) -> Tuple[str, bool]:
    """Return (normalized short tool name, is_own_weaveprofiler_mcp_tool)."""
    name = (tool_name or "").strip()
    low = name.lower()
    if low.startswith("mcp__"):
        own = "weaveprofiler" in low or "weaveprof_" in low
        return low.rsplit("__", 1)[-1], own
    if low.startswith("mcp_weaveprofiler_"):
        return low[len("mcp_weaveprofiler_"):], True
    if low.startswith("weaveprof_"):
        return low, True
    return low, False


def _extract_tool(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Extract (tool_name, tool_input_dict) from Claude/Codex/Qwen or Antigravity camelCase payloads."""
    tool_call = payload.get("toolCall") if isinstance(payload.get("toolCall"), dict) else None
    tool_name = str(payload.get("tool_name") or (tool_call.get("name") if tool_call else "") or "")
    tool_input = payload.get("tool_input")
    if tool_input is None and tool_call is not None:
        tool_input = tool_call.get("args")
    if isinstance(tool_input, str):
        tool_input = {"command": tool_input}
    return tool_name, (tool_input if isinstance(tool_input, dict) else {})


def _iter_ci_values(mapping: Dict[str, Any], keys: Sequence[str]) -> Iterator[Any]:
    wanted = {k.lower() for k in keys}
    for k, v in mapping.items():
        if str(k).lower() in wanted:
            yield v


def _shell_command(tool_input: Dict[str, Any]) -> str:
    parts: List[str] = []
    for v in _iter_ci_values(tool_input, ("command", "cmd", "commandline", "command_line",
                                          "bench", "bench_cmd", "build_cmd")):
        if isinstance(v, (list, tuple)):
            parts.append(" ".join(str(c) for c in v))
        elif isinstance(v, str) and v.strip():
            parts.append(v)
    return "\n".join(parts)


def _own_output_targets(tool_input: Dict[str, Any], cwd: Path) -> List[Path]:
    out: List[Path] = []
    for v in _iter_ci_values(tool_input, ("out_dir", "out", "run_dir")):
        if isinstance(v, str) and v.strip() and "\n" not in v:
            out.append(_resolve(v, cwd))
    return out


def _resolve(p: str, cwd: Path) -> Path:
    q = Path(os.path.expanduser(p.strip().strip("\"'")))
    if not q.is_absolute():
        q = cwd / q
    try:
        return q.resolve()
    except OSError:
        return Path(os.path.abspath(str(q)))


def _parse_patch(text: str, cwd: Path) -> List[Tuple[Path, str, str]]:
    out: List[Tuple[Path, str, str]] = []
    cur: Optional[List[Any]] = None
    diff_minus: Optional[str] = None
    for line in text.splitlines():
        m = _PATCH_HDR_RE.match(line)
        if m:
            cur = [_resolve(m.group(2), cwd), [], "delete" if m.group(1) == "Delete" else "write"]
            out.append(cur)  # type: ignore[arg-type]
            continue
        m = _PATCH_MOVE_RE.match(line)
        if m and cur is not None:
            cur = [_resolve(m.group(1), cwd), cur[1], "write"]
            out.append(cur)  # type: ignore[arg-type]
            continue
        if line.startswith("--- "):
            diff_minus = line[4:].strip()
            continue
        if line.startswith("+++ "):
            tgt = line[4:].strip().split("\t")[0]
            op = "write"
            if tgt == "/dev/null" and diff_minus:
                tgt, op = diff_minus, "delete"
            for pre in ("a/", "b/"):
                if tgt.startswith(pre):
                    tgt = tgt[2:]
            cur = [_resolve(tgt, cwd), [], op]
            out.append(cur)  # type: ignore[arg-type]
            continue
        if cur is not None and line.startswith("+"):
            cur[1].append(line[1:])
    return [(c[0], "\n".join(c[1]), c[2]) for c in out]


def edit_targets(tool_name: str, tool_input: Any, cwd: Path) -> List[Tuple[Path, str, str]]:
    """Files a tool call proposes to write: [(resolved_path, added_text, 'write'|'delete')]."""
    short, own = _tool_kind(tool_name)
    if own or not isinstance(tool_input, dict):
        return []
    targets: List[Tuple[Path, str, str]] = []
    if short in EDIT_TOOLS:
        added_parts = []
        for v in _iter_ci_values(tool_input, ("content", "new_string", "new_source", "file_text", "new_str",
                                              "contents", "text", "code", "replacement", "replacementcontent",
                                              "codecontent")):
            if isinstance(v, str):
                added_parts.append(v)
        for lst in _iter_ci_values(tool_input, ("edits", "replacementchunks", "replacement_chunks")):
            if not isinstance(lst, list):
                continue
            for ed in lst:
                if isinstance(ed, dict):
                    for v in _iter_ci_values(ed, ("new_string", "new_str", "content", "replacement",
                                                  "replacementcontent")):
                        if isinstance(v, str):
                            added_parts.append(v)
        for v in _iter_ci_values(tool_input, ("file_path", "path", "notebook_path", "filename",
                                              "target_file", "targetfile", "file")):
            if isinstance(v, str) and v.strip() and "\n" not in v:
                targets.append((_resolve(v, cwd), "\n".join(added_parts), "write"))
                break
    for s in _strings(tool_input):
        if "*** Begin Patch" in s or ("\n+++ " in s and "\n--- " in s) or s.startswith("--- "):
            targets.extend(_parse_patch(s, cwd))
    return targets


# ---------------------------------------------------------------------------
# Hook event handlers
# ---------------------------------------------------------------------------

class HookBlock(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{TAG} BLOCKED ({code}): {message}")


def _same(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _under(p: Path, root: Path) -> bool:
    pa, ra = os.path.normcase(str(p)), os.path.normcase(str(root))
    return pa == ra or pa.startswith(ra.rstrip("\\/") + os.sep)


class Context:
    def __init__(self, payload: Dict[str, Any], args: argparse.Namespace):
        self.payload = payload
        self.cwd = Path(payload.get("cwd") or args.workspace or os.getcwd()).resolve()
        self.session_dir = resolve_session_dir(session_dir=args.session_dir, workspace=args.workspace,
                                               cwd=str(self.cwd), root=getattr(args, "sessions_root", None))
        self.session = read_session(self.session_dir)
        ws = (self.session or {}).get("workspace") or args.workspace or str(self.cwd)
        self.workspace = Path(ws).resolve()
        self._valid: Optional[List[Dict[str, Any]]] = None

    def protected(self) -> List[Dict[str, Any]]:
        return list((self.session or {}).get("protected") or [])

    def protected_entry(self, path: Path) -> Optional[Dict[str, Any]]:
        for e in self.protected():
            if _same(Path(e.get("path", "")), path):
                return e
        return None

    def valid_seals(self) -> List[Dict[str, Any]]:
        if self._valid is None:
            self._valid = valid_pass_seals(self.session_dir, self.session) if self.session else []
        return self._valid


def handle_pre(ctx: Context) -> None:
    payload = ctx.payload
    tool_name, tool_input = _extract_tool(payload)
    short, own = _tool_kind(tool_name)
    blob = _norm("\n".join(_strings(tool_input)))

    # (a) secrets / supervisor internals
    needles = list(SECRET_NEEDLES) + [_norm(str(ctx.session_dir)), _norm(str(ctx.session_dir.parent))]
    for n in needles:
        if n and n in blob:
            raise HookBlock("SECRET_OR_SUPERVISOR_ACCESS",
                            f"tool `{tool_name}` references protected supervisor state ({n!r}). The supervisor key, "
                            "session store, hook and MCP server are off-limits; use MCP `weaveprof_status` to inspect "
                            "session state.")
    is_shell = (short in SHELL_TOOLS) or (not own and short not in EDIT_TOOLS and short not in READ_ONLY_TOOLS)
    cmd = _shell_command(tool_input)
    all_vals_text = "\n".join(_strings(tool_input))
    scan_text = all_vals_text if is_shell else cmd
    if _SAMPLING_RE.search(all_vals_text):
        raise HookBlock("LOOP_SAMPLING_FORBIDDEN",
                        "WP_SAMPLE > 1 violates the full-audit policy; keep WP_SAMPLE at 0 or 1.")
    if scan_text and _INSTALLER_RE.search(scan_text):
        raise HookBlock("SUPERVISOR_RECONFIG",
                        "the agent may not re-run the wp installer / guard-exec (that would reset the baseline). "
                        "Ask the user to re-provision if the protected set must change.")
    targets = edit_targets(tool_name, tool_input, ctx.cwd)
    check_paths: List[Path] = [p for p, _a, _op in targets] + (_own_output_targets(tool_input, ctx.cwd) if own else [])
    ws_norm = os.path.normcase(str(ctx.workspace)).replace("\\", "/").rstrip("/")
    home_norm = os.path.normcase(str(Path.home().resolve())).replace("\\", "/").rstrip("/")
    for path in check_paths:
        name = path.name.lower()
        if name.endswith("seal.json") or name in ("verify.json", "episode_ledger.json"):
            raise HookBlock("EVIDENCE_TAMPER",
                            f"`{path.name}` is sealed harness evidence and cannot be written/deleted by the agent. "
                            "Use MCP `weaveprof_verify_seal` to check seals.")
        np = os.path.normcase(str(path)).replace("\\", "/")
        # `.agents/hooks.json` is protected even when WeaveProfiler ships its hooks inside the native
        # plugin bundle `.agents/plugins/weaveprofiler/hooks.json` so an agent cannot disable/override it.
        cfg = (np == ws_norm + "/.claude" or np.startswith(ws_norm + "/.claude/")
               or np == ws_norm + "/.codex" or np.startswith(ws_norm + "/.codex/")
               or np == ws_norm + "/.qwen" or np.startswith(ws_norm + "/.qwen/")
               or np == ws_norm + "/.mcp.json"
               or np == ws_norm + "/.agents/hooks.json"
               or np == ws_norm + "/.agents/plugins/weaveprofiler"
               or np.startswith(ws_norm + "/.agents/plugins/weaveprofiler/")
               or np == ws_norm + "/.agents/skills/weaveprofiler"
               or np.startswith(ws_norm + "/.agents/skills/weaveprofiler/")
               or np == home_norm + "/.claude" or np.startswith(home_norm + "/.claude/")
               or np == home_norm + "/.codex" or np.startswith(home_norm + "/.codex/")
               or np == home_norm + "/.qwen" or np.startswith(home_norm + "/.qwen/"))
        if cfg or _under(path, SKILL_ROOT) or _under(path, ctx.session_dir.parent):
            raise HookBlock("SUPERVISOR_CONFIG_TAMPER",
                            f"`{path}` is WeaveProfiler supervisor/agent configuration and is read-only for the agent.")
    if scan_text and _EVIDENCE_WRITE_RE.search(scan_text):
        raise HookBlock("EVIDENCE_TAMPER",
                        "command writes/moves/deletes seal.json / verify.json / episode_ledger.json evidence.")
    if scan_text and _CONFIG_WRITE_RE.search(scan_text):
        raise HookBlock("SUPERVISOR_CONFIG_TAMPER",
                        "command modifies `.claude/`, `.codex/`, `.qwen/`, `.agents/`, or `.mcp.json` (hook/MCP wiring).")

    # (a2) PNG-only deliverable enforcement: block generating `.perfetto*.gz` / `.trace.json.gz`
    if _session_png_only(ctx.session):
        for path in check_paths:
            if _is_forbidden_gz_path(path):
                raise HookBlock(
                    "FORBIDDEN_GZ_ARTIFACT",
                    f"PNG-only delivery is enforced: refusing to write `{path.name}`. "
                    "Render `overview.png` directly via `wp.profile(..., out='overview.png')`, "
                    "`weaveprof_megakernel_audit`, or `wp_decode.py ... -o overview.png`.",
                )
        if own and isinstance(tool_input, dict) and tool_input.get("write_perfetto_gz"):
            raise HookBlock(
                "FORBIDDEN_GZ_ARTIFACT",
                "PNG-only delivery is enforced: `write_perfetto_gz=True` is forbidden. "
                "Keep `render_overview_png=True` and `write_perfetto_gz=False`.",
            )
        if scan_text and not _GZ_CLEANUP_ONLY_RE.match(scan_text.strip()):
            if _FORBIDDEN_GZ_CMD_RE.search(scan_text):
                raise HookBlock(
                    "FORBIDDEN_GZ_ARTIFACT",
                    "PNG-only delivery is enforced: do NOT emit `.perfetto.json.gz` / `.trace.json.gz` / "
                    "`.perfetto-trace.gz` or call `export_perfetto_native.py` / `--write-gz`. "
                    "Render `overview.png` directly via `wp_decode.py ... -o overview.png` or `weaveprof/viz.py`.",
                )

    if ctx.session is None:
        if short in READ_ONLY_TOOLS or own:
            return
        if targets or is_shell:
            raise HookBlock("SESSION_MISSING",
                            f"no WeaveProfiler session found at {ctx.session_dir.name}; enforcement fails closed. "
                            "Ask the user to re-run the wp agent installer for this workspace.")
        return

    pol_ok, pol_err = verify_session_policy(ctx.session_dir, ctx.session)
    if not pol_ok and short not in READ_ONLY_TOOLS and not own:
        raise HookBlock("SESSION_POLICY_TAMPERED", pol_err)

    # (b) probe pollution + (c) seal-before-optimize for protected files
    protected_hits = [(p, a, op, ctx.protected_entry(p)) for p, a, op in targets]
    protected_hits = [h for h in protected_hits if h[3] is not None]
    for path, added, _op, entry in protected_hits:
        toks = [t for t in PROBE_TOKENS if t in added and t not in _read_text(entry.get("backup", ""))]
        if toks:
            raise HookBlock("PROBE_POLLUTION",
                            f"proposed edit to protected `{entry.get('rel')}` adds probe instrumentation {toks}. Keep "
                            "production kernel files clean; call MCP `weaveprof_megakernel_audit` or "
                            "`weaveprof_profile`, which instrument an isolated temporary copy.")
    shell_hits: List[Dict[str, Any]] = []
    if is_shell and scan_text:
        for e in ctx.protected():
            base = re.escape(Path(e.get("path", "")).name)
            if re.search(r"(?<![\w.-])" + base + r"(?![\w-])", scan_text):
                shell_hits.append(e)
        for e in shell_hits:
            toks = [t for t in PROBE_TOKENS if t in scan_text and t not in _read_text(e.get("backup", ""))]
            if toks:
                raise HookBlock("PROBE_POLLUTION",
                                f"shell command touching protected `{e.get('rel')}` carries probe tokens {toks}; keep "
                                "production kernel source probe-free and use the MCP profilers.")
    if protected_hits or shell_hits:
        writing_shell = [e for e in shell_hits
                         if _shell_write_re(re.escape(Path(e.get("path", "")).name)).search(scan_text)]
        if (protected_hits or writing_shell) and not ctx.valid_seals():
            rels = sorted({str(h[3].get("rel")) for h in protected_hits} | {str(e.get("rel")) for e in writing_shell})
            raise HookBlock("SEAL_BEFORE_OPTIMIZE",
                            f"protected kernel file(s) {rels} are locked until an HMAC-verified PASS seal of the "
                            "baseline exists. First call " + SEALING_TOOLS_HINT + " on the unmodified kernel; edits "
                            "unlock automatically once the seal is registered (check with `weaveprof_status`).")


def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _is_stop_active(payload: Dict[str, Any]) -> bool:
    return bool(
        payload.get("stop_hook_active")
        or payload.get("continuationActive")
        or _safe_int(payload.get("executionNum")) > 0
    )


def _detect_protocol(requested: Optional[str], payload: Dict[str, Any]) -> str:
    req = (requested or "auto").strip().lower()
    if req in ("claude", "codex", "qwen", "agy"):
        return req
    if req == "antigravity":
        return "agy"
    if any(k in payload for k in ("hookEventName", "toolCall", "continuationActive", "executionNum", "workspacePaths")):
        return "agy"
    if os.environ.get("QWEN_PROJECT_DIR"):
        return "qwen"
    return "claude"


def _set_pending_post_alert(session_dir: Path, message: Optional[str]) -> None:
    def _apply(d: Dict[str, Any]) -> None:
        if message:
            d["pending_post_alert"] = message
        else:
            d.pop("pending_post_alert", None)

    update_session(session_dir, _apply)


def _pop_pending_post_alert(session_dir: Path) -> Optional[str]:
    holder: Dict[str, Optional[str]] = {"msg": None}

    def _apply(d: Dict[str, Any]) -> None:
        val = d.pop("pending_post_alert", None)
        if isinstance(val, str) and val.strip():
            holder["msg"] = val

    update_session(session_dir, _apply)
    return holder["msg"]


def handle_post(ctx: Context, *, protocol: str = "claude") -> None:
    if ctx.session is None:
        return
    pol_ok, pol_err = verify_session_policy(ctx.session_dir, ctx.session)
    if not pol_ok:
        blk = HookBlock("SESSION_POLICY_TAMPERED", pol_err)
        if protocol == "agy":
            _set_pending_post_alert(ctx.session_dir, str(blk))
        raise blk
    cfg_mut = mutated_config_files(ctx.session)
    if cfg_mut:
        restored_cfg = restore_config_files(ctx.session, cfg_mut)
        blk = HookBlock(
            "SUPERVISOR_CONFIG_TAMPER",
            f"hook/MCP configuration file(s) {cfg_mut} were modified or deleted; exact installed copies "
            f"were restored ({restored_cfg}). Supervisor wiring is read-only for the agent.",
        )
        if protocol == "agy":
            _set_pending_post_alert(ctx.session_dir, str(blk))
        raise blk
    mutated = mutated_since_baseline(ctx.session)
    if mutated and not ctx.valid_seals():
        restored = restore_protected(ctx.session, mutated)
        blk = HookBlock(
            "UNAUTHORIZED_PRE_PROFILE_MUTATION",
            f"protected kernel file(s) {mutated} changed before any verified PASS seal existed (e.g. via "
            f"shell/script). The exact baseline bytes were restored ({restored}). Call "
            + SEALING_TOOLS_HINT + " on the baseline first; edits unlock after the seal.",
        )
        if protocol == "agy":
            _set_pending_post_alert(ctx.session_dir, str(blk))
        raise blk
    polluted = {}
    for e in ctx.protected():
        toks = new_probe_tokens(e)
        if toks:
            polluted[str(e.get("rel"))] = toks
    if polluted:
        blk = HookBlock(
            "PROBE_POLLUTION",
            f"protected kernel source now contains probe macros {polluted}. Remove them: production "
            "kernel source must stay probe-free; profiling instruments an isolated temporary copy via "
            + SEALING_TOOLS_HINT + ".",
        )
        if protocol == "agy":
            _set_pending_post_alert(ctx.session_dir, str(blk))
        raise blk
    if _session_png_only(ctx.session):
        since_ts = float(ctx.session.get("created_at") or 0.0) - 1.0
        bad_gz, _ = _scan_dir_artifacts(ctx.workspace, since_ts=since_ts, max_depth=3)
        _tn, ti = _extract_tool(ctx.payload)
        for cand_dir in _extract_candidate_dirs("\n".join(_strings(ti)), ctx.cwd):
            if not _under(cand_dir, ctx.workspace):
                extra_gz, _ = _scan_dir_artifacts(cand_dir, since_ts=since_ts, max_depth=2)
                bad_gz.extend(extra_gz)
        if bad_gz:
            removed = _purge_forbidden_gz(bad_gz)
            blk = HookBlock(
                "FORBIDDEN_GZ_ARTIFACT",
                f"PNG-only delivery policy forbids `.perfetto*.gz` / `.trace.json.gz` artifacts; "
                f"automatically removed {removed[:4]}. Render `overview.png` directly via `wp_decode.py -o overview.png` "
                "or `weaveprof/viz.py` instead.",
            )
            if protocol == "agy":
                _set_pending_post_alert(ctx.session_dir, str(blk))
            raise blk


def handle_post_invocation(ctx: Context) -> Dict[str, Any]:
    if ctx.session is None:
        return {}
    alert = _pop_pending_post_alert(ctx.session_dir)
    if not alert:
        return {}
    return {
        "injectSteps": [{"systemMessage": {"systemMessage": alert}}],
        "terminationBehavior": "force_continue",
    }


def handle_subagent_stop(ctx: Context, *, protocol: str = "claude") -> None:
    """SubagentStop checks only unauthorized mutations, config integrity, and probe pollution
    (subagents do not own top-level session sealing and must not share `stop_blocks`)."""
    if ctx.session is None:
        return
    pol_ok, pol_err = verify_session_policy(ctx.session_dir, ctx.session)
    if not pol_ok:
        raise HookBlock("SESSION_POLICY_TAMPERED", pol_err)
    cfg_mut = mutated_config_files(ctx.session)
    if cfg_mut:
        restored_cfg = restore_config_files(ctx.session, cfg_mut)
        raise HookBlock(
            "SUPERVISOR_CONFIG_TAMPER",
            f"subagent modified/deleted supervisor config file(s) {cfg_mut}; restored ({restored_cfg}).",
        )
    mutated = mutated_since_baseline(ctx.session)
    if mutated and not ctx.valid_seals():
        restored = restore_protected(ctx.session, mutated)
        raise HookBlock(
            "UNAUTHORIZED_PRE_PROFILE_MUTATION",
            f"subagent modified protected kernel file(s) {mutated} before any verified PASS seal; restored ({restored}).",
        )
    polluted = {}
    for e in ctx.protected():
        toks = new_probe_tokens(e)
        if toks:
            polluted[str(e.get("rel"))] = toks
    if polluted:
        raise HookBlock(
            "PROBE_POLLUTION",
            f"subagent left probe macros {polluted} in protected kernel source; strip them before finishing.",
        )


def handle_stop(ctx: Context, *, protocol: str = "claude") -> None:
    if protocol == "agy":
        term = str(ctx.payload.get("terminationReason") or "").strip().upper()
        if term not in _AGY_GATE_STOP_REASONS:
            return
    stop_active = _is_stop_active(ctx.payload)
    if ctx.session is None:
        if stop_active:
            sys.stderr.write(f"{TAG} WARNING: no session found; allowing stop to avoid a loop.\n")
            return
        raise HookBlock("SESSION_MISSING", f"no WeaveProfiler session at {ctx.session_dir.name}; cannot verify "
                                           "completion. Ask the user to re-run the wp agent installer.")
    comp = verify_session_completion(session_dir=str(ctx.session_dir))
    if comp["ok"]:
        return
    holder: Dict[str, int] = {}

    def _inc(d: Dict[str, Any]) -> None:
        base = int(d.get("stop_blocks") or 0) if stop_active else 0
        n = base + 1
        holder["n"] = n
        d["stop_blocks"] = 0 if n > STOP_BLOCK_LIMIT else n

    update_session(ctx.session_dir, _inc)
    n = holder.get("n", STOP_BLOCK_LIMIT + 1)
    if n > STOP_BLOCK_LIMIT:
        sys.stderr.write(f"{TAG} WARNING: stop gate unmet after {n - 1} blocked stops; allowing stop to avoid an "
                         f"infinite loop. Unresolved: {comp['blockers']}\n")
        return
    raise HookBlock("STOP_GATE", f"cannot finish yet (block {n}/{STOP_BLOCK_LIMIT}): " + " | ".join(comp["blockers"]))


def _emit_json(doc: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _emit_allow(protocol: str, event: str, extra: Optional[Dict[str, Any]] = None) -> int:
    if protocol == "agy":
        if event in ("PreToolUse", "Stop", "SubagentStop"):
            _emit_json({"decision": "allow"})
        elif event == "PostInvocation":
            _emit_json(extra or {})
        else:
            _emit_json({})
        return EXIT_ALLOW
    if protocol == "qwen":
        _emit_json({"decision": "allow"})
        return EXIT_ALLOW
    return EXIT_ALLOW


def _emit_block(protocol: str, event: str, message: str) -> int:
    if protocol == "agy":
        if event == "PreToolUse":
            _emit_json({"decision": "deny", "reason": message})
            return EXIT_ALLOW
        if event in ("Stop", "SubagentStop"):
            _emit_json({"decision": "continue", "reason": message})
            return EXIT_ALLOW
        if event == "PostInvocation":
            _emit_json({
                "injectSteps": [{"systemMessage": {"systemMessage": message}}],
                "terminationBehavior": "force_continue",
            })
            return EXIT_ALLOW
        # Antigravity PostToolUse contract expects `{}`; mutation was reverted on disk
        # and queued for PostInvocation system-message injection.
        sys.stderr.write(message + "\n")
        _emit_json({})
        return EXIT_ALLOW
    if protocol == "qwen":
        if event == "PreToolUse":
            _emit_json({
                "decision": "block",
                "reason": message,
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": message,
                },
            })
            return EXIT_ALLOW
        if event in ("PostToolUse", "PostToolUseFailure"):
            _emit_json({
                "decision": "block",
                "reason": message,
                "systemMessage": message,
                "hookSpecificOutput": {
                    "hookEventName": event,
                    "additionalContext": message,
                },
            })
            return EXIT_ALLOW
        if event in ("Stop", "SubagentStop"):
            _emit_json({
                "decision": "block",
                "reason": message,
                "stopReason": message,
                "systemMessage": message,
            })
            return EXIT_ALLOW
        _emit_json({"decision": "block", "reason": message})
        return EXIT_ALLOW
    sys.stderr.write(message + "\n")
    return EXIT_BLOCK


def _read_payload() -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8", "replace")
    except (OSError, ValueError, AttributeError) as exc:
        return {}, f"stdin read error: {exc}"
    if not raw.strip():
        return {}, None
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        return {}, f"invalid JSON hook payload: {exc}"
    if not isinstance(doc, dict):
        return {}, "hook payload JSON root must be an object"
    return doc, None


# ---------------------------------------------------------------------------
# Global WeaveProfiler PNG-only & Stop-gate guard (user-level ~/.qwen / ~/.claude)
# ---------------------------------------------------------------------------

def _global_state_path(payload: Dict[str, Any], root: Optional[str] = None) -> Path:
    sid = str(payload.get("session_id") or payload.get("sessionId") or os.environ.get("QWEN_SESSION_ID") or "default")
    safe_sid = re.sub(r"[^A-Za-z0-9._-]", "_", sid)[:80] or "default"
    return sessions_root(root) / GLOBAL_GUARDS_DIRNAME / f"{safe_sid}.json"


def _read_global_state(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(doc, dict):
            return doc
    except (OSError, ValueError):
        pass
    return {"active": False, "started_at": 0.0, "touched_dirs": [], "png_files": [], "stop_blocks": 0}


def _save_global_state(path: Path, state: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, state)
    except OSError:
        pass


def _handle_global_guard(event: str, payload: Dict[str, Any], args: argparse.Namespace, *, protocol: str) -> None:
    """User-level PNG-only + no-.gz + Stop-gate guard when no workspace session is provisioned.

    Transparent (zero interference) for non-WeaveProfiler sessions; activates automatically as
    soon as any tool call in the session touches WeaveProfiler profiling tools or artifacts.
    """
    if _allow_gz_env():
        return
    cwd = Path(payload.get("cwd") or args.workspace or os.getcwd()).resolve()
    state_path = _global_state_path(payload, getattr(args, "sessions_root", None))
    state = _read_global_state(state_path)
    tool_name, tool_input = _extract_tool(payload)
    all_text = "\n".join(_strings(tool_input))
    cmd_text = _shell_command(tool_input) or all_text
    targets = edit_targets(tool_name, tool_input, cwd)

    short, _own = _tool_kind(tool_name)
    if event == "PreToolUse":
        # Always block creating forbidden .perfetto*.gz / .trace.json.gz files!
        for path, _a, _op in targets:
            if _is_forbidden_gz_path(path):
                raise HookBlock(
                    "FORBIDDEN_GZ_ARTIFACT",
                    f"PNG-only delivery is enforced: refusing to write `{path.name}`. "
                    "Deliver `overview.png` instead (via `wp_decode.py ... -o overview.png`, "
                    "`weaveprof/viz.py`, or `wp.profile(..., out='overview.png')`).",
                )
        if short not in READ_ONLY_TOOLS and cmd_text and not _GZ_CLEANUP_ONLY_RE.match(cmd_text.strip()) and _FORBIDDEN_GZ_CMD_RE.search(cmd_text):
            raise HookBlock(
                "FORBIDDEN_GZ_ARTIFACT",
                "PNG-only delivery is enforced: do NOT generate `.perfetto.json.gz` / `.trace.json.gz` / "
                "`.perfetto-trace.gz` or invoke `export_perfetto_native.py` / `--write-gz`. "
                "Render `overview.png` directly via `wp_decode.py ... -o overview.png` or `weaveprof/viz.py`.",
            )
        if short not in READ_ONLY_TOOLS and _WP_PROFILING_ACTIVITY_RE.search(all_text) and not _HELP_ONLY_CMD_RE.match(cmd_text.strip()):
            if not state.get("active"):
                state["active"] = True
                state["started_at"] = round(time.time(), 3)
            touched = list(state.get("touched_dirs") or [])
            for d in _extract_candidate_dirs(all_text, cwd):
                ds = str(d)
                if ds not in touched:
                    touched.append(ds)
            state["touched_dirs"] = touched[-24:]
            _save_global_state(state_path, state)
        return

    if event in ("PostToolUse", "PostToolUseFailure"):
        if short in READ_ONLY_TOOLS:
            return
        triggered = bool(state.get("active")) or (
            bool(_WP_PROFILING_ACTIVITY_RE.search(all_text))
            and not _HELP_ONLY_CMD_RE.match(cmd_text.strip())
        )
        if not triggered:
            return
        if not state.get("active"):
            state["active"] = True
            state["started_at"] = round(time.time() - 5.0, 3)
        touched = list(state.get("touched_dirs") or [])
        for d in _extract_candidate_dirs(all_text, cwd):
            ds = str(d)
            if ds not in touched:
                touched.append(ds)
        state["touched_dirs"] = touched[-24:]
        since_ts = float(state.get("started_at") or 0.0) - 2.0
        bad_gz: List[Path] = []
        found_png: List[str] = list(state.get("png_files") or [])
        for ds in state["touched_dirs"]:
            gz_list, png_list = _scan_dir_artifacts(Path(ds), since_ts=since_ts, max_depth=3)
            bad_gz.extend(gz_list)
            for pp in png_list:
                ps = str(pp)
                if ps not in found_png:
                    found_png.append(ps)
        state["png_files"] = [p for p in found_png if _is_valid_png(p)][-32:]
        _save_global_state(state_path, state)
        if bad_gz:
            removed = _purge_forbidden_gz(bad_gz)
            raise HookBlock(
                "FORBIDDEN_GZ_ARTIFACT",
                f"PNG-only delivery policy forbids `.perfetto*.gz` / `.trace.json.gz` artifacts; "
                f"automatically deleted {removed[:4]}. Render `overview.png` directly via `wp_decode.py -o overview.png` "
                "or `weaveprof/viz.py` instead.",
            )
        return

    if event == "Stop":
        if protocol == "agy":
            term = str(payload.get("terminationReason") or "").strip().upper()
            if term not in _AGY_GATE_STOP_REASONS:
                return
        if not state.get("active"):
            return
        since_ts = float(state.get("started_at") or 0.0) - 2.0
        bad_gz: List[Path] = []
        valid_pngs: List[str] = [p for p in (state.get("png_files") or []) if _is_valid_png(p)]
        for ds in list(state.get("touched_dirs") or []):
            gz_list, png_list = _scan_dir_artifacts(Path(ds), since_ts=since_ts, max_depth=3)
            bad_gz.extend(gz_list)
            for pp in png_list:
                ps = str(pp)
                if ps not in valid_pngs:
                    valid_pngs.append(ps)
        blockers: List[str] = []
        if bad_gz:
            removed = _purge_forbidden_gz(bad_gz)
            blockers.append(
                f"FORBIDDEN_GZ_ARTIFACT: deleted disallowed `.perfetto*.gz` / `.trace.json.gz` files {removed[:4]}."
            )
        if not valid_pngs:
            blockers.append(
                "MISSING_OVERVIEW_PNG: you ran WeaveProfiler profiling in this session without producing a valid "
                "PNG deliverable (`overview.png`). Render the 4-panel PNG via `wp_decode.py ... -o overview.png`, "
                "`weaveprof/viz.py`, or `wp.profile(..., out='overview.png')` and inspect it before finishing."
            )
        if not blockers:
            state["stop_blocks"] = 0
            state["png_files"] = valid_pngs[-32:]
            _save_global_state(state_path, state)
            return
        stop_active = _is_stop_active(payload)
        base = int(state.get("stop_blocks") or 0) if stop_active else 0
        n = base + 1
        state["stop_blocks"] = 0 if n > STOP_BLOCK_LIMIT else n
        _save_global_state(state_path, state)
        if n > STOP_BLOCK_LIMIT:
            sys.stderr.write(f"{TAG} WARNING: global PNG Stop gate unmet after {n - 1} blocks; allowing stop.\n")
            return
        raise HookBlock("STOP_GATE", f"cannot finish yet (block {n}/{STOP_BLOCK_LIMIT}): " + " | ".join(blockers))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="WeaveProfiler (wp) Claude Code / Codex CLI / Qwen Code / Antigravity enforcement hook"
    )
    ap.add_argument(
        "--event",
        choices=("PreToolUse", "PostToolUse", "PostToolUseFailure", "PostInvocation", "Stop", "SubagentStop"),
    )
    ap.add_argument("--protocol", default="auto", choices=("auto", "claude", "codex", "qwen", "agy", "antigravity"))
    ap.add_argument("--workspace")
    ap.add_argument("--session-dir")
    ap.add_argument("--sessions-root")
    ap.add_argument("--global-guard", action="store_true",
                    help="User-level guard: delegates to workspace session when present, otherwise enforces PNG-only & Stop gate on WeaveProfiler sessions")
    args = ap.parse_args(list(argv) if argv is not None else None)
    payload, parse_err = _read_payload()
    event = args.event or str(payload.get("hookEventName") or payload.get("hook_event_name") or "")
    protocol = _detect_protocol(args.protocol, payload)
    try:
        if parse_err is not None:
            raise HookBlock("MALFORMED_HOOK_PAYLOAD", parse_err)
        ctx = Context(payload, args)
        if args.global_guard and ctx.session is None:
            _handle_global_guard(event, payload, args, protocol=protocol)
            return _emit_allow(protocol, event, None)
        extra: Optional[Dict[str, Any]] = None
        if event == "PreToolUse":
            handle_pre(ctx)
        elif event in ("PostToolUse", "PostToolUseFailure"):
            handle_post(ctx, protocol=protocol)
        elif event == "PostInvocation":
            extra = handle_post_invocation(ctx)
        elif event == "SubagentStop":
            handle_subagent_stop(ctx, protocol=protocol)
        elif event == "Stop":
            handle_stop(ctx, protocol=protocol)
        return _emit_allow(protocol, event, extra)
    except HookBlock as blk:
        return _emit_block(protocol, event, str(blk))
    except Exception as exc:  # fail closed with a visible reason
        return _emit_block(protocol, event, f"{TAG} BLOCKED (HOOK_ERROR): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
