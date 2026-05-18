"""Pre-recorded voice asset library.

Loads `assets/voice/map.zh-CN.json` (mapping symbolic event keys to .wav files)
plus the .wav files themselves, and serves them as raw bytes ready to push down
the audio_out WebSocket.

The JSON shape is unknown until the user uploads their map.zh-CN.json, so the
loader is forgiving: any value that's a string is treated as a filename; any
value that's a dict is searched for a 'file' / 'filename' / 'path' field.

When no asset matches a requested key, the caller is expected to fall back to
runtime TTS synthesis (edge-tts).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class VoiceAssets:
    def __init__(self, root: Path, map_path: Path | None = None):
        self.root = Path(root)
        self.map_path = map_path or (self.root / "map.zh-CN.json")
        self._key_to_file: dict[str, Path] = {}
        self._cache: dict[str, bytes] = {}
        if self.map_path.exists():
            self._load_map()
        else:
            log.warning("voice map not found: %s (will fall back to TTS for all)", self.map_path)

    def _load_map(self) -> None:
        try:
            data = json.loads(self.map_path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("failed to parse %s", self.map_path)
            return

        for k, v in _flatten(data):
            fname = _value_to_filename(v)
            if not fname:
                continue
            path = self.root / fname
            if not path.exists():
                # try with case-insensitive search; some maps mix .wav / .WAV
                alt = _ci_find(self.root, fname)
                if alt is not None:
                    path = alt
            if path.exists():
                self._key_to_file[k] = path
            else:
                log.debug("voice asset missing: key=%s file=%s", k, fname)

        log.info("voice assets loaded: %d entries", len(self._key_to_file))

    def has(self, key: str) -> bool:
        return key in self._key_to_file

    def get_bytes(self, key: str) -> bytes | None:
        if key in self._cache:
            return self._cache[key]
        path = self._key_to_file.get(key)
        if path is None:
            return None
        try:
            data = path.read_bytes()
        except Exception:
            log.exception("read failed: %s", path)
            return None
        self._cache[key] = data
        return data

    def known_keys(self) -> list[str]:
        return sorted(self._key_to_file.keys())


def _flatten(obj, prefix: str = "") -> list[tuple[str, object]]:
    """Walk a (possibly nested) dict, yielding (dotted_key, value) leaves."""
    out: list[tuple[str, object]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                out.extend(_flatten(v, key))
            else:
                out.append((key, v))
                # also expose the leaf key (without prefix) for convenience
                if prefix:
                    out.append((str(k), v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                out.extend(_flatten(v, key))
            else:
                out.append((key, v))
    return out


def _value_to_filename(v) -> str | None:
    if isinstance(v, str) and (v.lower().endswith(".wav") or v.lower().endswith(".mp3")):
        return v
    if isinstance(v, dict):
        for fld in ("file", "filename", "path", "wav"):
            if fld in v and isinstance(v[fld], str):
                return v[fld]
    return None


def _ci_find(root: Path, fname: str) -> Path | None:
    target = fname.lower()
    for p in root.iterdir():
        if p.name.lower() == target:
            return p
    return None
