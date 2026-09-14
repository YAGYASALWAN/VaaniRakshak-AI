"""Authenticated, byte-budgeted range reads of pinned SEA-Spoof Parquet files."""
from collections import OrderedDict
import hashlib
import io
import json
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from vaanirakshak.baseline_download import save_json

MAX_BYTES = 30_000_000_000
MAX_SCOPE_CACHE_BYTES = 4_000_000_000
REPOSITORY = "Jack-ppkdczgx/SEA-Spoof"
REVISION = "132f5dca9b6efe39cf1d3b54a858f167f5a421fc"


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not (host == "huggingface.co" or host.endswith(".hf.co") or host.endswith(".huggingface.co")):
            raise ValueError("Unexpected dataset redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if host != "huggingface.co":
            redirected.remove_header("Authorization")
        return redirected


class Transfer:
    def __init__(self, root, token):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "transfer.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {"reserved_bytes": 0}
        if not token or not token.isascii() or any(c.isspace() for c in token):
            raise ValueError("Sign in to your approved Hugging Face account using hf auth login")
        self.token = token
        self.opener = build_opener(SafeRedirect())
        self.cache = OrderedDict()
        self.range_cache_root = self.root / "range_cache"
        self._scope_id: str | None = None
        self._scope_dir: Path | None = None
        self._scope_cached_bytes = 0

    @property
    def used(self):
        return self.ledger["reserved_bytes"]

    @staticmethod
    def _scope_key(scope_id: str) -> str:
        return hashlib.blake2s(scope_id.encode("utf-8"), digest_size=12).hexdigest()

    def _scope_directory(self, scope_id: str) -> Path:
        return self.range_cache_root / self._scope_key(scope_id)

    def begin_scope(self, scope_id: str) -> None:
        """Enable persistent retry caching for one logical preparation unit.

        A scope is intended to wrap one SEA row group. Successfully fetched remote
        ranges are cached atomically on disk. If the process crashes, the next run
        can reopen the same scope and reuse those ranges without reserving/downloading
        them again. A successful caller should end the scope with ``clear=True``.
        """
        value = str(scope_id).strip()
        if not value:
            raise ValueError("scope_id must be nonempty")
        if self._scope_id is not None:
            raise RuntimeError("A transfer retry-cache scope is already active")
        directory = self._scope_directory(value)
        directory.mkdir(parents=True, exist_ok=True)
        identity_path = directory / "scope.json"
        identity = {
            "scope_id": value,
            "repository": REPOSITORY,
            "revision": REVISION,
        }
        if identity_path.exists():
            saved = json.loads(identity_path.read_text(encoding="utf-8"))
            if saved != identity:
                raise ValueError("Persistent range-cache scope identity changed")
        else:
            save_json(identity_path, identity)

        # A process may die after writing a temporary cache file but before its
        # atomic rename. Those files are never valid cache entries and must not
        # consume the row-group cache budget on the next run.
        for temporary in directory.glob("*.tmp"):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

        self._scope_id = value
        self._scope_dir = directory
        self._scope_cached_bytes = sum(
            path.stat().st_size for path in directory.glob("*.bin") if path.is_file()
        )

    def end_scope(self, scope_id: str, *, clear: bool) -> None:
        value = str(scope_id).strip()
        if self._scope_id != value or self._scope_dir is None:
            raise RuntimeError("Attempted to end a transfer scope that is not active")
        directory = self._scope_dir
        self._scope_id = None
        self._scope_dir = None
        self._scope_cached_bytes = 0
        if clear and directory.exists():
            shutil.rmtree(directory)
            try:
                self.range_cache_root.rmdir()
            except OSError:
                pass

    def clear_scope(self, scope_id: str) -> None:
        value = str(scope_id).strip()
        if not value:
            raise ValueError("scope_id must be nonempty")
        if self._scope_id == value:
            raise RuntimeError("Cannot clear the active transfer scope")
        directory = self._scope_directory(value)
        if directory.exists():
            shutil.rmtree(directory)
        try:
            self.range_cache_root.rmdir()
        except OSError:
            pass

    def _range_cache_paths(self, item, offset: int, size: int) -> tuple[Path, Path] | None:
        if self._scope_dir is None:
            return None
        material = json.dumps(
            {
                "repository": REPOSITORY,
                "revision": REVISION,
                "path": item["path"],
                "file_size": item["size"],
                "offset": offset,
                "size": size,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.blake2s(material, digest_size=16).hexdigest()
        return self._scope_dir / f"{key}.bin", self._scope_dir / f"{key}.json"

    def _drop_range_pair(self, data_path: Path, meta_path: Path) -> None:
        for path in (data_path, meta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._scope_cached_bytes = sum(
            path.stat().st_size for path in self._scope_dir.glob("*.bin") if path.is_file()
        ) if self._scope_dir is not None else 0

    def _load_persistent_range(self, item, offset: int, size: int) -> bytes | None:
        paths = self._range_cache_paths(item, offset, size)
        if paths is None:
            return None
        data_path, meta_path = paths
        if not data_path.is_file() and not meta_path.is_file():
            return None
        if not data_path.is_file() or not meta_path.is_file():
            # An incomplete pair may be left if the process died between the two
            # atomic file commits. It is not trusted and should not consume cache
            # capacity on the next attempt.
            self._drop_range_pair(data_path, meta_path)
            return None
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = {
                "repository": REPOSITORY,
                "revision": REVISION,
                "path": item["path"],
                "file_size": item["size"],
                "offset": offset,
                "size": size,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError("range cache identity mismatch")
            if data_path.stat().st_size != size:
                raise ValueError("range cache length mismatch")
            data = data_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if metadata.get("sha256") != digest:
                raise ValueError("range cache digest mismatch")
            return data
        except (OSError, ValueError, json.JSONDecodeError):
            self._drop_range_pair(data_path, meta_path)
            return None

    def _save_persistent_range(self, item, offset: int, size: int, data: bytes) -> None:
        paths = self._range_cache_paths(item, offset, size)
        if paths is None or size <= 0:
            return
        if size > MAX_SCOPE_CACHE_BYTES or self._scope_cached_bytes + size > MAX_SCOPE_CACHE_BYTES:
            return
        data_path, meta_path = paths
        temporary = data_path.with_suffix(".bin.tmp")
        temporary.write_bytes(data)
        temporary.replace(data_path)
        metadata = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "path": item["path"],
            "file_size": item["size"],
            "offset": offset,
            "size": size,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        save_json(meta_path, metadata)
        self._scope_cached_bytes += size

    def fetch(self, item, offset, size):
        if size == 0:
            return b""
        if not 0 <= offset < item["size"] or offset + size > item["size"]:
            raise ValueError("Invalid Parquet byte range")
        key = (item["path"], offset, size)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        persistent = self._load_persistent_range(item, offset, size)
        if persistent is not None:
            if size <= 1024**2:
                self.cache[key] = persistent
                while len(self.cache) > 16:
                    self.cache.popitem(last=False)
            return persistent

        if self.used + size > MAX_BYTES:
            raise ValueError("30 GB transfer ceiling reached. Completed features are retained; no more data was requested.")
        # Reserve before the request, including failed attempts, to avoid a crash bypassing the ceiling.
        self.ledger["reserved_bytes"] += size
        save_json(self.ledger_path, self.ledger)
        headers = {"Authorization": "Bearer " + self.token, "Accept-Encoding": "identity",
                   "Range": f"bytes={offset}-{offset + size - 1}", "User-Agent": "VaaniRakshak-SEA30/1"}
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{item['path']}"
        try:
            with self.opener.open(Request(url, headers=headers), timeout=90) as response:
                expected = f"bytes {offset}-{offset+size-1}/{item['size']}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Source did not honor byte-range loading; refusing a full-file fallback")
                pieces, received, last = [], 0, time.monotonic()
                while received < size:
                    block = response.read(min(8 * 1024**2, size - received))
                    if not block:
                        raise ConnectionError("Interrupted range read; rerun to reuse completed cached ranges")
                    pieces.append(block)
                    received += len(block)
                    if time.monotonic() - last >= 5:
                        print(f"  Reading audio/metadata range: {received/size:.1%}; reserved {self.used/1e9:.2f}/30 GB", flush=True)
                        last = time.monotonic()
                data = b"".join(pieces)
        except HTTPError as error:
            code = error.code
            error.close()
            raise ValueError(f"Hugging Face returned HTTP {code}. Check access for the signed-in account; no token is printed.") from None

        self._save_persistent_range(item, offset, size, data)
        # Only small metadata reads are also cached in memory. Large scoped reads are
        # persisted on disk only until the caller commits that preparation scope.
        if size <= 1024**2:
            self.cache[key] = data
            while len(self.cache) > 16:
                self.cache.popitem(last=False)
        return data


class RangeFile(io.RawIOBase):
    def __init__(self, client, item):
        super().__init__()
        self.client, self.item, self.position = client, item, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.item["size"] + offset if whence == 2 else -1
        if position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = max(0, self.item["size"] - self.position) if size < 0 else min(size, max(0, self.item["size"] - self.position))
        data = self.client.fetch(self.item, self.position, size)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
