"""Authenticated, byte-budgeted range reads of pinned SEA-Spoof Parquet files."""
from collections import OrderedDict
import io
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from vaanirakshak.baseline_download import save_json

MAX_BYTES = 30_000_000_000
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

    @property
    def used(self):
        return self.ledger["reserved_bytes"]

    def fetch(self, item, offset, size):
        if size == 0:
            return b""
        if not 0 <= offset < item["size"] or offset + size > item["size"]:
            raise ValueError("Invalid Parquet byte range")
        key = (item["path"], offset, size)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
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
                        raise ConnectionError("Interrupted range read; rerun to reuse completed feature groups")
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
        # Only small metadata reads are cached; audio is converted to features instead.
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
