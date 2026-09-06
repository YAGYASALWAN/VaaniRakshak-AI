"""Pinned original Parquet acquisition with hard payload bounds and no viewer audio."""
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request, build_opener

from vaanirakshak.data.source_probe import NoRedirects

SOURCES = {
    "indicvoices": ("ai4bharat/indicvoices_r", "5f4495c91d500742a58d1be2ab07d77f73c0acf8"),
    "indicsynth": ("vdivyasharma/IndicSynth", "c0a10386b723717aff682f757bd67f72983f269f"),
}
MAX_DOWNLOAD = 8 * 1024**3
MAX_SHARD = 600 * 1024**2


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Downloader:
    def __init__(self, token):
        if not token or not token.isascii() or any(c.isspace() for c in token):
            raise ValueError("Token must be nonempty ASCII without whitespace")
        self.token = token
        self.opener = build_opener(NoRedirects())
        self.bytes_read = 0
        self.metadata_bytes = 0

    def open(self, url):
        for _ in range(6):
            parsed = urlsplit(url)
            host = parsed.hostname or ""
            allowed = host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co")
            if parsed.scheme != "https" or not allowed or parsed.username or parsed.port:
                raise ValueError("Refusing unexpected download host")
            headers = {"Accept-Encoding": "identity", "User-Agent": "VaaniRakshak-baseline/0.1"}
            # Rebuild headers at every redirect. Never forward the token to CDN hosts.
            if host == "huggingface.co":
                headers["Authorization"] = "Bearer " + self.token
            try:
                return self.opener.open(Request(url, headers=headers), timeout=60)
            except HTTPError as exc:
                status = exc.code
                location = exc.headers.get("Location")
                exc.close()
                if status in (301, 302, 303, 307, 308) and location:
                    url = urljoin(url, location)
                    continue
                raise ValueError(f"Hugging Face request failed: HTTP {status}") from None
            except (URLError, OSError):
                raise ValueError("Download connection failed; rerun to reuse completed shards") from None
        raise ValueError("Too many download redirects")

    def listing(self, source, language):
        repo, revision = SOURCES[source]
        prefix = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{language}"
        url = prefix + "?limit=1000"
        files = []
        for _ in range(8):
            with self.open(url) as response:
                if response.headers.get_content_type() != "application/json":
                    raise ValueError("Expected JSON repository listing")
                payload = response.read(2 * 1024**2 + 1)
                self.metadata_bytes += len(payload)
                if len(payload) > 2 * 1024**2 or self.metadata_bytes > 16 * 1024**2:
                    raise ValueError("Repository listing exceeds metadata budget")
                entries = json.loads(payload)
                if not isinstance(entries, list):
                    raise ValueError("Invalid repository listing")
                for item in entries:
                    path = item.get("path", "")
                    if item.get("type") == "file" and re.fullmatch(language + r"/train-\d+-of-\d+\.parquet", path):
                        size = item.get("size")
                        if isinstance(size, int) and 0 < size <= MAX_SHARD:
                            digest = (item.get("lfs") or {}).get("oid")
                            if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
                                raise ValueError("Invalid source checksum")
                            files.append({"source": source, "language": language, "repository": repo,
                                          "revision": revision, "path": path, "size": size, "sha256": digest})
                link = response.headers.get("Link", "")
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            if not match:
                return sorted(files, key=lambda x: x["path"])
            url = match.group(1)
            if not url.startswith(prefix + "?"):
                raise ValueError("Unexpected listing pagination")
        raise ValueError("Repository listing pagination limit reached")

    def download(self, item, folder):
        name = hashlib.sha256((item["repository"] + item["revision"] + item["path"]).encode()).hexdigest()
        dest = Path(folder) / (name + ".parquet")
        receipt = dest.with_suffix(".json")
        expected = item["size"]
        if dest.exists() and receipt.exists():
            saved = json.loads(receipt.read_text())
            actual = file_sha(dest)
            if dest.stat().st_size == expected and actual == saved.get("sha256") and (
                    not item["sha256"] or actual == item["sha256"]):
                print("Using completed shard:", item["source"], item["path"], flush=True)
                return dest
            raise ValueError("Cached shard checksum mismatch; remove the affected cache file")
        if expected > MAX_SHARD or self.bytes_read + expected > MAX_DOWNLOAD:
            raise ValueError("8 GiB download budget would be exceeded")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(dest.parent).free < expected + 2 * 1024**3:
            raise ValueError("Insufficient free disk space; leave at least 2 GiB beyond the shard")
        url = f'https://huggingface.co/datasets/{item["repository"]}/resolve/{item["revision"]}/{quote(item["path"], safe="/")}'
        partial = dest.with_suffix(".partial")
        print("Downloading:", item["source"], item["path"], f'({expected / 1024**2:.0f} MiB)', flush=True)
        try:
            with self.open(url) as response, partial.open("wb") as output:
                if response.status != 200:
                    raise ValueError("Expected complete original shard response")
                length = response.headers.get("Content-Length")
                if length is None or int(length) != expected:
                    raise ValueError("Shard size differs from pinned repository listing")
                received = 0
                while received < expected:
                    chunk = response.read(min(1024**2, expected - received))
                    if not chunk:
                        raise ValueError("Interrupted shard download; rerun to retry this shard")
                    received += len(chunk)
                    self.bytes_read += len(chunk)
                    output.write(chunk)
            digest = file_sha(partial)
            if item["sha256"] and digest != item["sha256"]:
                raise ValueError("Downloaded shard checksum mismatch")
            partial.replace(dest)
            save_json(receipt, {"sha256": digest, "source": item})
            return dest
        finally:
            partial.unlink(missing_ok=True)


def make_plan(client, seed=42):
    """Spread four randomly selected shards per corpus/language, at most 16 total."""
    selected = []
    for source in SOURCES:
        for language in ("Hindi", "Punjabi"):
            files = client.listing(source, language)
            if len(files) < 4:
                raise ValueError(f"Need four bounded original train shards for {source}/{language}")
            rng = random.Random(f"{seed}:{source}:{language}")
            selected.extend(rng.sample(files, 4))
    if sum(item["size"] for item in selected) > MAX_DOWNLOAD:
        raise ValueError("Selected original shards exceed 8 GiB; no audio downloaded")
    return {"schema": "baseline-download-v1", "seed": seed, "files": selected,
            "planned_bytes": sum(item["size"] for item in selected), "max_bytes": MAX_DOWNLOAD}
