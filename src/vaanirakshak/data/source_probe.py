"""Bounded public JSON schema discovery. Never downloads linked audio or shards."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SOURCES = {"indicvoices": "ai4bharat/indicvoices_r",
           "indicsynth": "vdivyasharma/IndicSynth"}
LANGUAGES = ("Hindi", "Punjabi", "Tamil", "Telugu", "Bengali", "Marathi")
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024


class NoRedirects(HTTPRedirectHandler):
    """Fail on redirects rather than following an unexpected download endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class MetadataClient:
    """GET client restricted to the two metadata API hosts, with optional auth.

    Every response must be JSON, at most 2 MiB; cumulative response payload budget
    is 32 MiB. Tokens stay in memory. No retries or binary decoding are used.
    """

    def __init__(self, token: str | None = None) -> None:
        if token is not None and (not token or not token.isascii()
                                  or any(c.isspace() for c in token)):
            raise ValueError("Token must be nonempty ASCII without whitespace")
        self._token = token
        self.bytes_read = 0
        self.opener = build_opener(NoRedirects())

    def get(self, url: str) -> dict[str, Any]:
        """Read bounded JSON, reject non-object responses and unexpected paths."""
        parsed = urlsplit(url)
        allowed = (
            parsed.hostname == "datasets-server.huggingface.co"
            and parsed.path in {"/splits", "/rows"}
        ) or (
            parsed.hostname == "huggingface.co"
            and parsed.path in {f"/api/datasets/{repo}" for repo in SOURCES.values()}
        )
        if parsed.scheme != "https" or not allowed or parsed.username or parsed.port:
            raise ValueError("Only approved public JSON metadata endpoints are allowed")
        budget = min(MAX_RESPONSE_BYTES, MAX_TOTAL_BYTES - self.bytes_read)
        if budget <= 0:
            raise ValueError("Metadata session payload budget exhausted")
        request = Request(url, headers={"Accept": "application/json",
                                      "User-Agent": "VaaniRakshak-schema-probe/0.1"})
        if self._token:
            request.add_header("Authorization", "Bearer " + self._token)
        with self.opener.open(request, timeout=20) as response:
            if response.headers.get_content_type() != "application/json":
                raise ValueError("Refusing non-JSON response")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > budget:
                raise ValueError("Metadata response exceeds byte budget")
            payload = response.read(budget + 1)
            self.bytes_read += len(payload)
        if len(payload) > budget:
            raise ValueError("Metadata response exceeds byte budget")
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        return result


def select_subsets(payload: dict[str, Any], repository: str) -> dict[str, str | None]:
    """Resolve requested names against actual advertised train subsets, case-insensitively."""
    splits = payload.get("splits")
    if not isinstance(splits, list):
        raise ValueError("Split response lacks a splits array")
    available = [
        item["config"] for item in splits
        if item.get("dataset") == repository and item.get("split") == "train"
        and isinstance(item.get("config"), str)
    ]
    matches = {}
    for language in LANGUAGES:
        found = sorted(set(x for x in available if x.casefold() == language.casefold()))
        if len(found) > 1:
            raise ValueError(f"Ambiguous source config for {language}")
        matches[language] = found[0] if found else None
    return matches


def inspect_rows(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    """Summarize a structural preview. Never infer corpus balance from first rows.

    Audio values may be signed/transcoded viewer assets. We record only whether
    audio metadata exists; source audio properties and stable locators need later
    verification. Original response is stored locally as provenance.
    """
    rows = payload.get("rows")
    features = payload.get("features")
    if not isinstance(rows, list) or not isinstance(features, list) or len(rows) > limit:
        raise ValueError("Malformed or oversized rows response")
    names = [f["name"] for f in features if isinstance(f, dict) and isinstance(f.get("name"), str)]
    if not rows or not names or len(names) != len(set(names)):
        raise ValueError("No usable preview rows/features, or duplicate field names")
    nonnull = dict.fromkeys(names, 0)
    for item in rows:
        if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
            raise ValueError("Malformed row envelope")
        if not isinstance(item.get("row_idx"), int) or item["row_idx"] < 0:
            raise ValueError("Row lacks valid source index")
        if item.get("truncated_cells"):
            raise ValueError("Truncated metadata cells cannot be used for schema verification")
        for name in names:
            value = item["row"].get(name)
            if value is not None and value != "":
                nonnull[name] += 1
    return {
        "rows_observed": len(rows), "feature_names": names,
        "feature_types": {f["name"]: f.get("type") for f in features if isinstance(f, dict) and f.get("name") in names},
        "nonnull_in_preview": nonnull,
        "num_rows_total_reported": payload.get("num_rows_total"),
        "partial": payload.get("partial"),
        "purpose": "schema inspection only; first rows are not a balanced sample",
        "stable_audio_locator_verified": False,
    }


def run_probe(output: Path, client: MetadataClient | None = None,
              rows_per_subset: int = 3) -> dict[str, Any]:
    """Inspect at most 36 preview rows across two sources and six languages.

    Inputs: a NEW output directory and optional injectable client for offline tests.
    Outputs: raw JSON responses plus a compact probe_summary.json. Errors are
    retained per request; unavailable fields/subsets are not fabricated. No audio
    downloads, adaptation into training candidates, or train/test splitting.
    """
    if not 1 <= rows_per_subset <= 3:
        raise ValueError("This schema probe permits only 1–3 rows per subset")
    output.mkdir(parents=True, exist_ok=False)
    http = client or MetadataClient()
    report: dict[str, Any] = {
        "stage": "milestone_3_objective_1", "audio_downloaded": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_per_subset": rows_per_subset, "sources": {},
        "limits": {"response_bytes": MAX_RESPONSE_BYTES, "total_payload_bytes": MAX_TOTAL_BYTES},
        "revision_warning": "Hub SHA is observed, not proof that viewer cache uses that revision.",
        "next_gate": "Review metadata availability and provenance before choosing any audio subset.",
    }

    def fetch(url: str, filename: str) -> dict[str, Any]:
        value = http.get(url)
        content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        (output / filename).write_text(content, encoding="utf-8")
        return value

    for alias, repository in SOURCES.items():
        source: dict[str, Any] = {"repository": repository, "subsets": {}, "errors": []}
        report["sources"][alias] = source
        try:
            info = fetch(f"https://huggingface.co/api/datasets/{repository}", f"{alias}_hub.json")
            source["hub_sha_observed"] = info.get("sha")
            source["gated"] = info.get("gated")
        except (OSError, ValueError) as exc:
            source["errors"].append({"stage": "hub_info", "error": str(exc)})
        try:
            url = "https://datasets-server.huggingface.co/splits?" + urlencode({"dataset": repository})
            payload = fetch(url, f"{alias}_splits.json")
            configs = select_subsets(payload, repository)
        except (OSError, ValueError) as exc:
            source["errors"].append({"stage": "splits", "error": str(exc)})
            continue
        for language, subset in configs.items():
            if subset is None:
                source["subsets"][language] = {"status": "no_matching_train_config"}
                continue
            try:
                url = "https://datasets-server.huggingface.co/rows?" + urlencode({
                    "dataset": repository, "config": subset, "split": "train",
                    "offset": 0, "length": rows_per_subset})
                filename = f"{alias}_{language}_rows.json"
                payload = fetch(url, filename)
                item = inspect_rows(payload, rows_per_subset)
                item.update({"status": "inspected", "config": subset, "split": "train",
                             "offset": 0, "raw_response": filename,
                             "snapshot_sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest()})
                source["subsets"][language] = item
            except (OSError, ValueError) as exc:
                source["subsets"][language] = {"status": "error", "error": str(exc)}
    report["payload_bytes_read"] = http.bytes_read
    report["status"] = "review_required"
    (output / "probe_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
