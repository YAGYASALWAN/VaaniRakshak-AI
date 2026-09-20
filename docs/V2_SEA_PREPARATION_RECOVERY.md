# VaaniRakshak V2 SEA preparation recovery

The V2 SEA-Spoof preparation path is designed to fail closed under interruption while avoiding unnecessary repeated downloads of remote Parquet ranges that were already completed successfully.

## First recovery command

Before blindly restarting an interrupted run, inspect it offline:

```powershell
python scripts/v2_data_status.py `
  --data-root data/v2_sea_en `
  --strict
```

The doctor uses no network and reports transfer reservation, plan/receipt progress, retry-cache usage, disk headroom and manifest/audit state. A blocked status should be investigated before resuming.

`Transfer` also validates `transfer.json` itself. Invalid JSON, a non-integer reservation, a negative value or a reservation beyond the 30 GB ceiling is rejected before network transfer, even if the doctor command was skipped.

## Durable boundaries

Two preparation units have durable commits:

1. **SEA source metadata scan** — committed by `metadata/<source-id>.json`.
2. **Selected SEA row group** — committed by `receipts/<row-group-id>.json`.

Only a committed sidecar/receipt is authoritative. Local audio files created before a row-group receipt exists are not treated as dataset membership on the next run.

## Persistent retry-range cache

During either unit, `Transfer.begin_scope(...)` enables a bounded persistent cache under:

```text
<output>/range_cache/
```

The cache is keyed by the pinned dataset repository/revision plus source path, source file size, byte offset and byte count. Each completed cached range stores a SHA-256 digest and is verified before reuse.

The per-scope cache is capped at 4 GB. It is a retry cache, not a second dataset archive.

## Successful unit

```text
begin retry scope
    ↓
fetch pinned remote ranges
    ↓
perform metadata scan / materialize canonical audio
    ↓
atomically write durable metadata sidecar or row-group receipt
    ↓
end scope with clear=True
```

Once the durable commit succeeds, cached remote ranges are deleted. If a process dies after the durable commit but before cache deletion, the next run validates the sidecar/receipt and then removes the stale cache without another remote read.

## Interrupted unit

```text
begin retry scope
    ↓
fetch one or more ranges
    ↓
exception / interruption before durable receipt
    ↓
end scope with clear=False
```

Completed range-cache entries remain on disk. A fresh `Transfer` process reopening the same scope verifies and reuses those ranges without incrementing `transfer.json` for those exact completed ranges.

The row group or metadata unit itself is rerun from its beginning. This is intentional: only the final receipt is authoritative.

## Disk safety

Preparation maintains a 2 GB runtime free-space floor. The check is repeated during metadata scanning and row-group batches, not only at startup.

If free space falls below that floor, preparation aborts through the same fail-safe path. The active retry scope is retained and the run can be resumed after freeing disk space.

## Corruption handling

A persistent range is trusted only when:

- data and metadata files both exist;
- repository/revision/path/file-size/offset/size identity matches;
- cached byte length matches;
- SHA-256 matches.

Corrupt or incomplete pairs are deleted and fetched again. Refetches remain charged by the fail-closed transfer ledger. Orphaned `*.tmp` cache files from a process crash are removed when a scope reopens.

## Transfer-budget semantics

`transfer.json` remains the hard cumulative reservation ledger with a 30 GB ceiling. A completed cached range that is reused after restart is **not reserved twice**.

The remaining limitation is narrower: if an HTTP range request itself is interrupted before the full requested range is received and committed to the retry cache, a later retry may consume additional fail-closed reservation budget. The system does not pretend partially received bytes are a complete cache entry.

Do not delete `transfer.json` to bypass the ceiling.

## Local audio after a crash

Canonical FLAC writes use a temporary file followed by an atomic replace. A completed FLAC may therefore exist even when the surrounding row-group receipt was never committed.

That FLAC is **not trusted as manifest state**. On retry the row group is processed again and the file is deterministically overwritten/revalidated before a new receipt is committed.

## Tests

CI covers:

- completed persistent range reused by a fresh `Transfer` without a second network call or budget increment;
- malformed transfer ledgers rejected fail-closed;
- corrupted cached range rejected and refetched;
- incomplete data/metadata cache pair rejected;
- orphan temporary cache file cleanup;
- nested scopes refused;
- failed metadata scan retains its retry scope;
- failed row-group materialization retains its retry scope;
- runtime disk-floor enforcement;
- a tiny local Parquet corpus exercises metadata scan → canonical FLAC → receipt → manifest → rerun from receipt without gated network data;
- the offline data-status doctor detects incomplete plans, stale manifests and manifest/receipt count mismatches.

## Operational rule

After an ordinary interruption:

```powershell
python scripts/v2_data_status.py --data-root data/v2_sea_en --strict
```

If the directory is safe to resume, rerun the same preparation command with the same output directory:

```powershell
python -m vaanirakshak.v2_prepare_sea `
  --output data/v2_sea_en `
  --budget-gb 26
```

Do not delete receipts, metadata sidecars, `range_cache`, `v2_sea_plan.json`, or `transfer.json` unless deliberately abandoning that preparation directory and starting a separate experiment.
