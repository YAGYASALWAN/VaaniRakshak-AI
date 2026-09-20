# VaaniRakshak V2 SEA data-status doctor

Use `scripts/v2_data_status.py` to inspect a V2 SEA preparation directory without contacting Hugging Face or mutating the run.

## Basic status

```powershell
python scripts/v2_data_status.py `
  --data-root data/v2_sea_en
```

The report includes:

- current preparation state;
- whether automatic resume appears safe;
- free disk space;
- cumulative `transfer.json` reservation and remaining 30 GB ceiling;
- selected plan summary;
- number of cached source metadata files;
- committed row-group receipts versus planned row groups;
- records represented by committed receipts;
- persistent retry-cache scopes/files/bytes;
- manifest fingerprint and leakage-audit result when a manifest exists;
- the recommended next action.

The command is offline. `network_used` is always reported as `false`.

## States

### `not_started`

No plan, manifest, metadata cache or transfer ledger was found. Run the V2 preflight before starting preparation.

### `planning`

Metadata/transfer state exists but no final selected-group plan is available yet. Rerun the same preparation command with the same output directory.

### `materializing`

A plan exists and at least one planned row group still lacks a durable receipt. Rerun the same preparation command. Persistent completed range-cache entries can be reused.

### `finalizing`

All planned groups have receipts but there is not yet an audit-clean authoritative manifest. Rerun preparation so it can rebuild/audit/finalize the manifest.

### `complete`

A manifest passes the V2 leakage audit and, when a preparation plan exists, every planned group has a committed receipt. When both receipt and manifest counts are available they must agree.

### `blocked`

An integrity/safety problem was detected, such as malformed transfer ledger state, invalid receipt JSON, failed manifest audit, or a completed-plan/manifest receipt-count mismatch. Do not blindly resume; investigate the reported `issues` first.

## Strict mode

For scripts/automation:

```powershell
python scripts/v2_data_status.py `
  --data-root data/v2_sea_en `
  --strict
```

`--strict` exits with code `2` if the doctor marks the directory unsafe to resume.

## Save a report

```powershell
python scripts/v2_data_status.py `
  --data-root data/v2_sea_en `
  --output reports/v2_data_status.json
```

This is useful before and after a long preparation session.

## Optional audio filesystem scan

By default the doctor does **not** recursively count every FLAC file because a large corpus can make that slow.

Use:

```powershell
python scripts/v2_data_status.py `
  --data-root data/v2_sea_en `
  --scan-audio
```

This reports FLAC count/bytes and warns when fewer FLAC files are visible than committed receipt records. It is a diagnostic check, not a replacement for receipt/manifest integrity.

## Recovery workflow

After an ordinary interruption:

```powershell
python scripts/v2_data_status.py --data-root data/v2_sea_en --strict
```

If the state is resumable, use the **same** preparation directory:

```powershell
python -m vaanirakshak.v2_prepare_sea `
  --output data/v2_sea_en `
  --budget-gb 26
```

Do not delete `transfer.json`, `v2_sea_plan.json`, metadata sidecars, receipts or `range_cache` in an attempt to force a restart. Those artifacts are part of the fail-closed recovery contract.

For the detailed retry-cache semantics, see `docs/V2_SEA_PREPARATION_RECOVERY.md`.
