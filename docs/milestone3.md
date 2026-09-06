# Milestone 3 — controlled real-data pilot

Milestone 2's full 50-test suite passed on the user's Windows environment
(screenshot: 50 passed in 5.08s). Earlier Milestone 2 execution-environment
limitations remain a record of that session, not the latest project test status.

Objective 1 starts with real-source schema discovery. The new probe accesses
public JSON metadata only: two repositories, their advertised splits and at most
three rows in each of six candidate train subsets. It makes at most 16 requests,
with a 2 MiB response bound, a 32 MiB cumulative payload bound and 20-second
per-request timeout. It never requests ASVspoof, follows audio links, downloads
Parquet shards, decodes audio, or invokes load_dataset.

Run from the project root with the existing environment active:

```bash
python scripts/probe_dataset_sources.py
```

If a source returns 401/403, use a Hugging Face read token from the same account
that has access to the dataset. Create it at https://huggingface.co/settings/tokens
and run:

```bash
python scripts/probe_dataset_sources.py --ask-token
```

Paste the token at the hidden terminal prompt and press Enter. Nothing is shown
while pasting. The token is kept in memory and added only to HTTPS metadata
requests to the two approved hosts; redirects remain disabled. It is not written
to project files or reports. Never paste the token into chat or a command argument.
The token does not grant dataset access beyond the account's existing permissions.

Upload the printed `probe_summary.json` path to the next chat turn. It contains
field names, missing-value counts, subset availability and errors; it does not
include transcripts, speaker values or signed audio URLs. Full JSON responses
remain in the same ignored `data/metadata/source_probe_.../` directory.
Exit code 2 means at least one request/subset needs inspection; the summary
is still saved when individual requests fail.

The probe discovers config spelling from advertised subsets. A missing match is
reported; it never substitutes an unrelated subset or claims a language absent
from the entire corpus merely because no matching train config is advertised.
The first three rows are for schema inspection only, not diversity estimates,
sampling or claims about corpus balance.

## Concepts

**Schema discovery:** observe field names and types before assuming a mapping.
**Data provenance:** retain source repository, config, split, row indices and
snapshot checksums so records can be traced.
**Source revision:** the observed Hub SHA identifies repository state, but the
viewer may serve an older cache; this probe does not prove revision pinning.
**Stable locator:** an enduring reference to original audio. Viewer audio URLs
can expire and may reference transcoded previews. They are not approved original
training-audio locations, and rates/codecs inferred from them must not be called
raw source properties.

## Remaining sequence

1. Run the probe, inspect errors and actual fields; verify original metadata and
   audio access paths. Check restoration/enhancement history of genuine audio.
2. Obtain bounded metadata exports suitable for diversity-aware selection. Use
   the existing adapters and audit/planner; resolve unknown original durations
   rather than borrowing duration from reference recordings.
3. Produce a reviewable pilot manifest, file count, expected transfer size and
   hard download limits. Request approval for the concrete audio acquisition.
4. Acquire approved originals with source revisions/checksums and validate local
   audio. Preserve immutable originals and separately record rejected files.
5. Construct splits using connected source/target speakers and reference
   recordings, with generator holdouts validated jointly. Do not merely run
   existing speaker and generator split functions in sequence.
6. Reuse Milestone 1 preprocessing; inherit split assignments for every window.
   Preserve raw versus processed measurements and audit dropped/padded clips.
7. Verify manifests, duplicates, leakage, class/language time balance and the
   full test suite. Only then consider Milestone 4 training.

Live discovery could not be verified from the agent's restricted network in this
session. Offline tests verify parsing, request bounds, no audio URL requests and
failure reporting; they do not certify live service availability.

References: [rows API](https://huggingface.co/docs/dataset-viewer/rows),
[splits API](https://huggingface.co/docs/dataset-viewer/splits),
[source observations](dataset_sources.md).
