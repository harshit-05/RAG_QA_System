# S3-10: Backup and restore

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | SRS §11 (backup and DR: a scheduled snapshot of the store and its manifest, a documented restore runbook, stated RPO and RTO); DEC-25 |
| **Depends on** | S3-9 |
| **Model** | opus |
| **Plan-first** | yes |
| **Branch** | `feat/s3-10-backup-restore`. Risky: a restore switches the live index. |

## Goal

The index can be backed up on a schedule and restored in minutes, and both are proven
against a real Qdrant server: local mode has no snapshots (the maintainer's addition to
DEC-3). A published generation is never written again, so its snapshot is consistent
without pausing anything. A restore is one more generation and one more alias switch, so
DEC-19's guarantees hold through it: all or nothing, and the previous generation kept. The
README states what is lost and how long recovery takes. Design: ARCHITECTURE.md §3.1, DEC-25.

## Scope

- **`rag_qa/backup.py`, the entry point `rag-backup`.**
  - **`create`**, under the writer lock, so recovery cannot delete the collection while it
    is snapshotted:
    1. `create_snapshot` of the alias's target;
    2. download it to `paths.backups/<collection>/` (default `backups/`, overridden by
       `RAG_BACKUP_PATH`, gitignored), beside a sidecar JSON: the generation, the manifest's
       summary (identities, documents, chunks), Qdrant's sha256, the server's version and
       the time;
    3. verify the download against the checksum;
    4. delete the server's copy;
    5. keep the newest `--keep` (7).
  - **`list`:** what is on disk, newest first, with the sidecar's summary.
  - **`restore FILE`:**
    1. verify the checksum;
    2. refuse a snapshot whose embedder identity differs from the config's, as `rag-query`
       would;
    3. upload it as a new generation name, through Qdrant's `snapshots/upload` endpoint.
       `qdrant-client` exposes it only through its generated HTTP API (checked in the
       architecture pass), so the plan settles the call;
    4. correct `generation` in the restored metadata;
    5. under the writer lock, switch the alias, then recover.

    It prints what it replaced, and says how to switch back (the previous generation is
    kept). `--dry-run` stops after step 2.
  - **`create --every 24h`** loops, for the compose service.
  - Exit codes as `rag-ingest`'s: 0 done, 1 failed after starting, 2 could not start.
- **The vectorstore seam:** `snapshot(store, generation, dest)` and
  `restore(store, path, *, expect_embedder)`. `backup.py` never imports `qdrant_client`
  (the architecture test).
- **Schedule:** a compose `backup` profile, with the app's image running
  `rag-backup create --every 24h --keep 7` and mounting `./backups`. README gives a cron line
  and a systemd timer for a host without compose.
- **Tests.**
  - **Hermetic:** the sidecar, the checksum verification, the refusals, retention and the
    lock, with the snapshot calls faked.
  - **`qdrant_server`, in CI's `eval-retrieval` job against its service container:** on a
    small collection, create, download, checksum, restore, switch, then a query that
    returns the same points.
- **README: "Backup and restore",** a runbook.
  - **The commands,** and checking a backup's sidecar.
  - **Restoring, then switching back** if needed.
  - **Copying `backups/` off the host** (an rclone or `aws s3 sync` line): the operator's
    step, an SRS difference.
  - **RPO:** the backup interval for the index. The corpus and git are the source of truth,
    and their backup is the operator's.
  - **RTO:** about a minute from a snapshot; a re-ingest from the corpus, from NFR-3's
    table.

## Out of scope

- Uploading to object storage from the code; Qdrant's own S3 snapshot storage.
- Backing up the job database or the keys file. The README says to back the keys file up as
  the secret it is.

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server -q   # snapshot round trip
uv run ruff check && uv run ruff format --check && uv run mypy

# the drill, on the real index
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-10/before.json
uv run rag-backup create && uv run rag-backup list
# lose the index: delete the alias's target collection (curl -X DELETE …), then
uv run rag-query; echo "exit $?"                           # 2: no index
time uv run rag-backup restore backups/rag_qa/<newest>.snapshot   # RTO, measured
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-10/after.json
diff ~/dev/rag-s3-10/before.json ~/dev/rag-s3-10/after.json && echo IDENTICAL
docker compose --profile backup up -d && docker compose logs backup | tail -3
gh pr checks
```

## Review notes for the human

- **Restore's order:** verify, check the embedder, upload under a new name, correct the
  metadata, then switch under the lock.
- **A failure before the switch** must leave the live index untouched, and nothing behind
  but an orphan generation, which recovery removes.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
