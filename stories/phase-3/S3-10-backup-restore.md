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
       `RAG_BACKUP_PATH`, gitignored), beside a sidecar JSON: the generation, the whole
       manifest (a few hundred bytes a document), Qdrant's sha256, the server's version and
       the time;
    3. verify the download against the checksum;
    4. delete the server's copy, in a `finally`. A snapshot file outlives its collection and
       cannot be listed or downloaded once the collection goes, so nothing could clean it up
       later (DEC-25, design review 2);
    5. keep the newest `--keep` (7).
  - **`list`:** what is on disk, newest first, with the sidecar's summary.
  - **`restore FILE`,** holding the writer lock from the upload to the end, as `create` and
    `rag-ingest` hold it (DEC-25, design review 1: with the lock taken only to switch, an
    ingest starting mid-restore would delete the uploaded generation as a crash's orphan, or
    publish over the restore):
    1. verify the checksum;
    2. refuse a snapshot the config cannot use, from its sidecar. The whole manifest must load
       under the current model, and `check_index`'s rule must pass: the embedder's identity,
       and the sparse one with a hybrid retriever (design review 2; the first draft checked
       only the embedder);
    3. take the writer lock;
    4. check that the new name is free, then upload under it, through Qdrant's
       `snapshots/upload` endpoint:
       `client.http.snapshots_api.recover_from_uploaded_snapshot(collection_name, wait,
       priority, checksum, snapshot)`, the only way `qdrant-client` exposes it (checked).
       **Uploaded to a name that exists, Qdrant replaces that collection's data in place:**
       given the alias's name, it overwrote the live generation, with no switch (checked,
       design review 2). So restore refuses a name taken by a collection or an alias. The
       checksum goes with the upload, and a wrong one is refused with a 400, leaving nothing;
    5. delete the copy the upload left in the server's `snapshots/` folder, then write the
       manifest whole, with `generation` corrected;
    6. switch the alias, then recover.

    It prints what it replaced, and says how to switch back (the previous generation is
    kept). `--dry-run` stops after step 2.
  - **`create --every 24h`** loops, for the compose service.
  - Exit codes as `rag-ingest`'s: 0 done, 1 failed after starting, 2 could not start.
- **The vectorstore seam:** `snapshot(store, generation, dest)` and
  `restore(store, path, *, expect_embedder)`. `backup.py` never imports `qdrant_client`
  (the architecture test).
- **Schedule:** a compose `backup` profile, with the app's image running
  `rag-backup create --every 24h --keep 7`. It mounts `./backups` and `./var`, under the
  app's uid: without `./var` it would not share the writer lock (DEC-25, design review 2).
  Each run takes the lock for itself, and a run that finds an ingest holding it tries again
  5 minutes later. README gives a cron line and a systemd timer for a host without compose.
- **Tests.**
  - **Hermetic:** the sidecar, the checksum verification, the refusals, retention and the
    lock, with the snapshot calls faked.
  - **`qdrant_server`, in CI's `eval-retrieval` job against its service container:** on a
    small collection, create, download, checksum, restore, switch, then a query that
    returns the same points.
  - **Its first assertion:** the restored generation carries the manifest in its metadata. A
    collection's `config.json` holds its metadata, so a snapshot should carry it (design
    review 1). Design review 2 ran the round trip against the server: the metadata, the
    payload index on `metadata.source` and every point came back. The sidecar's copy stays
    the fallback.
  - **Nothing left behind** (design review 2): a restore to a taken name is refused before
    any upload, and after `create`, `restore` and `recover`, the server's `snapshots/` folder
    holds nothing of ours. `recover` deletes a generation's snapshots before the generation.
- **README: "Backup and restore",** a runbook.
  - **The commands,** and checking a backup's sidecar.
  - **Restoring, then switching back** if needed. A switch back by hand makes the generation
    switched away from a newer orphan, which the next ingest deletes (design review 2).
  - **A lost live collection:** its alias goes with it, the previous generation survives
    (DEC-19), and one line points the alias at it.
  - **Never delete `snapshots/tmp/`** on Qdrant's volume: uploads land there first, and
    without it every upload fails.
  - **Copying `backups/` off the host** (an rclone or `aws s3 sync` line): the operator's
    step, an SRS difference.
  - **RPO:** the backup interval for the index. The corpus and git are the source of truth,
    and their backup is the operator's.
  - **RTO:** about a minute from a snapshot; a re-ingest from the corpus, from NFR-3's
    table.
  - **No alert fires on a stale backup:** the backup service has no metrics listener.
    `rag-backup list` shows the newest; check it after any change to the schedule.

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

# the drill, on the real index. Since S3-9 the stack's Qdrant wants its key, which settings
# never read from .env: load it into this shell first (caveat 32)
set -a; . ./.env; set +a
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-10/before.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-10/before.json && echo IDENTICAL   # §3.5
uv run rag-backup create && uv run rag-backup list
# lose the index: delete the alias's target collection, which removes the alias too:
#   curl -s -X DELETE -H "api-key: $RAG_QDRANT_API_KEY" 127.0.0.1:6333/collections/<target>
# then
uv run rag-query; echo "exit $?"                           # 2: no index
time uv run rag-backup restore backups/rag_qa/<newest>.snapshot   # RTO, measured
# the previous generation survived the loss, and the server keeps no snapshot of ours
curl -s -H "api-key: $RAG_QDRANT_API_KEY" 127.0.0.1:6333/collections \
  | jq -r '.result.collections[].name'
docker compose exec qdrant find /qdrant/snapshots -type f -not -path '*/tmp/*'   # nothing
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-10/after.json
diff ~/dev/rag-s3-10/before.json ~/dev/rag-s3-10/after.json && echo IDENTICAL
docker compose --profile backup up -d && docker compose logs backup | tail -3
gh pr checks
```

## Review notes for the human

- **Restore's order:** verify, check the embedder from the sidecar, take the lock, upload
  under a new name, correct the metadata, switch, recover. The lock is held from the upload
  to the end.
- **A failure before the switch** must leave the live index untouched, and nothing behind
  but an orphan generation, which recovery removes.
- **No taken name, no leftover** (design review 2): restore refuses a name that exists, and
  no snapshot file of ours stays on the server.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
