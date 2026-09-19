# Spec-Driven Delivery Loop

Lightweight process for implementing the SRS story-by-story with Claude Code.
Not BMAD ceremony, not YOLO autopilot: **spec → architecture pass → sharded
stories → implement with review checkpoints.**

The chain of truth:

```text
SRS.md  (what & why — stable, has FR/NFR/ISS IDs)
   └─▶ ARCHITECTURE.md  (how, per phase — written in plan mode, you confirm once)
          └─▶ stories/phase-N/*.md  (small verifiable units, one session each)
                 └─▶ code + commit  (one commit per story, story ID in message)
```

Session state lives in **files, not chat memory** — `stories/STATUS.md` is the
board, each story file carries its own status. Any fresh session can pick up
exactly where the last one stopped. `CLAUDE.md` auto-loads the rules.

---

## The loop at a glance

| Step | Frequency | Mode | Your role |
| --- | --- | --- | --- |
| 1. Architecture pass | once per phase | Plan mode | Confirm/redirect once |
| 2. Shard into stories | once per phase | Normal | Skim, approve |
| 3. Implement story | per story | Normal (plan mode if story says so) | Wait |
| 4. Review checkpoint | per story | Normal | Review diff, accept/reject |
| 5. Close out | per story | Normal | — |

Steps 3–5 repeat until the phase's exit criterion (SRS §12) is met. Then back
to step 1 for the next phase.

**Releases:** every phase exit is a release (SRS §12 rev 1.1): Phase 0 →
`v0.1`, Phase 1 → `v0.2`, Phase 2 → `v0.3`, Phase 3 → `v1.0`. The final
story of a phase bumps `pyproject.toml`, commits, and `git tag`s — see the
release mapping table in stories/STATUS.md.

---

## Step 1 — Architecture pass (once per phase)

Start a **fresh session in plan mode** (Shift+Tab in the CLI, or the plan
toggle in VS Code). Paste:

> Read CLAUDE.md, SRS.md §12 Phase N, and stories/STATUS.md. In plan mode,
> produce the architecture for Phase N only: file/module layout, key
> interfaces, library choices with versions, and how each open decision in
> STATUS.md should be resolved (with recommendation). Do not write code.
> When I confirm, write it to ARCHITECTURE.md under a "Phase N" section and
> update the Decisions log in stories/STATUS.md.

This is your **one confirmation pass** — read the plan, redirect anything you
disagree with, then approve. After this, implementation sessions don't
re-litigate design; they cite `ARCHITECTURE.md`.

Model: **Fable** for this step (it's the judgment-heavy one).

## Step 2 — Shard the phase into stories

Same session, after the architecture is confirmed:

> Shard ARCHITECTURE.md Phase N into stories in stories/phase-N/ using
> stories/TEMPLATE.md. Each story: completable in one session, independently
> verifiable, ordered by dependency, tagged with the SRS IDs it closes.
> Add them to the board in stories/STATUS.md.

Skim the stories, fix scoping you don't like, done. (Phase 0 is already
sharded — see `stories/phase-0/`.)

## Step 3 — Implement one story

Fresh session (or `/clear`). One story per session — this keeps context
small and matches scattered work sessions. Paste:

> Read CLAUDE.md, then stories/phase-0/S0-3-collapse-trees.md. Implement
> ONLY this story. Follow ARCHITECTURE.md; do not expand scope. Run every
> command in its Verification section and show me the output. Stop before
> committing.

Rules the story file enforces:

- **Scope is the story.** Anything discovered but out of scope becomes a
  note in the story's "Discovered" section, or a new story — never a detour.

- **Verification is not optional.** A story isn't done until its listed
  commands pass, output shown.

- If the story is marked `plan-first: yes`, start the session in plan mode.

Model: **Opus + /fast** for mechanical stories; **Fable** where the story
header says so (migration/design-heavy ones are pre-tagged).

## Step 4 — Review checkpoint (you)

When Claude stops before committing:

1. Look at the diff (`git diff` or the VS Code diff view).
2. Check the acceptance criteria in the story file against the shown output.
3. Optional for bigger stories: run `/code-review` in the same session.
4. Say "approved, commit" — or say what to change and loop within the session.

Keep this honest but light: acceptance criteria were written up front
precisely so review is "did the listed commands pass," not re-reading
every line.

## Step 5 — Close out

After approval, in the same session:

> Commit with message "S0-3: collapse v1/v2/temp into single package
> (ISS-10)". Update the story file status to Done with today's date and
> one line on any deviation from plan. Update stories/STATUS.md. If
> anything was discovered for later, add it as a stub story or a line in
> STATUS.md → Backlog.

Then `/clear` or close the session. Next session starts clean at Step 3
with the next story.

---

## Session hygiene (the part that makes scattered sessions work)

- **One story, one session, one commit.** If a story turns out too big,
  split it in the story file and do half — don't push through.

- **Never carry design decisions in chat.** If something got decided in
  conversation, it goes into ARCHITECTURE.md or the Decisions log in
  STATUS.md before the session ends, or it didn't happen.

- **Start every session with "Read CLAUDE.md"** — it chains to the board
  and the current story so you never re-explain the project.

- **The board is the memory.** `stories/STATUS.md` answers "where was I?"
  in ten seconds, weeks later.

## Escalation triggers — switch the story to Fable when…

- The obvious fix failed twice.
- The story touches LangChain 1.x API surface you haven't used yet.
- You're choosing between libraries/designs, not implementing a chosen one.
- A story's verification passes but behavior looks wrong anyway.

## What NOT to do

- Don't run multiple stories in one long session "while you're at it" —
  that's the YOLO failure mode this loop exists to prevent.

- Don't let a session edit `SRS.md` casually. The SRS changes only via an
  explicit "amend the SRS" session, with the change noted in its revision
  block. Stories cite the SRS; they don't mutate it.

- Don't put working docs in `docs/` — that directory is the RAG corpus and
  gets ingested. Project docs live at repo root or in `stories/`.
