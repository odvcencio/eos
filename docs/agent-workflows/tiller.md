# Active Tiller workflow

Read only when Tiller ambient mode is active. The original operating contract follows unchanged.

When Tiller ambient mode is active, the root Codex session is the orchestrator.
SessionStart adds this context up front; denied execution calls also point back
to this backend's lifecycle tools.
Use `.tiller/scratch/codex/` for terse shared handoff notes, reports, and
claims when useful.
Use Git/GitHub for VCS. Use Graft for coordination, work claims, shared
plans/notes, coord checks, structural inspection/review/blame when available.
Checkpoint verified wins at natural boundaries. Prefer the repo's configured
checkpoint tool when one is present; otherwise use normal Git/GitHub. Stage
explicit paths, inspect the diff, and never include unrelated dirty work.

Root Codex session:
- Read enough context yourself to route the work and make integration
  decisions. Do not spawn a subagent just to read a file, search the tree, or
  inspect ordinary context.
- Spend premium/reason-tier output on durable judgment artifacts: specs, plans,
  architecture notes, implementation docs, reviews, policy rationale,
  checkpoint decisions, distilled ambient state, and high-quality handoff
  briefs.
- Maintain a descriptor-backed task list. Each descriptor should look like a
  portable subagent/task packet that can be mapped to Codex, Claude Code,
  OpenCode, Cursor, or future harnesses.
- Descriptor fields: id/title, role/profile, objective, context paths,
  constraints, expected outputs, verification target, budget tier/model
  ceiling, sandbox/permission needs, dependencies/blockers, checkpoint
  criteria, and report contract.
- Send bulky execution output, shell logs, routine patching, and test loops to
  worker/debugger/cheap subagents.
- When the current run has `status.md` beside `ledger.jsonl`, read `status.md` first for
  compact run state before raw ledger files, including `Distillation`,
  `Arbiter Next Action`, `Stale/Late Work`, `Recommended Next Actions`,
  checkpoint candidates, and advisory `Spend Budget` bands. Read `Distillation`
  before raw logs or transcripts. Prioritize `Arbiter Next Action` before raw
  ledger reads; use `Recommended Next Actions` as legacy/fallback context. If
  spend is warn/over, choose whether to compact, checkpoint, or proceed before
  spending more premium output.
- Keep root output compact; write durable docs/plans when they compound.
- Queue/background independent descriptors and continue useful orchestration.
  Wait only for descriptors that block the next integration decision. Update
  descriptors from returned reports.
- Use root read/search tools and safe read-only shell commands for lightweight
  inspection.
- Load relevant skills directly when a domain-specific workflow applies. For
  Sirena diagram work, use `using-sirena`.
- Do not run implementation shell commands, build/test commands, edit source
  files, or apply patches from the root premium/reason-tier session.

Right-sizing matrix:
- root: direct reads/searches and routing decisions; no subagent needed for
  ordinary context.
- `tiller-scout`: `gpt-5.4-mini` for cheap bounded reconnaissance,
  inventories, docs/log snippets, and simple summaries.
- `tiller-summary`: `gpt-5.4-mini` for compact status updates,
  distilled ambient state, run ledger summaries, stale/late report triage,
  checkpoint candidate synthesis, and Arbiter next-action bookkeeping.
- `tiller-worker`: `gpt-5.5 medium` for bounded implementation, edits,
  builds, and tests.
- `tiller-debugger`: `gpt-5.5 high` for root-cause analysis plus fixes.
- `tiller-investigator`/`tiller-reviewer`: `gpt-5.5 xhigh` read-only for deep
  tracing, adversarial review, and high-stakes verification.
- `tiller-architect`/`tiller-deep-report`: `gpt-5.5 xhigh` for architecture,
  research synthesis, and high-consequence tradeoffs.

Codex delegation mechanics:
- Use the normal Codex multi-agent tools (`spawn_agent`, `wait_agent`,
  `send_input`, `resume_agent`, `close_agent`) with `agent_type` set to one of
  the `tiller-*` agents.
- Use `tiller-summary` for compact status updates, distilled ambient state,
  run ledger summaries, stale/late report triage, checkpoint candidate
  synthesis, and Arbiter next-action bookkeeping instead of spending root output on
  routine status compaction.
  Prefer `status.md` first when it is present in the run directory; prioritize
  `Distillation` and `Arbiter Next Action`; use `Recommended Next Actions` as
  legacy/fallback context. When `Stale/Late Work` is not `none`, triage it
  before raw logs; when `Spend Budget` is warn/over, recommend
  compact/checkpoint/proceed.
- Keep delegated prompts bounded. Include the concrete task, relevant paths,
  expected output, and verification target when known.
- Continue useful orchestration while agents run. When a result returns, review
  it, integrate it, and close the agent.
- Require descriptor-compatible subagent reports to cover: Outcome;
  Distillation when useful; files changed or inspected; verification commands
  and results; caveats or residual risk; checkpoint candidate yes/no;
  Arbiter next action. Use returned reports to update task status,
  distilled state, and checkpoint decisions. Ask subagents to summarize long
  logs and point at files/reports instead of pasting bulky output.
- Treat coherent verified slices as checkpoint candidates. Ask execution agents
  to report exact changed files, verification, and caveats so the checkpoint can
  be committed cleanly with the configured checkpoint tool or normal Git/GitHub.
- If a root tool call is denied by `DenyExecution`, do not retry a variant of
  the same root command. Use `spawn_agent` with the appropriate `agent_type`,
  then `wait_agent`/`close_agent`.
- Tell subagents to read relevant `.tiller/scratch/codex/` notes first when
  present and write final reports or handoff notes there when useful.

Depth model:
- Root orchestrator is depth 0.
- Depth-1 agents may dispatch bounded follow-up work.
- Depth-2 agents are terminal.

Reasoning-tier work should stay focused on investigation, review, planning,
and synthesis. Route mechanical edits and command execution to execution agents.
Prefer terse, direct, explicit technical artifacts and documentation: concrete
paths, commands, diagnostics, decisions, and next actions over broad prose.
