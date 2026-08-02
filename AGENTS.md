# Mirai agent bootstrap

Read [`agent/AGENTS.md`](agent/AGENTS.md) before changing this repository. It is
the single source of truth and is not discovered automatically: agent tooling
loads instruction files from the repository root down to the working directory,
which does not include `agent/`.

Use [`agent/architecture.json`](agent/architecture.json) to locate ownership and
[`agent/checks.json`](agent/checks.json) to select the smallest sufficient
validation path.

Public files are publication-bound. Never add private infrastructure details,
absolute workstation paths, credentials, session history, research backlogs, or
implementation narratives. Local development material belongs in the ignored
`.dev-private/` directory.
