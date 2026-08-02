# Mirai agent bootstrap

`agent/AGENTS.md` is the single source of truth for engineering rules. It is
imported below so the contract is in context before any edit; do not restate or
fork the rules in this file.

@agent/AGENTS.md

Use [`agent/architecture.json`](agent/architecture.json) to locate ownership and
[`agent/checks.json`](agent/checks.json) to select the smallest sufficient
validation path.

Public files are publication-bound. Never add private infrastructure details,
absolute workstation paths, credentials, session history, research backlogs, or
implementation narratives. Local development material belongs in the ignored
`.dev-private/` directory.
