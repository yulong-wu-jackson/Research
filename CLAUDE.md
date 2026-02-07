# Project Constitution

## Core Principles

1. **Research Before Code** — Use DeepWiki MCP / search online to learn external tools/libraries & best practice before implementation
2. **Ask When Uncertain** — Use `AskUserQuestion` to clarify requirements
3. **Verify Everything** — Run automated tests AND manual verification
4. **Quality Gates** — Invoke `spec-dev:code-reviewer` after each implementation step
5. **Never Skip Steps** — Complete each step fully before proceeding, there is no token and time constraints, you have infinite time to do tasks, so please make sure to review everything thoroughly and test them out.
6. **Leverage Existing Tools** — Research best practices online; prefer established frameworks over custom solutions
7. **Discuss Decisions** — Discuss and Explain technical choices with recommendations adhering to best practices; use `AskUserQuestion` for any unclear scope or design decisions

Decision should always aimed to benefit the overall research paper

## Tools & Environment

- **Python**: Use `uv` to run Python code
- **Frontend**: Must Use `pnpm` instead of `npm`
- **External Libraries**: Use DeepWiki for documentation research


Always choose the latest stable versions.

## TDD Workflow (Strictly Follow)

1. **Research** → Use DeepWiki MCP / WebSearch / WebFetch to understand tools/libraries, then understand the current repo overall structure and functionality of each file, then look into details for the full content of the file related to the current task
2. **Test First** → Write thorough tests expecting reliable product behavior
3. **Implement** → Write code following best practices
4. **Test** → Run and update tests; ensure functionality works before proceeding. If blocked by credentials or other issues, add a hard STOP, use `AskUserQuestion`
5. **Verify** → Manually verify functionality (use Playwright MCP if applicable), use linter to safely fix the apparent issues
6. **Review** → Invoke `/spec-dev:code-reviewer`
7. **Fix** → Critically review the subagent comments, look into the real files to verify, Address any issues from review
8. **Proceed** → Move to next step only after all checks pass

NOTE: Must discuss and AskUserQuestion when any design decision or technical decision needed to make, give your recommendation with explanation adhere with the best practice. If any scope or aspect is unclear, you must AskUserQuestion to clarify, to make sure everything is reliable and well defined.

IMPORTANT!: There is no token or time constraints, you have infinite time to do the tasks, so please make sure to review everything thoroughly and test them out.

## Git Commits

Use `/spec-dev:commit-msg` to write conventional commit messages. No authorship statements.
