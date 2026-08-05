# Security Policy

## Supported versions

logpose is pre-1.0. Fixes land on the latest release only.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a vulnerability

**Please do not open a public issue.** Use GitHub's private reporting —
[Report a vulnerability](https://github.com/xdadwal/logpose/security/advisories/new) —
or email <akshay.dadwal.rajput@gmail.com>.

Useful to include: what you can do with it, a minimal reproduction, the logpose
and Python versions, and which provider is involved. You will get an
acknowledgement within a few days.

Never include a real credential in a report. If your reproduction needs one, a
redacted placeholder plus the shape of the value is enough.

## What counts as a vulnerability here

logpose handles model-provider credentials and executes tool code on behalf of a
model, so the two areas that matter most are:

**Credential exposure.** Any path where a credential value reaches a log line, a
`repr`, an exception message, a traceback, or an outbound request that should
not carry it. This is treated as the highest-severity class in this project.
Concrete examples that would qualify:

- A credential surviving into `str(exc)` or `traceback.format_exception(...)`,
  including through a chained `__cause__`.
- A credential being sent to a host it was not configured for — for instance an
  ambient `$OPENAI_API_KEY` reaching a `localhost` model server.
- Two conflicting auth headers being sent at once, leaking one credential to a
  provider that has no business seeing it.
- logpose writing to the Claude Code credential store. Discovery is strictly
  read-only by design; a write is a bug, and a potentially destructive one.

**Tool execution boundary.** logpose ships no built-in shell, file, or network
tools — it runs exactly the functions you hand it. `discover_tools` is one of the
ways you hand them over: it **imports** every module under the target, executing
that module's top-level code, and registers every `ToolDef` it finds. A discovery
target must therefore never be built from untrusted input — passing one is
equivalent to importing the module yourself. A defect that causes it to
execute something you did not register, or to execute a registered tool with
arguments that bypassed schema validation, is in scope.

## What is out of scope

- **Prompt injection causing a model to call your tools in unintended ways.**
  logpose deliberately does not sandbox your tools. If a tool is dangerous when
  called with attacker-influenced arguments, that gate belongs in the tool.
  Design guidance is in the README.
- **Breakage of the Claude Code subscription auth path.** That path depends on
  undocumented details and is expected to break; see the disclaimer in the
  README. Report it as an ordinary bug.
- Vulnerabilities in `anthropic`, `httpx`, `pydantic`, or a model server —
  report those upstream.
