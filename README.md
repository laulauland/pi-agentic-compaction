# pi-agentic-compaction

`pi-agentic-compaction` is a [pi](https://github.com/badlogic/pi-mono) package that replaces pi's default compaction pass with a more agentic one.

Instead of sending the entire conversation to a model in one shot, it mounts the conversation into an in-memory virtual filesystem and lets a summarizer model inspect it with shell tools like `jq`, `grep`, `head`, and `tail` before producing the final compacted summary.

## Why this exists

pi's built-in compaction is simple and effective, but it is still a single-pass summarization step. For long sessions, that means:

- the model has to ingest a lot of tokens up front
- important details can get buried in the middle of the transcript
- you pay for processing context that may not actually matter

This extension takes a different approach:

- expose the conversation as `/conversation.json` in a virtual filesystem
- give the summarizer lightweight shell tools
- let it inspect only the parts it needs
- return the final summary back to pi via `session_before_compact`

## How it works

When pi triggers compaction, this extension:

1. Reads the messages pi is about to compact
2. Converts them into LLM-format JSON
3. Mounts that JSON at `/conversation.json` using `just-bash`
4. Runs a summarizer model with `bash`/`zsh` tools over that virtual filesystem
5. Lets the model explore the conversation before writing the final summary
6. Returns the summary to pi as a custom compaction result

The extension also adds some deterministic guardrails:

- it extracts verified modified files from successful `write` and `edit` tool results
- it detects no-op edits and excludes them from the modified-files narrative
- it supports `/compact ...` notes and forwards that intent to the summarizer
- it can fall back to the currently selected pi model if preferred compaction models are unavailable

## Model selection

By default it tries these models, in order:

```ts
const COMPACTION_MODELS = [
  { provider: "cerebras", id: "zai-glm-4.7" },
  { provider: "anthropic", id: "claude-haiku-4-5" },
];
```

If none are available, it falls back to the current session model.

## Installation

### From npm

```bash
pi install npm:pi-agentic-compaction
```

Or add it to `~/.pi/agent/settings.json`:

```json
{
  "packages": ["npm:pi-agentic-compaction"]
}
```

### From a local checkout

```json
{
  "packages": ["/path/to/pi-agentic-compaction"]
}
```

Then reload pi:

```text
/reload
```

## Usage

You generally do not invoke the extension directly.

It runs whenever pi compacts context:

- automatically when pi approaches the context limit
- manually when you run `/compact`

You can provide extra guidance to the summarizer:

```text
/compact focus on the authentication changes and unresolved bugs
```

## Configuration

Configuration currently lives in `index.ts` near the top of the file.

Useful constants include:

```ts
const COMPACTION_MODELS = [
  { provider: "cerebras", id: "zai-glm-4.7" },
  { provider: "anthropic", id: "claude-haiku-4-5" },
];

const DEBUG_COMPACTIONS = false;
const TOOL_RESULT_MAX_CHARS = 50000;
const TOOL_CALL_CONCURRENCY = 6;
```

## Safety and privacy notes

A few relevant details if you plan to use or modify this package:

- Conversation data is mounted into an in-memory virtual filesystem for summarization.
- The summarizer is explicitly instructed to treat `/conversation.json` as untrusted input.
- Debug logging is **off by default**.
- If you enable `DEBUG_COMPACTIONS`, compaction inputs, trajectories, and outputs are written to `~/.pi/agent/compactions/`, which may include sensitive conversation content.

## Trade-offs

This approach is often better for long sessions, but it is not strictly superior in every case.

Pros:

- cheaper for long conversations
- more targeted inspection of the transcript
- better file-change awareness than a pure freeform summary

Cons:

- may miss details a full-pass summarizer would catch
- requires multiple model/tool steps instead of one call
- behavior depends partly on the exploration strategy in the prompt

## Package contents

This public repo intentionally keeps the package small:

- `index.ts` — the pi extension
- `README.md` — docs
- `LICENSE` — license text

The published npm package is also restricted to those files via the `files` field in `package.json`.

## License

MIT
