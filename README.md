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

The agentic compaction loop is a good fit for small, fast models. The task is structured: navigate a JSON file, run a few shell commands, and emit a summary in a defined format. That plays to the strengths of instruction-following models like `gpt-5.4-mini` — models that are reliable on well-specified tasks, respond quickly, and are cheap enough that multiple tool-call steps do not become a bottleneck.

By default the extension tries these models, in order:

```ts
const COMPACTION_MODELS = [
  { provider: "cerebras", id: "zai-glm-4.7" },
  { provider: "openai", id: "gpt-5.4-mini" },
];
```

If none are available, it falls back to the current session model.

You can override that interactively with:

```text
/compaction-model
```

That command opens a picker where you can:

- choose multiple compaction models
- reorder them into fallback order
- switch between `scoped` models (from pi's `enabledModels`) and `all` available models
- persist the selection to pi settings

Picker controls:

- type to filter
- `Enter` toggles the highlighted model
- `Alt+↑/↓` reorders a selected model
- `Tab` switches between `all` and `scoped`
- `Ctrl+A` selects all visible models
- `Ctrl+X` clears visible selections
- `Ctrl+S` saves
- `Esc` cancels

You can also choose the save target explicitly:

```text
/compaction-model global
/compaction-model project
```

The effective config is stored under a namespaced block in pi settings:

```json
{
  "pi-agentic-compaction": {
    "models": [
      "cerebras/zai-glm-4.7",
      "openai/gpt-5.4-mini"
    ]
  }
}
```

Locations follow normal pi settings precedence:

- global: `~/.pi/agent/settings.json`
- project: `.pi/settings.json`

Project settings override global settings.

At runtime, the extension tries the persisted models in order and skips any that are unavailable, unauthenticated, or no longer registered. If none work, it falls back to the session model.

### Steerable compaction

Because the summarizer runs as a separate model in its own agentic loop, its behavior is directly steerable. You can pass guidance via `/compact` notes:

```text
/compact focus on the authentication changes and unresolved bugs
```

The note is forwarded into the summarizer's system prompt, biasing both its exploration strategy and its output. Small, instruction-following models tend to respect this kind of explicit steering reliably, which makes the behavior predictable without requiring prompt engineering on the user's part.

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

The extension runs whenever pi compacts context:

- automatically when pi approaches the context limit
- manually when you run `/compact`

You can provide extra guidance to the summarizer:

```text
/compact focus on the authentication changes and unresolved bugs
```

To configure the ordered fallback models used for compaction:

```text
/compaction-model
```

Note: the picker is a TUI command, so it is only available in interactive pi sessions.

## Configuration

There are two layers of configuration:

### 1. Interactive model selection (recommended)

Use `/compaction-model` and persist the selected ordered fallback list into pi settings under:

```json
{
  "pi-agentic-compaction": {
    "models": ["cerebras/zai-glm-4.7", "openai/gpt-5.4-mini"]
  }
}
```

### 2. Code-level defaults

If no persisted model list exists, the extension falls back to the defaults in `index.ts`:

```ts
const COMPACTION_MODELS = [
  { provider: "cerebras", id: "zai-glm-4.7" },
  { provider: "openai", id: "gpt-5.4-mini" },
];
```

When writing settings, the extension updates only the `pi-agentic-compaction.models` field and preserves the rest of the settings file.

Other implementation-level constants still live in `index.ts`, for example:

```ts
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

The agentic approach has different characteristics from a single-pass summarization, and those trade-offs interact with model size in specific ways.

Pros:

- cheaper per compaction for long conversations, since the model reads only what it queries
- more targeted inspection of the transcript rather than ingesting everything at once
- better file-change awareness than a pure freeform summary
- steerable: `/compact` notes let you direct what the summarizer pays attention to
- the structured, tool-use format suits small instruction-following models well

Cons:

- may miss details a full-pass summarizer would catch, since exploration is model-driven
- requires multiple model/tool steps instead of one call
- a smaller model is less likely to self-correct if its exploration strategy is suboptimal
- for sessions with subtle cross-cutting context, a larger model may produce more coherent summaries

### Accuracy considerations

The trade-offs above are worth thinking through when choosing a compaction model.

The agentic loop structure partially compensates for the limitations of smaller models: the model can re-query sections of the transcript it is uncertain about rather than relying on a single pass over everything. And because the format is well-specified — run some shell commands, write a summary — the task plays to the strengths of models that follow instructions precisely.

That said, small models can struggle when sessions involve nuanced reasoning, implicit dependencies, or ambiguous cause-and-effect chains. If the exploration strategy in the prompt does not surface the right parts of the transcript, a smaller model is less likely to recover from that.

If summary quality on complex sessions matters more than speed or cost, either:

- use `/compaction-model` to pick larger models interactively, or
- update `COMPACTION_MODELS` in `index.ts` to change the default fallback list

The rest of the extension is model-agnostic.

## Package contents

This public repo intentionally keeps the package small:

- `index.ts` — the pi extension
- `README.md` — docs
- `LICENSE` — license text

The published npm package is also restricted to those files via the `files` field in `package.json`.

## License

MIT
