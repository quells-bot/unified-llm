# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
go test ./...                    # run all tests
go test ./llm/... -run TestName  # run a single test
go build ./...                   # build
go vet ./...                     # vet
go fmt ./...                     # format
go run ./examples/tools/         # run the tools example (requires AWS credentials)
```

## Architecture

This is a Go library (`github.com/quells-bot/unified-llm`) with a single package `llm` that supports multiple LLM backends through a `Provider` interface. The AWS Bedrock provider uses `aws-sdk-go-v2/service/bedrockruntime`; the OpenAI-compatible provider uses only stdlib (`net/http` + `encoding/json`).

### Core design

`Conversation` is the central state object — it holds the model ID, system prompts, message history, tool definitions, inference config, and cumulative token usage. It is fully JSON-serializable, designed to be usable as a Temporal workflow payload or stored between turns.

`Client.Send(ctx, conv, messages...)` is the primary entry point. It:
1. Appends the new messages to a **copy** of `conv.Messages` (caller's conversation is never mutated)
2. Runs the middleware chain
3. Delegates to the configured `Provider` implementation
4. Appends the assistant response and accumulates usage onto the returned conversation

The returned `Conversation` is the updated state; the returned `*Response` is the per-turn result.

### Provider interface (`client.go`)

`Provider` is the abstraction that decouples `Client` from any specific backend:

```go
type Provider interface {
    Send(ctx context.Context, conv *Conversation) (*Response, error)
}
```

Each provider owns its full pipeline: type translation, API call, response translation, and error classification into `*Error` with the appropriate `ErrorKind`.

- **`NewClient(bedrock)`** — convenience constructor for the Bedrock provider (backward compatible)
- **`NewClientWithProvider(provider)`** — generic constructor for any provider

### Providers

**BedrockProvider** (`provider_bedrock.go`):
- Wraps `BedrockConverser` interface (mockable in tests)
- Uses `toConverseInput` / `fromConverseOutput` in `converse.go` for translation
- Consecutive `RoleTool` messages are merged into a single Bedrock user message
- Anthropic models get cache points automatically appended after system blocks and tool definitions
- Error classification via Bedrock SDK exception types

**OpenAIProvider** (`provider_openai.go`):
- Calls `POST {baseURL}/v1/chat/completions` — works with llama.cpp, vLLM, Ollama, OpenAI, etc.
- Stdlib only (`net/http` + `encoding/json`), no additional dependencies
- Configurable via `WithAPIKey(key)` and `WithHTTPClient(c)`
- Error classification via HTTP status codes

### Tool handling pattern

Tools are defined with `NewTool(name, description, params...)` using `StringParam`, `IntegerParam`, `BoolParam`, etc. (and `Optional*` variants). `ToolDefinition.ParseArgs(tc)` validates required fields and types. `ToolCallData.Result(content)` and `.ErrorResult(content)` create the `Message` values to pass back to `Send`. See `examples/tools/main.go` for the full tool loop pattern.

### Middleware

`WithMiddleware(m ...Middleware)` on `NewClient` or `NewClientWithProvider`. First registered = outermost wrapper. Signature: `func(ctx, conv *Conversation, next SendFunc) (*Response, error)`. Middleware works identically regardless of provider.

### Error handling

All errors from `Send` are `*llm.Error` with a `Kind` field (`ErrRateLimit`, `ErrContextLength`, `ErrContentFilter`, etc.) and an `Unwrap`-able `Cause`. Each provider classifies its own errors into these kinds.
