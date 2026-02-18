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

This is a Go library (`github.com/quells-bot/unified-llm`) with a single package `llm` that wraps the AWS Bedrock Converse API. The only dependency is `aws-sdk-go-v2/service/bedrockruntime`.

### Core design

`Conversation` is the central state object — it holds the model ID, system prompts, message history, tool definitions, inference config, and cumulative token usage. It is fully JSON-serializable, designed to be usable as a Temporal workflow payload or stored between turns.

`Client.Send(ctx, conv, messages...)` is the primary entry point. It:
1. Appends the new messages to a **copy** of `conv.Messages` (caller's conversation is never mutated)
2. Runs the middleware chain
3. Calls Bedrock Converse via `BedrockConverser` (an interface, mockable in tests)
4. Appends the assistant response and accumulates usage onto the returned conversation

The returned `Conversation` is the updated state; the returned `*Response` is the per-turn result.

### Key translation layer (`converse.go`)

- Consecutive `RoleTool` messages are merged into a single Bedrock user message (Bedrock requires all tool results for an assistant turn in one message)
- Anthropic models (`strings.Contains(model, "anthropic.")`) get cache points automatically appended after the last system block and after the last tool definition
- `toConverseInput` / `fromConverseOutput` handle all translation between the library's types and Bedrock SDK types

### Tool handling pattern

Tools are defined with `NewTool(name, description, params...)` using `StringParam`, `IntegerParam`, `BoolParam`, etc. (and `Optional*` variants). `ToolDefinition.ParseArgs(tc)` validates required fields and types. `ToolCallData.Result(content)` and `.ErrorResult(content)` create the `Message` values to pass back to `Send`. See `examples/tools/main.go` for the full tool loop pattern.

### Middleware

`WithMiddleware(m ...Middleware)` on `NewClient`. First registered = outermost wrapper. Signature: `func(ctx, conv *Conversation, next SendFunc) (*Response, error)`.

### Error handling

All errors from `Send` are `*llm.Error` with a `Kind` field (`ErrRateLimit`, `ErrContextLength`, `ErrContentFilter`, etc.) and an `Unwrap`-able `Cause`.
