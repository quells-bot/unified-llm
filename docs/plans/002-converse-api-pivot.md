# Plan 002: Pivot to Converse API with Conversation Serialization

## Status: Proposed

## Summary

Replace the current `InvokeModel` + per-provider `Adapter` pattern with a
`Conversation`-centric wrapper around the Bedrock Converse API. The
`Conversation` type holds the entire conversation state as serializable data,
making it a natural fit for Temporal workflow payloads.

## Motivation

The project exists to provide ergonomic types that serialize cleanly for
Temporal persistence. The current architecture routes through `InvokeModel`
(raw bytes) and requires per-provider adapters to translate between unified
types and provider-specific JSON formats. The Converse API already provides a
unified structured interface across all Bedrock models, making the adapter
layer redundant.

## Architecture

### Before

```
Unified Types -> Adapter (per-provider) -> JSON bytes -> InvokeModel -> JSON bytes -> Adapter -> Unified Types
```

### After

```
Conversation -> toConverseInput() -> bedrock.Converse() -> fromConverseOutput() -> Conversation
```

One translation layer. Not pluggable, not an interface -- there is only one
target API.

## Key Types

### Conversation (new, central type)

```go
type Conversation struct {
    Model    string           `json:"model"`
    System   []string         `json:"system,omitempty"`
    Messages []Message        `json:"messages"`
    Tools    []ToolDefinition `json:"tools,omitempty"`
    Config   Config           `json:"config,omitempty"`
    Usage    Usage            `json:"usage"`
}
```

Pure data. No client reference, no hidden state. Temporal workflows pass this
into activities and get it back updated.

### Config (new, replaces scattered params)

```go
type Config struct {
    MaxTokens     *int        `json:"max_tokens,omitempty"`
    Temperature   *float64    `json:"temperature,omitempty"`
    TopP          *float64    `json:"top_p,omitempty"`
    StopSequences []string    `json:"stop_sequences,omitempty"`
    ToolChoice    *ToolChoice `json:"tool_choice,omitempty"`
}
```

### Response (slimmed down)

```go
type Response struct {
    Message      Message      `json:"message"`
    FinishReason FinishReason `json:"finish_reason"`
    Usage        Usage        `json:"usage"`
}
```

Per-turn return value from `Send`. No `Raw` field -- the Converse API already
gives structured data.

### Existing types (kept, evolved)

- `Message` -- add json tags, keep as-is
- `ContentPart` -- add json tags, keep as-is
- `ToolCallData`, `ToolResultData`, `ThinkingData` -- add json tags
- `ToolDefinition` -- add json tags
- `Usage` -- add json tags, drop `Raw`
- `FinishReason` -- drop `Raw` field

## API Surface

```go
client := llm.NewClient(bedrock)

// Start a conversation
conv := llm.NewConversation("us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    llm.WithSystem("You are a helpful assistant."),
    llm.WithTools(tools...),
    llm.WithMaxTokens(4096),
)

// Send returns updated conversation + response
conv, resp, err := client.Send(ctx, conv, llm.UserMessage("hello"))

// Check for tool use
if resp.FinishReason == llm.FinishReasonToolUse {
    results := executeTools(resp.Message.ToolCalls())
    conv, resp, err = client.Send(ctx, conv, results...)
}
```

### Send semantics

1. Append provided messages to the conversation
2. Translate to Converse SDK types, call `bedrock.Converse()`
3. Append assistant response, accumulate usage
4. Return updated conversation + per-turn response

### Serialization round-trip

```go
data, _ := json.Marshal(conv)
// ... persist to Temporal, database, file, etc.
var restored llm.Conversation
json.Unmarshal(data, &restored)
conv, resp, err = client.Send(ctx, restored, nextMessage)
```

## Converse API Translation

### Internal functions (not exported)

```go
func toConverseInput(conv *Conversation) *bedrockruntime.ConverseInput
func fromConverseOutput(out *bedrockruntime.ConverseOutput) (*Message, *Usage, FinishReason, error)
```

### Content block mapping

| Our Type               | Converse SDK Type                    |
|------------------------|--------------------------------------|
| ContentPart{Text}      | ContentBlockMemberText               |
| ContentPart{ToolCall}  | ContentBlockMemberToolUse            |
| ContentPart{ToolResult}| ContentBlockMemberToolResult         |
| ContentPart{Image}     | ContentBlockMemberImage              |
| ContentPart{Thinking}  | ContentBlockMemberThinking           |
| System strings         | SystemContentBlockMemberText         |
| ToolDefinition         | ToolMemberToolSpec                   |

### Graceful degradation

When building `ConverseInput`, check the model ID prefix and silently drop
unsupported features:

- **Anthropic models** (`anthropic.`, `us.anthropic.`): Full feature set.
  Prompt caching via `CachePointBlock`, extended thinking via
  `AdditionalModelRequestFields`.
- **Other models**: Strip thinking blocks from message history, skip cache
  points, omit anthropic-specific additional fields.

Simple model-prefix check, not a capability registry.

## Middleware

Kept as-is. The middleware mechanism is orthogonal to the API pivot.
Middleware wraps `Send` instead of `InvokeModel`. Useful for:

- Logging / observability
- Token usage tracking
- Injecting prompt cache points
- Rate limiting

## File Changes

| File           | Action  | Notes                                          |
|----------------|---------|-------------------------------------------------|
| `types.go`     | Evolve  | Add json tags, add Conversation/Config, drop Request, slim Response |
| `client.go`    | Rewrite | Converse instead of InvokeModel, Send(Conversation, ...Message) |
| `converse.go`  | New     | toConverseInput / fromConverseOutput translation |
| `adapter.go`   | Delete  | Adapter interface no longer needed               |
| `anthropic.go` | Delete  | Provider-specific translation no longer needed   |
| `openai.go`    | Delete  | Provider-specific translation no longer needed   |
| `middleware.go` | Update | Wrap Send instead of InvokeModel                 |
| `errors.go`    | Keep    | Error classification still relevant              |
| Tests          | Rewrite | Test against Conversation/Send API               |

## Implementation Order

1. Add json tags to existing types, add Conversation and Config types
2. Write converse.go translation layer (toConverseInput / fromConverseOutput)
3. Rewrite client.go around Conversation + Send
4. Update middleware to wrap Send
5. Delete adapter.go, anthropic.go, openai.go
6. Rewrite tests
7. Update examples if any exist

## Open Questions

None remaining. Ready for implementation.
