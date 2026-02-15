# Unified LLM Client — Go Library Design

## Context

This document describes the design of a Go library that provides a unified interface for calling LLM providers through AWS Bedrock. It is scoped to Anthropic (Claude) and OpenAI-style models accessed via the Bedrock Runtime SDK v2's `InvokeModel` API.

The primary consumer is a Temporal-based agentic workflow. Tool execution happens externally as Temporal activities — the library handles single LLM calls and message translation, not tool loops or orchestration.

### Reference

- [SPEC.md](/SPEC.md) — Language-agnostic unified LLM client specification
- [BRAINSTORM.md](/BRAINSTORM.md) — Design process guidelines

### Scope

| Concern | Decision |
|---|---|
| Language | Go |
| Transport | AWS Bedrock Runtime SDK v2, `InvokeModel` API |
| Providers | Anthropic (Claude), OpenAI (Chat Completions format on Bedrock) |
| Prompt caching | Must have. Auto-inject for Anthropic. |
| Tool use | Must have. Passive (no execute handlers). |
| Streaming | Not a goal |
| Async | Not a goal (Go is synchronous) |
| Retries | Not in library. Handled by AWS SDK config + Temporal activity retry policies. |
| Gemini | Not a goal (not on Bedrock) |

### Why not Bedrock Converse API?

Bedrock's Converse API provides a unified message format across all Bedrock models, but it does not expose Anthropic's `cache_control` annotations. Without `cache_control`, every turn of an agentic conversation reprocesses the full system prompt and history at full price. With it, cached input tokens cost 90% less. This library is essentially a custom Converse API that preserves prompt caching.

---

## Architecture

Three layers, all in a single Go package (`llm`):

```
┌─────────────────────────────────────────────────┐
│  Layer 3: Client                                │
│  Complete(), middleware chain, provider routing  │
├─────────────────────────────────────────────────┤
│  Layer 2: Adapters                              │
│  AnthropicAdapter, OpenAIAdapter                │
│  BuildInvokeInput() / ParseResponse()           │
├─────────────────────────────────────────────────┤
│  Layer 1: Types                                 │
│  Message, Request, Response, Role, ContentPart  │
│  Usage, ToolDefinition, ToolChoice, Error       │
└─────────────────────────────────────────────────┘
```

### Package Layout

```
github.com/quells-bot/unified-llm/
├── go.mod
├── go.sum
├── llm/
│   ├── types.go           // Message, Request, Response, Role, ContentPart, etc.
│   ├── client.go          // Client, middleware, Complete()
│   ├── errors.go          // Error, ErrorKind, Bedrock error classification
│   ├── anthropic.go       // AnthropicAdapter
│   ├── openai.go          // OpenAIAdapter
│   ├── anthropic_test.go  // Golden file tests for request/response translation
│   └── openai_test.go     // Golden file tests for request/response translation
├── docs/
│   └── plans/
│       └── 2026-02-15-unified-llm-client-design.md
├── SPEC.md
└── BRAINSTORM.md
```

Module path: `github.com/quells-bot/unified-llm`
Package import: `github.com/quells-bot/unified-llm/llm`

---

## Types (Layer 1)

### Role

```go
type Role string

const (
    RoleSystem    Role = "system"
    RoleUser      Role = "user"
    RoleAssistant Role = "assistant"
    RoleTool      Role = "tool"
)
```

No `Developer` role. Both Anthropic and OpenAI-on-Bedrock merge developer instructions with the system prompt.

### ContentPart

Tagged union using a `Kind` field. Only one data field is populated per part.

```go
type ContentKind string

const (
    ContentText         ContentKind = "text"
    ContentImage        ContentKind = "image"
    ContentToolCall     ContentKind = "tool_call"
    ContentToolResult   ContentKind = "tool_result"
    ContentThinking     ContentKind = "thinking"
)

type ContentPart struct {
    Kind       ContentKind
    Text       string          // populated when Kind == ContentText
    Image      *ImageData      // populated when Kind == ContentImage
    ToolCall   *ToolCallData   // populated when Kind == ContentToolCall
    ToolResult *ToolResultData // populated when Kind == ContentToolResult
    Thinking   *ThinkingData   // populated when Kind == ContentThinking
}
```

### Supporting Data Types

```go
type ImageData struct {
    URL       string // URL or data URI
    Data      []byte // raw image bytes (alternative to URL)
    MediaType string // e.g., "image/png"
}

type ToolCallData struct {
    ID        string          // provider-assigned unique ID
    Name      string          // tool name
    Arguments json.RawMessage // parsed JSON arguments
}

type ToolResultData struct {
    ToolCallID string // correlates to ToolCallData.ID
    Content    string // tool output (text)
    IsError    bool   // true if tool execution failed
}

type ThinkingData struct {
    Text      string // reasoning content
    Signature string // Anthropic signature for round-tripping
}
```

### Message

```go
type Message struct {
    Role       Role
    Content    []ContentPart
    ToolCallID string // for tool result messages, links to the tool call
}
```

**Convenience constructors:**

```go
func SystemMessage(text string) Message
func UserMessage(text string) Message
func AssistantMessage(text string) Message
func ToolResultMessage(callID, content string, isError bool) Message
```

**Convenience accessor:**

```go
// Text concatenates all text content parts in the message.
func (m Message) Text() string
```

### ToolDefinition

```go
type ToolDefinition struct {
    Name        string          // unique identifier, [a-zA-Z][a-zA-Z0-9_]*, max 64 chars
    Description string          // human-readable description for the model
    Parameters  json.RawMessage // JSON Schema with root type "object"
}
```

### ToolChoice

```go
type ToolChoiceMode string

const (
    ToolChoiceAuto     ToolChoiceMode = "auto"
    ToolChoiceNone     ToolChoiceMode = "none"
    ToolChoiceRequired ToolChoiceMode = "required"
    ToolChoiceNamed    ToolChoiceMode = "named"
)

type ToolChoice struct {
    Mode     ToolChoiceMode
    ToolName string // required when Mode == ToolChoiceNamed
}
```

### Request

```go
type Request struct {
    Model            string            // Bedrock model ID (e.g., "anthropic.claude-opus-4-6-20250514")
    Messages         []Message         // the conversation
    Provider         string            // "anthropic" or "openai"; uses default if empty
    Tools            []ToolDefinition  // optional
    ToolChoice       *ToolChoice       // optional; defaults to auto if Tools is non-empty
    Temperature      *float64          // optional
    TopP             *float64          // optional
    MaxTokens        *int              // optional (required for Anthropic; adapter defaults to 4096)
    StopSequences    []string          // optional
    ReasoningEffort  string            // "low", "medium", "high" (OpenAI reasoning models)
    ProviderOptions  map[string]any    // escape hatch for provider-specific params
}
```

### Response

```go
type Response struct {
    ID           string       // provider-assigned response ID
    Model        string       // actual model used
    Provider     string       // which provider fulfilled the request
    Message      Message      // the assistant's response
    FinishReason FinishReason // why generation stopped
    Usage        Usage        // token counts
    Raw          []byte       // raw provider response JSON
}
```

**Convenience accessors:**

```go
// Text returns concatenated text from all text content parts.
func (r *Response) Text() string

// ToolCalls returns all tool call content parts from the response message.
func (r *Response) ToolCalls() []ToolCallData
```

### FinishReason

```go
type FinishReason struct {
    Reason string // unified: "stop", "length", "tool_calls", "content_filter", "error"
    Raw    string // provider's native finish reason string
}
```

Provider mapping:

| Provider  | Provider Value  | Unified Value  |
|-----------|----------------|----------------|
| Anthropic | end_turn       | stop           |
| Anthropic | stop_sequence  | stop           |
| Anthropic | max_tokens     | length         |
| Anthropic | tool_use       | tool_calls     |
| OpenAI    | stop           | stop           |
| OpenAI    | length         | length         |
| OpenAI    | tool_calls     | tool_calls     |
| OpenAI    | content_filter | content_filter |

### Usage

```go
type Usage struct {
    InputTokens     int  // tokens in the prompt
    OutputTokens    int  // tokens generated by the model
    CacheReadTokens int  // tokens served from prompt cache
    CacheWriteTokens int // tokens written to prompt cache
    ReasoningTokens int  // tokens used for reasoning/thinking
}

// Add sums two Usage values.
func (u Usage) Add(other Usage) Usage
```

Provider field mapping:

| Unified Field    | Anthropic Field                  | OpenAI Field                                 |
|-----------------|----------------------------------|----------------------------------------------|
| InputTokens     | usage.input_tokens               | usage.prompt_tokens                          |
| OutputTokens    | usage.output_tokens              | usage.completion_tokens                      |
| CacheReadTokens | usage.cache_read_input_tokens    | usage.prompt_tokens_details.cached_tokens    |
| CacheWriteTokens| usage.cache_creation_input_tokens| (not provided)                               |
| ReasoningTokens | (sum of thinking block tokens)   | usage.completion_tokens_details.reasoning_tokens |

---

## Adapter Interface (Layer 2)

```go
// Adapter translates between unified types and a provider's native format.
type Adapter interface {
    // Provider returns the provider name (e.g., "anthropic", "openai").
    Provider() string

    // BuildInvokeInput translates a unified Request into Bedrock InvokeModel parameters.
    BuildInvokeInput(req *Request) (*InvokeInput, error)

    // ParseResponse translates a raw Bedrock InvokeModel response into a unified Response.
    ParseResponse(body []byte, req *Request) (*Response, error)
}

// InvokeInput carries the parameters for a Bedrock InvokeModel call.
type InvokeInput struct {
    ModelID     string // Bedrock model ID
    Body        []byte // serialized JSON in the provider's native format
    ContentType string // e.g., "application/json"
    Accept      string // e.g., "application/json"
}
```

### AnthropicAdapter

Translates to/from the Anthropic Messages API format.

**Request translation (`BuildInvokeInput`):**

1. Extract system messages from `req.Messages` into a top-level `system` field (array of content blocks).
2. Set `anthropic_version: "bedrock-2023-05-31"` (required by Bedrock).
3. Translate remaining messages to Anthropic format:
   - `RoleUser` → `"user"` role
   - `RoleAssistant` → `"assistant"` role
   - `RoleTool` → `"user"` role with `tool_result` content blocks
4. Enforce strict user/assistant alternation by merging consecutive same-role messages.
5. Translate `ToolDefinition` → `{"name", "description", "input_schema"}` format.
6. Translate `ToolChoice`:
   - `auto` → `{"type": "auto"}`
   - `none` → omit tools from request entirely
   - `required` → `{"type": "any"}`
   - `named` → `{"type": "tool", "name": "..."}`
7. Set `max_tokens` (default 4096 if not specified — Anthropic requires it).
8. Preserve thinking blocks and signatures in assistant messages for round-tripping.
9. Merge `req.ProviderOptions["anthropic"]` into the request body.

**Auto-inject `cache_control`:**

1. On the last content block of the `system` array: add `"cache_control": {"type": "ephemeral"}`.
2. On the last tool definition (if tools are present): add `"cache_control": {"type": "ephemeral"}`.

**Response translation (`ParseResponse`):**

1. Parse the Anthropic response JSON.
2. Map content blocks:
   - `"text"` → `ContentPart{Kind: ContentText, Text: ...}`
   - `"tool_use"` → `ContentPart{Kind: ContentToolCall, ToolCall: ...}`
   - `"thinking"` → `ContentPart{Kind: ContentThinking, Thinking: ...}`
3. Map `stop_reason` → `FinishReason` (see table above).
4. Map usage fields → `Usage` (see table above).
5. Store raw response bytes in `Response.Raw`.

### OpenAIAdapter

Translates to/from the OpenAI Chat Completions format (this is what Bedrock uses for OpenAI-compatible models).

**Request translation (`BuildInvokeInput`):**

1. Translate messages to Chat Completions format:
   - `RoleSystem` → `"system"` role
   - `RoleUser` → `"user"` role
   - `RoleAssistant` → `"assistant"` role with optional `tool_calls` array
   - `RoleTool` → `"tool"` role with `tool_call_id`
2. Translate `ToolDefinition` → `{"type": "function", "function": {"name", "description", "parameters"}}`.
3. Translate `ToolChoice`:
   - `auto` → `"auto"`
   - `none` → `"none"`
   - `required` → `"required"`
   - `named` → `{"type": "function", "function": {"name": "..."}}`
4. Map `ReasoningEffort` → `reasoning_effort` in request body (for o-series models).
5. Merge `req.ProviderOptions["openai"]` into the request body.

**Prompt caching:** Automatic on Bedrock for OpenAI models. No annotations needed. Cache usage is reported in the response.

**Response translation (`ParseResponse`):**

1. Parse the Chat Completions response JSON.
2. Extract `choices[0].message.content` → text content parts.
3. Extract `choices[0].message.tool_calls` → tool call content parts.
4. Map `choices[0].finish_reason` → `FinishReason`.
5. Map `usage` → `Usage`.
6. Store raw response bytes in `Response.Raw`.

---

## Client (Layer 3)

```go
// Client routes requests to adapters and calls Bedrock InvokeModel.
type Client struct {
    bedrock         BedrockInvoker
    adapters        map[string]Adapter
    defaultProvider string
    middleware      []Middleware
}

// BedrockInvoker abstracts the Bedrock InvokeModel call for testing.
type BedrockInvoker interface {
    InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}
```

### Construction

```go
func NewClient(bedrock BedrockInvoker, opts ...ClientOption) *Client

type ClientOption func(*clientConfig)

func WithAdapter(a Adapter) ClientOption
func WithDefaultProvider(provider string) ClientOption
func WithMiddleware(m ...Middleware) ClientOption
```

The caller creates the `bedrockruntime.Client` themselves with their own AWS config (region, credentials, retry settings) and passes it in. This keeps AWS configuration entirely out of the library.

**Example:**

```go
cfg, _ := config.LoadDefaultConfig(ctx, config.WithRegion("us-east-1"))
brc := bedrockruntime.NewFromConfig(cfg)

client := llm.NewClient(brc,
    llm.WithAdapter(llm.NewAnthropicAdapter()),
    llm.WithAdapter(llm.NewOpenAIAdapter()),
    llm.WithDefaultProvider("anthropic"),
    llm.WithMiddleware(loggingMiddleware),
)
```

### Middleware

```go
type Middleware func(ctx context.Context, req *Request, next CompleteFunc) (*Response, error)
type CompleteFunc func(ctx context.Context, req *Request) (*Response, error)
```

Middleware wraps each `Complete` call in onion/chain-of-responsibility order. First registered = outermost wrapper.

**Example: logging middleware**

```go
func loggingMiddleware(ctx context.Context, req *Request, next llm.CompleteFunc) (*llm.Response, error) {
    start := time.Now()
    log.Printf("LLM request: provider=%s model=%s", req.Provider, req.Model)

    resp, err := next(ctx, req)
    if err != nil {
        log.Printf("LLM error after %v: %v", time.Since(start), err)
        return nil, err
    }

    log.Printf("LLM response: tokens=%d latency=%v",
        resp.Usage.InputTokens+resp.Usage.OutputTokens, time.Since(start))
    return resp, nil
}
```

### Complete

```go
func (c *Client) Complete(ctx context.Context, req *Request) (*Response, error)
```

**Flow:**

1. **Resolve provider.** Use `req.Provider` if set, otherwise `c.defaultProvider`. Return `ErrConfig` if neither is set.
2. **Look up adapter.** Return `ErrConfig` if the provider has no registered adapter.
3. **Build middleware chain.** Wrap the core function with middleware in registration order.
4. **Execute.** The core function:
   a. `adapter.BuildInvokeInput(req)` → `InvokeInput`
   b. `c.bedrock.InvokeModel(ctx, invokeModelInput)` → raw response
   c. `adapter.ParseResponse(responseBody, req)` → `Response`
5. **Classify errors.** If `InvokeModel` returns an error, classify it (see Error Handling below).
6. **Return.** `(*Response, error)`

---

## Error Handling

```go
type Error struct {
    Kind     ErrorKind
    Provider string
    Message  string
    Cause    error  // underlying AWS SDK or adapter error
    Raw      []byte // raw response body if available
}

func (e *Error) Error() string
func (e *Error) Unwrap() error

type ErrorKind int

const (
    ErrConfig        ErrorKind = iota // misconfiguration (no provider, no adapter)
    ErrAdapter                        // marshal/unmarshal failure in adapter
    ErrAuthentication                 // 401/403 — invalid credentials
    ErrNotFound                       // 404 — model not found
    ErrInvalidRequest                 // 400 — bad request parameters
    ErrRateLimit                      // 429 — throttled
    ErrServer                         // 500+ — Bedrock internal error
    ErrContextLength                  // input too large for model
    ErrContentFilter                  // blocked by safety guardrails
)
```

**Classification:** The `Complete` method inspects AWS SDK errors and maps them:

| Bedrock Error | ErrorKind |
|---|---|
| `AccessDeniedException` | `ErrAuthentication` |
| `ValidationException` | `ErrInvalidRequest` |
| `ResourceNotFoundException` | `ErrNotFound` |
| `ThrottlingException` | `ErrRateLimit` |
| `ModelTimeoutException` | `ErrServer` |
| `InternalServerException` | `ErrServer` |
| `ModelErrorException` | `ErrServer` |
| Message contains "context length" or "too many tokens" | `ErrContextLength` |
| Message contains "content filter" or "guardrail" | `ErrContentFilter` |

The `Cause` field preserves the original AWS SDK error. Temporal retry policies can use `errors.As` to match on either the library's `*Error` type or the underlying AWS error type.

---

## Prompt Caching

### Anthropic

The Anthropic adapter auto-injects `cache_control: {"type": "ephemeral"}` breakpoints:

| Breakpoint | Placement | Purpose |
|---|---|---|
| 1 | Last content block of the `system` array | Cache system prompt across all turns |
| 2 | Last tool definition | Cache tool definitions across all turns |

Anthropic allows up to 4 breakpoints. This uses 2, leaving room for caller-specified breakpoints via `ProviderOptions` if needed.

**Cache lifetime:** 5 minutes, refreshed on each hit. System prompt and tool definitions are stable across turns, so they get near-100% cache hit rates as long as turns complete within 5 minutes.

**Usage reporting:** `Usage.CacheReadTokens` and `Usage.CacheWriteTokens` are populated from the Anthropic response's `cache_read_input_tokens` and `cache_creation_input_tokens` fields.

### OpenAI

Caching is automatic on Bedrock for OpenAI models. No annotations needed. `Usage.CacheReadTokens` is populated from `prompt_tokens_details.cached_tokens` if available.

---

## Tool Use

Tools are passive — the library defines them and returns tool calls, but never executes them.

**Defining tools:**

```go
tools := []llm.ToolDefinition{
    {
        Name:        "get_weather",
        Description: "Get the current weather for a location",
        Parameters:  json.RawMessage(`{
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }`),
    },
}
```

**Using tools in a Temporal workflow:**

```go
// Activity 1: Initial LLM call
resp, err := client.Complete(ctx, &llm.Request{
    Model:    "anthropic.claude-sonnet-4-5-20250514",
    Messages: []llm.Message{llm.UserMessage("What's the weather in SF?")},
    Tools:    tools,
})

// Workflow checks if the model wants tool calls
if resp.FinishReason.Reason == "tool_calls" {
    for _, tc := range resp.ToolCalls() {
        // Execute tool as a Temporal activity
        result := workflow.ExecuteActivity(ctx, tc.Name, tc.Arguments)
        // ... collect results
    }

    // Activity 2: Send tool results back
    messages := append(originalMessages,
        resp.Message, // assistant message with tool calls
        llm.ToolResultMessage(tc.ID, result, false),
    )
    resp, err = client.Complete(ctx, &llm.Request{
        Model:    "anthropic.claude-sonnet-4-5-20250514",
        Messages: messages,
        Tools:    tools,
    })
}
```

---

## Testing Strategy

Adapters are pure data translators — they take Go types and produce JSON, or take JSON and produce Go types. This makes them ideal for golden file testing.

**Golden file tests for each adapter:**
- `BuildInvokeInput`: Given a `Request`, assert the serialized JSON matches a golden file.
- `ParseResponse`: Given raw response JSON, assert the parsed `Response` matches expected values.

**Test cases per adapter:**
- Simple text message
- Multi-turn conversation
- System prompt extraction
- Tool definitions
- Tool calls in response
- Tool results in request
- Thinking blocks (Anthropic only)
- Cache control injection (Anthropic only)
- ToolChoice modes
- Error responses

**Client tests:**
- Provider routing (correct adapter selected)
- Default provider fallback
- Middleware execution order
- Error classification from AWS SDK errors
- Missing provider → `ErrConfig`

The `BedrockInvoker` interface enables testing the client without real AWS calls.

---

## Dependencies

- `github.com/aws/aws-sdk-go-v2/service/bedrockruntime` — Bedrock InvokeModel API
- Standard library only for everything else (`encoding/json`, `context`, `errors`, `fmt`)

No third-party dependencies beyond the AWS SDK.

---

## What's Deferred

These features from SPEC.md are explicitly out of scope for the initial implementation:

| Feature | Why deferred |
|---|---|
| Streaming (`stream()`) | Not a goal |
| High-level `Generate()` with tool loop | Tool loops are Temporal workflows |
| `GenerateObject()` / structured output | Can be added later |
| Retries / retry policy | AWS SDK config + Temporal handle this |
| Audio/Document content types | Not broadly supported on Bedrock |
| Model catalog | Callers use Bedrock model IDs directly |
| Rate limit info from headers | Bedrock SDK handles throttling |
| Module-level default client | Callers create clients explicitly |
| Gemini adapter | Not on Bedrock |
| OpenAI Responses API adapter | Bedrock uses Chat Completions format |
