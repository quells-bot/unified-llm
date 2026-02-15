# Unified LLM Client — Go Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a Go library (`llm` package) that provides a unified interface for calling Anthropic and OpenAI models through AWS Bedrock, with prompt caching, tool use, and middleware support.

**Architecture:** Three-layer design in a single Go package: Layer 1 (types and errors), Layer 2 (Anthropic and OpenAI adapters that translate between unified types and provider-native JSON), Layer 3 (Client that routes requests through middleware to adapters and calls Bedrock InvokeModel).

**Tech Stack:** Go, AWS SDK v2 (`bedrockruntime`), standard library (`encoding/json`, `context`, `errors`). No other dependencies. Golden file testing for adapters.

**Reference:** [Design Document](2026-02-15-unified-llm-client-design.md) — contains all type definitions, adapter specifications, and API contracts.

---

### Task 1: Project Initialization

**Files:**
- Create: `go.mod`
- Create: `llm/doc.go`

**Step 1: Initialize the Go module**

Run: `go mod init github.com/quells-bot/unified-llm`
Expected: `go.mod` created with module path

**Step 2: Add AWS SDK dependency**

Run: `go get github.com/aws/aws-sdk-go-v2/service/bedrockruntime`
Expected: `go.mod` and `go.sum` updated with bedrockruntime dependency

**Step 3: Create package doc file**

Create `llm/doc.go`:

```go
// Package llm provides a unified interface for calling LLM providers through AWS Bedrock.
//
// The package supports Anthropic (Claude) and OpenAI models with automatic prompt caching,
// tool use, and a middleware system for cross-cutting concerns.
package llm
```

**Step 4: Verify the module compiles**

Run: `cd /home/sprite/unified-llm && go build ./...`
Expected: Clean build, no errors

**Step 5: Commit**

```bash
git add go.mod go.sum llm/doc.go
git commit -m "chore: initialize Go module with AWS SDK dependency"
```

---

### Task 2: Core Types — Role, ContentPart, Message

**Files:**
- Create: `llm/types.go`
- Create: `llm/types_test.go`

**Step 1: Write tests for convenience constructors and Message.Text()**

Create `llm/types_test.go`:

```go
package llm

import (
	"testing"
)

func TestSystemMessage(t *testing.T) {
	m := SystemMessage("you are helpful")
	if m.Role != RoleSystem {
		t.Errorf("got role %q, want %q", m.Role, RoleSystem)
	}
	if len(m.Content) != 1 || m.Content[0].Kind != ContentText || m.Content[0].Text != "you are helpful" {
		t.Errorf("unexpected content: %+v", m.Content)
	}
}

func TestUserMessage(t *testing.T) {
	m := UserMessage("hello")
	if m.Role != RoleUser {
		t.Errorf("got role %q, want %q", m.Role, RoleUser)
	}
	if m.Text() != "hello" {
		t.Errorf("got text %q, want %q", m.Text(), "hello")
	}
}

func TestAssistantMessage(t *testing.T) {
	m := AssistantMessage("hi there")
	if m.Role != RoleAssistant {
		t.Errorf("got role %q, want %q", m.Role, RoleAssistant)
	}
	if m.Text() != "hi there" {
		t.Errorf("got text %q, want %q", m.Text(), "hi there")
	}
}

func TestToolResultMessage(t *testing.T) {
	m := ToolResultMessage("call-123", "result data", false)
	if m.Role != RoleTool {
		t.Errorf("got role %q, want %q", m.Role, RoleTool)
	}
	if len(m.Content) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(m.Content))
	}
	if m.Content[0].Kind != ContentToolResult {
		t.Errorf("got kind %q, want %q", m.Content[0].Kind, ContentToolResult)
	}
	tr := m.Content[0].ToolResult
	if tr.ToolCallID != "call-123" || tr.Content != "result data" || tr.IsError {
		t.Errorf("unexpected tool result: %+v", tr)
	}
}

func TestToolResultMessageError(t *testing.T) {
	m := ToolResultMessage("call-456", "something broke", true)
	if !m.Content[0].ToolResult.IsError {
		t.Error("expected IsError to be true")
	}
}

func TestMessageTextConcatenatesAllTextParts(t *testing.T) {
	m := Message{
		Role: RoleAssistant,
		Content: []ContentPart{
			{Kind: ContentText, Text: "hello "},
			{Kind: ContentToolCall, ToolCall: &ToolCallData{ID: "1", Name: "foo"}},
			{Kind: ContentText, Text: "world"},
		},
	}
	if got := m.Text(); got != "hello world" {
		t.Errorf("got %q, want %q", got, "hello world")
	}
}

func TestMessageTextEmptyWhenNoTextParts(t *testing.T) {
	m := Message{
		Role:    RoleAssistant,
		Content: []ContentPart{{Kind: ContentToolCall, ToolCall: &ToolCallData{ID: "1", Name: "foo"}}},
	}
	if got := m.Text(); got != "" {
		t.Errorf("got %q, want empty string", got)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run 'TestSystemMessage|TestUserMessage|TestAssistantMessage|TestToolResultMessage|TestMessageText'`
Expected: FAIL — types not defined

**Step 3: Implement types.go**

Create `llm/types.go`:

```go
package llm

import (
	"encoding/json"
	"strings"
)

// Role represents a message participant.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentKind identifies the type of a ContentPart.
type ContentKind string

const (
	ContentText       ContentKind = "text"
	ContentImage      ContentKind = "image"
	ContentToolCall   ContentKind = "tool_call"
	ContentToolResult ContentKind = "tool_result"
	ContentThinking   ContentKind = "thinking"
)

// ContentPart is a tagged union — only the field matching Kind is populated.
type ContentPart struct {
	Kind       ContentKind
	Text       string          // Kind == ContentText
	Image      *ImageData      // Kind == ContentImage
	ToolCall   *ToolCallData   // Kind == ContentToolCall
	ToolResult *ToolResultData // Kind == ContentToolResult
	Thinking   *ThinkingData   // Kind == ContentThinking
}

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

// Message is a single message in a conversation.
type Message struct {
	Role       Role
	Content    []ContentPart
	ToolCallID string // for tool result messages, links to the tool call
}

// Text concatenates all text content parts in the message.
func (m Message) Text() string {
	var b strings.Builder
	for _, p := range m.Content {
		if p.Kind == ContentText {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// SystemMessage creates a system message with a single text part.
func SystemMessage(text string) Message {
	return Message{
		Role:    RoleSystem,
		Content: []ContentPart{{Kind: ContentText, Text: text}},
	}
}

// UserMessage creates a user message with a single text part.
func UserMessage(text string) Message {
	return Message{
		Role:    RoleUser,
		Content: []ContentPart{{Kind: ContentText, Text: text}},
	}
}

// AssistantMessage creates an assistant message with a single text part.
func AssistantMessage(text string) Message {
	return Message{
		Role:      RoleAssistant,
		Content:   []ContentPart{{Kind: ContentText, Text: text}},
	}
}

// ToolResultMessage creates a tool result message.
func ToolResultMessage(callID, content string, isError bool) Message {
	return Message{
		Role: RoleTool,
		Content: []ContentPart{{
			Kind: ContentToolResult,
			ToolResult: &ToolResultData{
				ToolCallID: callID,
				Content:    content,
				IsError:    isError,
			},
		}},
		ToolCallID: callID,
	}
}

// ToolChoiceMode controls how the model selects tools.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNamed    ToolChoiceMode = "named"
)

// ToolChoice specifies how the model should select tools.
type ToolChoice struct {
	Mode     ToolChoiceMode
	ToolName string // required when Mode == ToolChoiceNamed
}

// ToolDefinition describes a tool the model can call.
type ToolDefinition struct {
	Name        string          // unique identifier
	Description string          // human-readable description
	Parameters  json.RawMessage // JSON Schema with root type "object"
}

// Request is the unified request to any LLM provider.
type Request struct {
	Model           string
	Messages        []Message
	Provider        string
	Tools           []ToolDefinition
	ToolChoice      *ToolChoice
	Temperature     *float64
	TopP            *float64
	MaxTokens       *int
	StopSequences   []string
	ReasoningEffort string         // "low", "medium", "high"
	ProviderOptions map[string]any // escape hatch
}

// FinishReason describes why generation stopped.
type FinishReason struct {
	Reason string // unified: "stop", "length", "tool_calls", "content_filter", "error"
	Raw    string // provider's native string
}

// Usage contains token counts from the response.
type Usage struct {
	InputTokens      int
	OutputTokens     int
	CacheReadTokens  int
	CacheWriteTokens int
	ReasoningTokens  int
}

// Add sums two Usage values.
func (u Usage) Add(other Usage) Usage {
	return Usage{
		InputTokens:      u.InputTokens + other.InputTokens,
		OutputTokens:     u.OutputTokens + other.OutputTokens,
		CacheReadTokens:  u.CacheReadTokens + other.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens + other.CacheWriteTokens,
		ReasoningTokens:  u.ReasoningTokens + other.ReasoningTokens,
	}
}

// Response is the unified response from any LLM provider.
type Response struct {
	ID           string
	Model        string
	Provider     string
	Message      Message
	FinishReason FinishReason
	Usage        Usage
	Raw          []byte // raw provider response JSON
}

// Text returns concatenated text from all text content parts.
func (r *Response) Text() string {
	return r.Message.Text()
}

// ToolCalls returns all tool call content parts from the response message.
func (r *Response) ToolCalls() []ToolCallData {
	var calls []ToolCallData
	for _, p := range r.Message.Content {
		if p.Kind == ContentToolCall && p.ToolCall != nil {
			calls = append(calls, *p.ToolCall)
		}
	}
	return calls
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add llm/types.go llm/types_test.go
git commit -m "feat: add core types — Role, ContentPart, Message, Request, Response"
```

---

### Task 3: Error Types

**Files:**
- Create: `llm/errors.go`
- Create: `llm/errors_test.go`

**Step 1: Write tests for Error type**

Create `llm/errors_test.go`:

```go
package llm

import (
	"errors"
	"fmt"
	"testing"
)

func TestErrorImplementsError(t *testing.T) {
	e := &Error{
		Kind:     ErrConfig,
		Provider: "anthropic",
		Message:  "no adapter registered",
	}
	want := "llm [config] anthropic: no adapter registered"
	if got := e.Error(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestErrorUnwrap(t *testing.T) {
	cause := fmt.Errorf("underlying issue")
	e := &Error{
		Kind:    ErrServer,
		Message: "bedrock failed",
		Cause:   cause,
	}
	if !errors.Is(e, cause) {
		t.Error("Unwrap should return the cause")
	}
}

func TestErrorKindString(t *testing.T) {
	tests := []struct {
		kind ErrorKind
		want string
	}{
		{ErrConfig, "config"},
		{ErrAdapter, "adapter"},
		{ErrAuthentication, "authentication"},
		{ErrNotFound, "not_found"},
		{ErrInvalidRequest, "invalid_request"},
		{ErrRateLimit, "rate_limit"},
		{ErrServer, "server"},
		{ErrContextLength, "context_length"},
		{ErrContentFilter, "content_filter"},
	}
	for _, tt := range tests {
		if got := tt.kind.String(); got != tt.want {
			t.Errorf("ErrorKind(%d).String() = %q, want %q", tt.kind, got, tt.want)
		}
	}
}

func TestErrorWithoutProvider(t *testing.T) {
	e := &Error{Kind: ErrConfig, Message: "no default provider"}
	want := "llm [config]: no default provider"
	if got := e.Error(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestError`
Expected: FAIL — Error type not defined

**Step 3: Implement errors.go**

Create `llm/errors.go`:

```go
package llm

import "fmt"

// ErrorKind classifies LLM errors.
type ErrorKind int

const (
	ErrConfig        ErrorKind = iota // misconfiguration
	ErrAdapter                        // marshal/unmarshal failure in adapter
	ErrAuthentication                 // 401/403
	ErrNotFound                       // 404
	ErrInvalidRequest                 // 400
	ErrRateLimit                      // 429
	ErrServer                         // 500+
	ErrContextLength                  // input too large
	ErrContentFilter                  // blocked by safety guardrails
)

var errorKindNames = [...]string{
	ErrConfig:         "config",
	ErrAdapter:        "adapter",
	ErrAuthentication: "authentication",
	ErrNotFound:       "not_found",
	ErrInvalidRequest: "invalid_request",
	ErrRateLimit:      "rate_limit",
	ErrServer:         "server",
	ErrContextLength:  "context_length",
	ErrContentFilter:  "content_filter",
}

func (k ErrorKind) String() string {
	if int(k) < len(errorKindNames) {
		return errorKindNames[k]
	}
	return fmt.Sprintf("unknown(%d)", k)
}

// Error is the library's error type.
type Error struct {
	Kind     ErrorKind
	Provider string
	Message  string
	Cause    error  // underlying error
	Raw      []byte // raw response body if available
}

func (e *Error) Error() string {
	if e.Provider != "" {
		return fmt.Sprintf("llm [%s] %s: %s", e.Kind, e.Provider, e.Message)
	}
	return fmt.Sprintf("llm [%s]: %s", e.Kind, e.Message)
}

func (e *Error) Unwrap() error {
	return e.Cause
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestError`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm/errors.go llm/errors_test.go
git commit -m "feat: add Error type with ErrorKind classification"
```

---

### Task 4: Adapter Interface

**Files:**
- Create: `llm/adapter.go`

**Step 1: Create the adapter interface file**

Create `llm/adapter.go`:

```go
package llm

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Adapter translates between unified types and a provider's native format.
type Adapter interface {
	// Provider returns the provider name (e.g., "anthropic", "openai").
	Provider() string

	// BuildInvokeInput translates a unified Request into Bedrock InvokeModel parameters.
	BuildInvokeInput(req *Request) (*InvokeInput, error)

	// ParseResponse translates a raw Bedrock response into a unified Response.
	ParseResponse(body []byte, req *Request) (*Response, error)
}

// InvokeInput carries the parameters for a Bedrock InvokeModel call.
type InvokeInput struct {
	ModelID     string // Bedrock model ID
	Body        []byte // serialized JSON in the provider's native format
	ContentType string // e.g., "application/json"
	Accept      string // e.g., "application/json"
}

// BedrockInvoker abstracts the Bedrock InvokeModel call for testing.
type BedrockInvoker interface {
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}
```

**Step 2: Verify the module compiles**

Run: `cd /home/sprite/unified-llm && go build ./...`
Expected: Clean build

**Step 3: Commit**

```bash
git add llm/adapter.go
git commit -m "feat: add Adapter interface and BedrockInvoker abstraction"
```

---

### Task 5: Anthropic Adapter — Request Building (BuildInvokeInput)

**Files:**
- Create: `llm/anthropic.go`
- Create: `llm/anthropic_test.go`
- Create: `llm/testdata/anthropic/` (golden files)

This task implements `AnthropicAdapter.BuildInvokeInput()` which translates unified Requests into Anthropic Messages API JSON. Key behaviors: system prompt extraction, user/assistant alternation enforcement, tool definition translation, tool choice mapping, cache_control injection, thinking block preservation.

**Step 1: Write golden file tests for request building**

Create golden test input files. Each golden file is the expected JSON output for a given Request.

Create `llm/testdata/anthropic/request_simple_text.json`:
```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 4096,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Hello, Claude"
        }
      ]
    }
  ]
}
```

Create `llm/testdata/anthropic/request_with_system.json`:
```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 4096,
  "system": [
    {
      "type": "text",
      "text": "You are a helpful assistant.",
      "cache_control": {
        "type": "ephemeral"
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Hello"
        }
      ]
    }
  ]
}
```

Create `llm/testdata/anthropic/request_with_tools.json`:
```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 4096,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is the weather in SF?"
        }
      ]
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string"
          }
        },
        "required": [
          "location"
        ]
      },
      "cache_control": {
        "type": "ephemeral"
      }
    }
  ]
}
```

Create `llm/testdata/anthropic/request_tool_result.json`:
```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 4096,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is the weather?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "call-1",
          "name": "get_weather",
          "input": {
            "location": "SF"
          }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "call-1",
          "content": "72°F and sunny"
        }
      ]
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string"
          }
        },
        "required": [
          "location"
        ]
      },
      "cache_control": {
        "type": "ephemeral"
      }
    }
  ]
}
```

Create `llm/testdata/anthropic/request_tool_choice_required.json`:
```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 4096,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Do something"
        }
      ]
    }
  ],
  "tools": [
    {
      "name": "my_tool",
      "description": "A tool",
      "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
      },
      "cache_control": {
        "type": "ephemeral"
      }
    }
  ],
  "tool_choice": {
    "type": "any"
  }
}
```

Create `llm/testdata/anthropic/request_tool_choice_named.json`:
```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 4096,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Do something"
        }
      ]
    }
  ],
  "tools": [
    {
      "name": "my_tool",
      "description": "A tool",
      "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
      },
      "cache_control": {
        "type": "ephemeral"
      }
    }
  ],
  "tool_choice": {
    "type": "tool",
    "name": "my_tool"
  }
}
```

Create `llm/testdata/anthropic/request_with_thinking.json`:
```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 4096,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "First question"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "thinking": "Let me think...",
          "signature": "sig123"
        },
        {
          "type": "text",
          "text": "My answer"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Follow up"
        }
      ]
    }
  ]
}
```

Create `llm/testdata/anthropic/request_with_temperature.json`:
```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 2048,
  "temperature": 0.7,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Be creative"
        }
      ]
    }
  ]
}
```

Now create the test file `llm/anthropic_test.go`:

```go
package llm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func loadGolden(t *testing.T, name string) []byte {
	t.Helper()
	path := filepath.Join("testdata", name)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read golden file %s: %v", path, err)
	}
	return data
}

func assertJSONEqual(t *testing.T, got, want []byte) {
	t.Helper()
	var gotVal, wantVal any
	if err := json.Unmarshal(got, &gotVal); err != nil {
		t.Fatalf("failed to parse got JSON: %v\nraw: %s", err, got)
	}
	if err := json.Unmarshal(want, &wantVal); err != nil {
		t.Fatalf("failed to parse want JSON: %v\nraw: %s", err, want)
	}
	gotNorm, _ := json.MarshalIndent(gotVal, "", "  ")
	wantNorm, _ := json.MarshalIndent(wantVal, "", "  ")
	if string(gotNorm) != string(wantNorm) {
		t.Errorf("JSON mismatch.\ngot:\n%s\nwant:\n%s", gotNorm, wantNorm)
	}
}

func TestAnthropicBuildInvokeInput_SimpleText(t *testing.T) {
	a := NewAnthropicAdapter()
	req := &Request{
		Model:    "anthropic.claude-sonnet-4-5-20250514",
		Messages: []Message{UserMessage("Hello, Claude")},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	if input.ModelID != "anthropic.claude-sonnet-4-5-20250514" {
		t.Errorf("model = %q", input.ModelID)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "anthropic/request_simple_text.json"))
}

func TestAnthropicBuildInvokeInput_WithSystem(t *testing.T) {
	a := NewAnthropicAdapter()
	req := &Request{
		Model: "anthropic.claude-sonnet-4-5-20250514",
		Messages: []Message{
			SystemMessage("You are a helpful assistant."),
			UserMessage("Hello"),
		},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "anthropic/request_with_system.json"))
}

func TestAnthropicBuildInvokeInput_WithTools(t *testing.T) {
	a := NewAnthropicAdapter()
	req := &Request{
		Model:    "anthropic.claude-sonnet-4-5-20250514",
		Messages: []Message{UserMessage("What is the weather in SF?")},
		Tools: []ToolDefinition{{
			Name:        "get_weather",
			Description: "Get the current weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		}},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "anthropic/request_with_tools.json"))
}

func TestAnthropicBuildInvokeInput_ToolResult(t *testing.T) {
	a := NewAnthropicAdapter()
	req := &Request{
		Model: "anthropic.claude-sonnet-4-5-20250514",
		Messages: []Message{
			UserMessage("What is the weather?"),
			{
				Role: RoleAssistant,
				Content: []ContentPart{{
					Kind:     ContentToolCall,
					ToolCall: &ToolCallData{ID: "call-1", Name: "get_weather", Arguments: json.RawMessage(`{"location":"SF"}`)},
				}},
			},
			ToolResultMessage("call-1", "72°F and sunny", false),
		},
		Tools: []ToolDefinition{{
			Name:        "get_weather",
			Description: "Get the current weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		}},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "anthropic/request_tool_result.json"))
}

func TestAnthropicBuildInvokeInput_ToolChoiceRequired(t *testing.T) {
	a := NewAnthropicAdapter()
	req := &Request{
		Model:    "anthropic.claude-sonnet-4-5-20250514",
		Messages: []Message{UserMessage("Do something")},
		Tools: []ToolDefinition{{
			Name:        "my_tool",
			Description: "A tool",
			Parameters:  json.RawMessage(`{"type":"object","properties":{},"required":[]}`),
		}},
		ToolChoice: &ToolChoice{Mode: ToolChoiceRequired},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "anthropic/request_tool_choice_required.json"))
}

func TestAnthropicBuildInvokeInput_ToolChoiceNamed(t *testing.T) {
	a := NewAnthropicAdapter()
	req := &Request{
		Model:    "anthropic.claude-sonnet-4-5-20250514",
		Messages: []Message{UserMessage("Do something")},
		Tools: []ToolDefinition{{
			Name:        "my_tool",
			Description: "A tool",
			Parameters:  json.RawMessage(`{"type":"object","properties":{},"required":[]}`),
		}},
		ToolChoice: &ToolChoice{Mode: ToolChoiceNamed, ToolName: "my_tool"},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "anthropic/request_tool_choice_named.json"))
}

func TestAnthropicBuildInvokeInput_WithThinking(t *testing.T) {
	a := NewAnthropicAdapter()
	req := &Request{
		Model: "anthropic.claude-sonnet-4-5-20250514",
		Messages: []Message{
			UserMessage("First question"),
			{
				Role: RoleAssistant,
				Content: []ContentPart{
					{Kind: ContentThinking, Thinking: &ThinkingData{Text: "Let me think...", Signature: "sig123"}},
					{Kind: ContentText, Text: "My answer"},
				},
			},
			UserMessage("Follow up"),
		},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "anthropic/request_with_thinking.json"))
}

func TestAnthropicBuildInvokeInput_WithTemperature(t *testing.T) {
	a := NewAnthropicAdapter()
	temp := 0.7
	maxTok := 2048
	req := &Request{
		Model:       "anthropic.claude-sonnet-4-5-20250514",
		Messages:    []Message{UserMessage("Be creative")},
		Temperature: &temp,
		MaxTokens:   &maxTok,
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "anthropic/request_with_temperature.json"))
}

func TestAnthropicProvider(t *testing.T) {
	a := NewAnthropicAdapter()
	if got := a.Provider(); got != "anthropic" {
		t.Errorf("got %q, want %q", got, "anthropic")
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestAnthropic`
Expected: FAIL — NewAnthropicAdapter not defined

**Step 3: Implement AnthropicAdapter.BuildInvokeInput**

Create `llm/anthropic.go`:

```go
package llm

import "encoding/json"

// AnthropicAdapter translates between unified types and the Anthropic Messages API format.
type AnthropicAdapter struct{}

// NewAnthropicAdapter creates a new AnthropicAdapter.
func NewAnthropicAdapter() *AnthropicAdapter {
	return &AnthropicAdapter{}
}

func (a *AnthropicAdapter) Provider() string { return "anthropic" }

// --- Anthropic request types ---

type anthropicRequest struct {
	AnthropicVersion string              `json:"anthropic_version"`
	MaxTokens        int                 `json:"max_tokens"`
	System           []anthropicContent  `json:"system,omitempty"`
	Messages         []anthropicMessage  `json:"messages"`
	Tools            []anthropicTool     `json:"tools,omitempty"`
	ToolChoice       any                 `json:"tool_choice,omitempty"`
	Temperature      *float64            `json:"temperature,omitempty"`
	TopP             *float64            `json:"top_p,omitempty"`
	StopSequences    []string            `json:"stop_sequences,omitempty"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type         string            `json:"type"`
	Text         string            `json:"text,omitempty"`
	ID           string            `json:"id,omitempty"`
	Name         string            `json:"name,omitempty"`
	Input        json.RawMessage   `json:"input,omitempty"`
	ToolUseID    string            `json:"tool_use_id,omitempty"`
	Content      string            `json:"content,omitempty"`
	IsError      *bool             `json:"is_error,omitempty"`
	Thinking     string            `json:"thinking,omitempty"`
	Signature    string            `json:"signature,omitempty"`
	CacheControl *cacheControl     `json:"cache_control,omitempty"`
}

type cacheControl struct {
	Type string `json:"type"`
}

type anthropicTool struct {
	Name         string          `json:"name"`
	Description  string          `json:"description"`
	InputSchema  json.RawMessage `json:"input_schema"`
	CacheControl *cacheControl   `json:"cache_control,omitempty"`
}

func (a *AnthropicAdapter) BuildInvokeInput(req *Request) (*InvokeInput, error) {
	ar := anthropicRequest{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        4096,
	}

	if req.MaxTokens != nil {
		ar.MaxTokens = *req.MaxTokens
	}
	ar.Temperature = req.Temperature
	ar.TopP = req.TopP
	if len(req.StopSequences) > 0 {
		ar.StopSequences = req.StopSequences
	}

	// Extract system messages
	var nonSystem []Message
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			for _, p := range m.Content {
				if p.Kind == ContentText {
					ar.System = append(ar.System, anthropicContent{Type: "text", Text: p.Text})
				}
			}
		} else {
			nonSystem = append(nonSystem, m)
		}
	}

	// Auto-inject cache_control on last system block
	if len(ar.System) > 0 {
		ar.System[len(ar.System)-1].CacheControl = &cacheControl{Type: "ephemeral"}
	}

	// Translate messages
	for _, m := range nonSystem {
		am := a.translateMessage(m)
		// Enforce strict user/assistant alternation: merge consecutive same-role messages
		if len(ar.Messages) > 0 && ar.Messages[len(ar.Messages)-1].Role == am.Role {
			ar.Messages[len(ar.Messages)-1].Content = append(ar.Messages[len(ar.Messages)-1].Content, am.Content...)
		} else {
			ar.Messages = append(ar.Messages, am)
		}
	}

	// Translate tools
	if len(req.Tools) > 0 {
		for _, td := range req.Tools {
			ar.Tools = append(ar.Tools, anthropicTool{
				Name:        td.Name,
				Description: td.Description,
				InputSchema: td.Parameters,
			})
		}
		// Auto-inject cache_control on last tool
		ar.Tools[len(ar.Tools)-1].CacheControl = &cacheControl{Type: "ephemeral"}
	}

	// Translate tool choice
	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case ToolChoiceAuto:
			ar.ToolChoice = map[string]string{"type": "auto"}
		case ToolChoiceNone:
			// Omit tools entirely
			ar.Tools = nil
			ar.ToolChoice = nil
		case ToolChoiceRequired:
			ar.ToolChoice = map[string]string{"type": "any"}
		case ToolChoiceNamed:
			ar.ToolChoice = map[string]string{"type": "tool", "name": req.ToolChoice.ToolName}
		}
	}

	// Merge provider options
	if opts, ok := req.ProviderOptions["anthropic"]; ok {
		if m, ok := opts.(map[string]any); ok {
			_ = m // provider options are merged at JSON level below
		}
	}

	body, err := json.Marshal(ar)
	if err != nil {
		return nil, &Error{Kind: ErrAdapter, Provider: "anthropic", Message: "failed to marshal request", Cause: err}
	}

	return &InvokeInput{
		ModelID:     req.Model,
		Body:        body,
		ContentType: "application/json",
		Accept:      "application/json",
	}, nil
}

func (a *AnthropicAdapter) translateMessage(m Message) anthropicMessage {
	am := anthropicMessage{}

	switch m.Role {
	case RoleUser:
		am.Role = "user"
	case RoleAssistant:
		am.Role = "assistant"
	case RoleTool:
		am.Role = "user" // Tool results are sent as user messages in Anthropic format
	}

	for _, p := range m.Content {
		switch p.Kind {
		case ContentText:
			am.Content = append(am.Content, anthropicContent{Type: "text", Text: p.Text})
		case ContentToolCall:
			am.Content = append(am.Content, anthropicContent{
				Type:  "tool_use",
				ID:    p.ToolCall.ID,
				Name:  p.ToolCall.Name,
				Input: p.ToolCall.Arguments,
			})
		case ContentToolResult:
			am.Content = append(am.Content, anthropicContent{
				Type:      "tool_result",
				ToolUseID: p.ToolResult.ToolCallID,
				Content:   p.ToolResult.Content,
			})
		case ContentThinking:
			am.Content = append(am.Content, anthropicContent{
				Type:      "thinking",
				Thinking:  p.Thinking.Text,
				Signature: p.Thinking.Signature,
			})
		case ContentImage:
			// Image support can be added later
		}
	}

	return am
}

// ParseResponse is implemented in the next task.
func (a *AnthropicAdapter) ParseResponse(body []byte, req *Request) (*Response, error) {
	return nil, &Error{Kind: ErrAdapter, Provider: "anthropic", Message: "not implemented"}
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestAnthropic`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm/anthropic.go llm/anthropic_test.go llm/testdata/
git commit -m "feat: add AnthropicAdapter.BuildInvokeInput with golden file tests"
```

---

### Task 6: Anthropic Adapter — Response Parsing (ParseResponse)

**Files:**
- Modify: `llm/anthropic.go`
- Modify: `llm/anthropic_test.go`
- Create: `llm/testdata/anthropic/response_*.json` (golden files)

**Step 1: Write golden file tests for response parsing**

Create `llm/testdata/anthropic/response_simple_text.json`:
```json
{
  "id": "msg_123",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-5-20250514",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I help you?"
    }
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 25,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 10
  }
}
```

Create `llm/testdata/anthropic/response_tool_use.json`:
```json
{
  "id": "msg_456",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-5-20250514",
  "content": [
    {
      "type": "text",
      "text": "Let me check the weather."
    },
    {
      "type": "tool_use",
      "id": "toolu_abc",
      "name": "get_weather",
      "input": {
        "location": "San Francisco"
      }
    }
  ],
  "stop_reason": "tool_use",
  "usage": {
    "input_tokens": 50,
    "output_tokens": 30
  }
}
```

Create `llm/testdata/anthropic/response_with_thinking.json`:
```json
{
  "id": "msg_789",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-5-20250514",
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me reason about this...",
      "signature": "sig_abc123"
    },
    {
      "type": "text",
      "text": "Here is my answer."
    }
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 100,
    "output_tokens": 50
  }
}
```

Add to `llm/anthropic_test.go`:

```go
func TestAnthropicParseResponse_SimpleText(t *testing.T) {
	a := NewAnthropicAdapter()
	body := loadGolden(t, "anthropic/response_simple_text.json")
	resp, err := a.ParseResponse(body, &Request{Model: "anthropic.claude-sonnet-4-5-20250514"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "msg_123" {
		t.Errorf("ID = %q", resp.ID)
	}
	if resp.Model != "claude-sonnet-4-5-20250514" {
		t.Errorf("Model = %q", resp.Model)
	}
	if resp.Provider != "anthropic" {
		t.Errorf("Provider = %q", resp.Provider)
	}
	if resp.Text() != "Hello! How can I help you?" {
		t.Errorf("Text = %q", resp.Text())
	}
	if resp.FinishReason.Reason != "stop" {
		t.Errorf("FinishReason.Reason = %q", resp.FinishReason.Reason)
	}
	if resp.FinishReason.Raw != "end_turn" {
		t.Errorf("FinishReason.Raw = %q", resp.FinishReason.Raw)
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 25 {
		t.Errorf("OutputTokens = %d", resp.Usage.OutputTokens)
	}
	if resp.Usage.CacheReadTokens != 5 {
		t.Errorf("CacheReadTokens = %d", resp.Usage.CacheReadTokens)
	}
	if resp.Usage.CacheWriteTokens != 10 {
		t.Errorf("CacheWriteTokens = %d", resp.Usage.CacheWriteTokens)
	}
}

func TestAnthropicParseResponse_ToolUse(t *testing.T) {
	a := NewAnthropicAdapter()
	body := loadGolden(t, "anthropic/response_tool_use.json")
	resp, err := a.ParseResponse(body, &Request{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason.Reason != "tool_calls" {
		t.Errorf("FinishReason.Reason = %q", resp.FinishReason.Reason)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].ID != "toolu_abc" {
		t.Errorf("ID = %q", calls[0].ID)
	}
	if calls[0].Name != "get_weather" {
		t.Errorf("Name = %q", calls[0].Name)
	}
	var args map[string]string
	json.Unmarshal(calls[0].Arguments, &args)
	if args["location"] != "San Francisco" {
		t.Errorf("location = %q", args["location"])
	}
	if resp.Text() != "Let me check the weather." {
		t.Errorf("Text = %q", resp.Text())
	}
}

func TestAnthropicParseResponse_WithThinking(t *testing.T) {
	a := NewAnthropicAdapter()
	body := loadGolden(t, "anthropic/response_with_thinking.json")
	resp, err := a.ParseResponse(body, &Request{})
	if err != nil {
		t.Fatal(err)
	}
	// Should have 2 content parts: thinking + text
	if len(resp.Message.Content) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(resp.Message.Content))
	}
	thinking := resp.Message.Content[0]
	if thinking.Kind != ContentThinking {
		t.Errorf("first part kind = %q", thinking.Kind)
	}
	if thinking.Thinking.Text != "Let me reason about this..." {
		t.Errorf("thinking text = %q", thinking.Thinking.Text)
	}
	if thinking.Thinking.Signature != "sig_abc123" {
		t.Errorf("signature = %q", thinking.Thinking.Signature)
	}
	if resp.Text() != "Here is my answer." {
		t.Errorf("Text = %q", resp.Text())
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run 'TestAnthropicParseResponse'`
Expected: FAIL — ParseResponse returns "not implemented"

**Step 3: Implement ParseResponse**

Replace the stub `ParseResponse` in `llm/anthropic.go`:

```go
// --- Anthropic response types ---

type anthropicResponse struct {
	ID         string             `json:"id"`
	Type       string             `json:"type"`
	Role       string             `json:"role"`
	Model      string             `json:"model"`
	Content    []anthropicRespContent `json:"content"`
	StopReason string             `json:"stop_reason"`
	Usage      anthropicUsage     `json:"usage"`
}

type anthropicRespContent struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
}

type anthropicUsage struct {
	InputTokens          int `json:"input_tokens"`
	OutputTokens         int `json:"output_tokens"`
	CacheReadInputTokens int `json:"cache_read_input_tokens"`
	CacheCreationTokens  int `json:"cache_creation_input_tokens"`
}

func (a *AnthropicAdapter) ParseResponse(body []byte, req *Request) (*Response, error) {
	var ar anthropicResponse
	if err := json.Unmarshal(body, &ar); err != nil {
		return nil, &Error{Kind: ErrAdapter, Provider: "anthropic", Message: "failed to unmarshal response", Cause: err, Raw: body}
	}

	msg := Message{Role: RoleAssistant}
	for _, c := range ar.Content {
		switch c.Type {
		case "text":
			msg.Content = append(msg.Content, ContentPart{Kind: ContentText, Text: c.Text})
		case "tool_use":
			msg.Content = append(msg.Content, ContentPart{
				Kind: ContentToolCall,
				ToolCall: &ToolCallData{
					ID:        c.ID,
					Name:      c.Name,
					Arguments: c.Input,
				},
			})
		case "thinking":
			msg.Content = append(msg.Content, ContentPart{
				Kind: ContentThinking,
				Thinking: &ThinkingData{
					Text:      c.Thinking,
					Signature: c.Signature,
				},
			})
		}
	}

	return &Response{
		ID:       ar.ID,
		Model:    ar.Model,
		Provider: "anthropic",
		Message:  msg,
		FinishReason: mapAnthropicFinishReason(ar.StopReason),
		Usage: Usage{
			InputTokens:      ar.Usage.InputTokens,
			OutputTokens:     ar.Usage.OutputTokens,
			CacheReadTokens:  ar.Usage.CacheReadInputTokens,
			CacheWriteTokens: ar.Usage.CacheCreationTokens,
		},
		Raw: body,
	}, nil
}

func mapAnthropicFinishReason(raw string) FinishReason {
	reason := "stop"
	switch raw {
	case "end_turn", "stop_sequence":
		reason = "stop"
	case "max_tokens":
		reason = "length"
	case "tool_use":
		reason = "tool_calls"
	default:
		reason = raw
	}
	return FinishReason{Reason: reason, Raw: raw}
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestAnthropic`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm/anthropic.go llm/anthropic_test.go llm/testdata/anthropic/response_*.json
git commit -m "feat: add AnthropicAdapter.ParseResponse with golden file tests"
```

---

### Task 7: OpenAI Adapter — Request Building (BuildInvokeInput)

**Files:**
- Create: `llm/openai.go`
- Create: `llm/openai_test.go`
- Create: `llm/testdata/openai/` (golden files)

**Step 1: Write golden file tests for request building**

Create `llm/testdata/openai/request_simple_text.json`:
```json
{
  "model": "us.amazon.nova-pro-v1:0",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ]
}
```

Create `llm/testdata/openai/request_with_system.json`:
```json
{
  "model": "us.amazon.nova-pro-v1:0",
  "messages": [
    {
      "role": "system",
      "content": "You are helpful."
    },
    {
      "role": "user",
      "content": "Hello"
    }
  ]
}
```

Create `llm/testdata/openai/request_with_tools.json`:
```json
{
  "model": "us.amazon.nova-pro-v1:0",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string"
            }
          },
          "required": [
            "location"
          ]
        }
      }
    }
  ]
}
```

Create `llm/testdata/openai/request_tool_result.json`:
```json
{
  "model": "us.amazon.nova-pro-v1:0",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather?"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call-1",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"SF\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call-1",
      "content": "72°F and sunny"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string"
            }
          },
          "required": [
            "location"
          ]
        }
      }
    }
  ]
}
```

Create `llm/testdata/openai/request_tool_choice_required.json`:
```json
{
  "model": "us.amazon.nova-pro-v1:0",
  "messages": [
    {
      "role": "user",
      "content": "Do something"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "my_tool",
        "description": "A tool",
        "parameters": {
          "type": "object",
          "properties": {},
          "required": []
        }
      }
    }
  ],
  "tool_choice": "required"
}
```

Create `llm/testdata/openai/request_tool_choice_named.json`:
```json
{
  "model": "us.amazon.nova-pro-v1:0",
  "messages": [
    {
      "role": "user",
      "content": "Do something"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "my_tool",
        "description": "A tool",
        "parameters": {
          "type": "object",
          "properties": {},
          "required": []
        }
      }
    }
  ],
  "tool_choice": {
    "type": "function",
    "function": {
      "name": "my_tool"
    }
  }
}
```

Create `llm/testdata/openai/request_with_temperature.json`:
```json
{
  "model": "us.amazon.nova-pro-v1:0",
  "messages": [
    {
      "role": "user",
      "content": "Be creative"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 2048
}
```

Create `llm/testdata/openai/request_with_reasoning.json`:
```json
{
  "model": "us.amazon.nova-pro-v1:0",
  "messages": [
    {
      "role": "user",
      "content": "Think hard"
    }
  ],
  "reasoning_effort": "high"
}
```

Create `llm/openai_test.go`:

```go
package llm

import (
	"encoding/json"
	"testing"
)

func TestOpenAIBuildInvokeInput_SimpleText(t *testing.T) {
	a := NewOpenAIAdapter()
	req := &Request{
		Model:    "us.amazon.nova-pro-v1:0",
		Messages: []Message{UserMessage("Hello")},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	if input.ModelID != "us.amazon.nova-pro-v1:0" {
		t.Errorf("model = %q", input.ModelID)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "openai/request_simple_text.json"))
}

func TestOpenAIBuildInvokeInput_WithSystem(t *testing.T) {
	a := NewOpenAIAdapter()
	req := &Request{
		Model: "us.amazon.nova-pro-v1:0",
		Messages: []Message{
			SystemMessage("You are helpful."),
			UserMessage("Hello"),
		},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "openai/request_with_system.json"))
}

func TestOpenAIBuildInvokeInput_WithTools(t *testing.T) {
	a := NewOpenAIAdapter()
	req := &Request{
		Model:    "us.amazon.nova-pro-v1:0",
		Messages: []Message{UserMessage("What is the weather?")},
		Tools: []ToolDefinition{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		}},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "openai/request_with_tools.json"))
}

func TestOpenAIBuildInvokeInput_ToolResult(t *testing.T) {
	a := NewOpenAIAdapter()
	req := &Request{
		Model: "us.amazon.nova-pro-v1:0",
		Messages: []Message{
			UserMessage("What is the weather?"),
			{
				Role: RoleAssistant,
				Content: []ContentPart{{
					Kind:     ContentToolCall,
					ToolCall: &ToolCallData{ID: "call-1", Name: "get_weather", Arguments: json.RawMessage(`{"location":"SF"}`)},
				}},
			},
			ToolResultMessage("call-1", "72°F and sunny", false),
		},
		Tools: []ToolDefinition{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		}},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "openai/request_tool_result.json"))
}

func TestOpenAIBuildInvokeInput_ToolChoiceRequired(t *testing.T) {
	a := NewOpenAIAdapter()
	req := &Request{
		Model:    "us.amazon.nova-pro-v1:0",
		Messages: []Message{UserMessage("Do something")},
		Tools: []ToolDefinition{{
			Name:        "my_tool",
			Description: "A tool",
			Parameters:  json.RawMessage(`{"type":"object","properties":{},"required":[]}`),
		}},
		ToolChoice: &ToolChoice{Mode: ToolChoiceRequired},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "openai/request_tool_choice_required.json"))
}

func TestOpenAIBuildInvokeInput_ToolChoiceNamed(t *testing.T) {
	a := NewOpenAIAdapter()
	req := &Request{
		Model:    "us.amazon.nova-pro-v1:0",
		Messages: []Message{UserMessage("Do something")},
		Tools: []ToolDefinition{{
			Name:        "my_tool",
			Description: "A tool",
			Parameters:  json.RawMessage(`{"type":"object","properties":{},"required":[]}`),
		}},
		ToolChoice: &ToolChoice{Mode: ToolChoiceNamed, ToolName: "my_tool"},
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "openai/request_tool_choice_named.json"))
}

func TestOpenAIBuildInvokeInput_WithTemperature(t *testing.T) {
	a := NewOpenAIAdapter()
	temp := 0.7
	maxTok := 2048
	req := &Request{
		Model:       "us.amazon.nova-pro-v1:0",
		Messages:    []Message{UserMessage("Be creative")},
		Temperature: &temp,
		MaxTokens:   &maxTok,
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "openai/request_with_temperature.json"))
}

func TestOpenAIBuildInvokeInput_WithReasoning(t *testing.T) {
	a := NewOpenAIAdapter()
	req := &Request{
		Model:           "us.amazon.nova-pro-v1:0",
		Messages:        []Message{UserMessage("Think hard")},
		ReasoningEffort: "high",
	}
	input, err := a.BuildInvokeInput(req)
	if err != nil {
		t.Fatal(err)
	}
	assertJSONEqual(t, input.Body, loadGolden(t, "openai/request_with_reasoning.json"))
}

func TestOpenAIProvider(t *testing.T) {
	a := NewOpenAIAdapter()
	if got := a.Provider(); got != "openai" {
		t.Errorf("got %q, want %q", got, "openai")
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestOpenAI`
Expected: FAIL — NewOpenAIAdapter not defined

**Step 3: Implement OpenAIAdapter.BuildInvokeInput**

Create `llm/openai.go`:

```go
package llm

import "encoding/json"

// OpenAIAdapter translates between unified types and the OpenAI Chat Completions format.
type OpenAIAdapter struct{}

// NewOpenAIAdapter creates a new OpenAIAdapter.
func NewOpenAIAdapter() *OpenAIAdapter {
	return &OpenAIAdapter{}
}

func (a *OpenAIAdapter) Provider() string { return "openai" }

// --- OpenAI request types ---

type openaiRequest struct {
	Model           string           `json:"model"`
	Messages        []openaiMessage  `json:"messages"`
	Tools           []openaiTool     `json:"tools,omitempty"`
	ToolChoice      any              `json:"tool_choice,omitempty"`
	Temperature     *float64         `json:"temperature,omitempty"`
	TopP            *float64         `json:"top_p,omitempty"`
	MaxTokens       *int             `json:"max_tokens,omitempty"`
	Stop            []string         `json:"stop,omitempty"`
	ReasoningEffort string           `json:"reasoning_effort,omitempty"`
}

type openaiMessage struct {
	Role       string             `json:"role"`
	Content    any                `json:"content"`               // string or null
	ToolCalls  []openaiToolCall   `json:"tool_calls,omitempty"`
	ToolCallID string             `json:"tool_call_id,omitempty"`
}

type openaiToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openaiToolFunction `json:"function"`
}

type openaiToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openaiTool struct {
	Type     string             `json:"type"`
	Function openaiToolDef      `json:"function"`
}

type openaiToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

func (a *OpenAIAdapter) BuildInvokeInput(req *Request) (*InvokeInput, error) {
	or := openaiRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
	}
	if len(req.StopSequences) > 0 {
		or.Stop = req.StopSequences
	}
	if req.ReasoningEffort != "" {
		or.ReasoningEffort = req.ReasoningEffort
	}

	// Translate messages
	for _, m := range req.Messages {
		om := a.translateMessage(m)
		or.Messages = append(or.Messages, om)
	}

	// Translate tools
	if len(req.Tools) > 0 {
		for _, td := range req.Tools {
			or.Tools = append(or.Tools, openaiTool{
				Type: "function",
				Function: openaiToolDef{
					Name:        td.Name,
					Description: td.Description,
					Parameters:  td.Parameters,
				},
			})
		}
	}

	// Translate tool choice
	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case ToolChoiceAuto:
			or.ToolChoice = "auto"
		case ToolChoiceNone:
			or.ToolChoice = "none"
		case ToolChoiceRequired:
			or.ToolChoice = "required"
		case ToolChoiceNamed:
			or.ToolChoice = map[string]any{
				"type":     "function",
				"function": map[string]string{"name": req.ToolChoice.ToolName},
			}
		}
	}

	body, err := json.Marshal(or)
	if err != nil {
		return nil, &Error{Kind: ErrAdapter, Provider: "openai", Message: "failed to marshal request", Cause: err}
	}

	return &InvokeInput{
		ModelID:     req.Model,
		Body:        body,
		ContentType: "application/json",
		Accept:      "application/json",
	}, nil
}

func (a *OpenAIAdapter) translateMessage(m Message) openaiMessage {
	om := openaiMessage{}

	switch m.Role {
	case RoleSystem:
		om.Role = "system"
		om.Content = m.Text()
	case RoleUser:
		om.Role = "user"
		om.Content = m.Text()
	case RoleAssistant:
		om.Role = "assistant"
		// Check for tool calls
		var toolCalls []openaiToolCall
		text := m.Text()
		for _, p := range m.Content {
			if p.Kind == ContentToolCall && p.ToolCall != nil {
				toolCalls = append(toolCalls, openaiToolCall{
					ID:   p.ToolCall.ID,
					Type: "function",
					Function: openaiToolFunction{
						Name:      p.ToolCall.Name,
						Arguments: string(p.ToolCall.Arguments),
					},
				})
			}
		}
		if len(toolCalls) > 0 {
			om.ToolCalls = toolCalls
			// content is null when there are tool calls and no text
			if text == "" {
				om.Content = nil
			} else {
				om.Content = text
			}
		} else {
			om.Content = text
		}
	case RoleTool:
		om.Role = "tool"
		om.ToolCallID = m.ToolCallID
		// Get the content from tool result
		for _, p := range m.Content {
			if p.Kind == ContentToolResult && p.ToolResult != nil {
				om.Content = p.ToolResult.Content
				break
			}
		}
	}

	return om
}

// ParseResponse is implemented in the next task.
func (a *OpenAIAdapter) ParseResponse(body []byte, req *Request) (*Response, error) {
	return nil, &Error{Kind: ErrAdapter, Provider: "openai", Message: "not implemented"}
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestOpenAI`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm/openai.go llm/openai_test.go llm/testdata/openai/
git commit -m "feat: add OpenAIAdapter.BuildInvokeInput with golden file tests"
```

---

### Task 8: OpenAI Adapter — Response Parsing (ParseResponse)

**Files:**
- Modify: `llm/openai.go`
- Modify: `llm/openai_test.go`
- Create: `llm/testdata/openai/response_*.json`

**Step 1: Write golden file tests for response parsing**

Create `llm/testdata/openai/response_simple_text.json`:
```json
{
  "id": "chatcmpl-abc",
  "model": "us.amazon.nova-pro-v1:0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "prompt_tokens_details": {
      "cached_tokens": 5
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  }
}
```

Create `llm/testdata/openai/response_tool_calls.json`:
```json
{
  "id": "chatcmpl-def",
  "model": "us.amazon.nova-pro-v1:0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"SF\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 30
  }
}
```

Create `llm/testdata/openai/response_with_reasoning.json`:
```json
{
  "id": "chatcmpl-ghi",
  "model": "us.amazon.nova-pro-v1:0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The answer is 42."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 100,
    "completion_tokens_details": {
      "reasoning_tokens": 80
    }
  }
}
```

Add to `llm/openai_test.go`:

```go
func TestOpenAIParseResponse_SimpleText(t *testing.T) {
	a := NewOpenAIAdapter()
	body := loadGolden(t, "openai/response_simple_text.json")
	resp, err := a.ParseResponse(body, &Request{Model: "us.amazon.nova-pro-v1:0"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "chatcmpl-abc" {
		t.Errorf("ID = %q", resp.ID)
	}
	if resp.Provider != "openai" {
		t.Errorf("Provider = %q", resp.Provider)
	}
	if resp.Text() != "Hello! How can I help?" {
		t.Errorf("Text = %q", resp.Text())
	}
	if resp.FinishReason.Reason != "stop" {
		t.Errorf("FinishReason.Reason = %q", resp.FinishReason.Reason)
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 20 {
		t.Errorf("OutputTokens = %d", resp.Usage.OutputTokens)
	}
	if resp.Usage.CacheReadTokens != 5 {
		t.Errorf("CacheReadTokens = %d", resp.Usage.CacheReadTokens)
	}
}

func TestOpenAIParseResponse_ToolCalls(t *testing.T) {
	a := NewOpenAIAdapter()
	body := loadGolden(t, "openai/response_tool_calls.json")
	resp, err := a.ParseResponse(body, &Request{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason.Reason != "tool_calls" {
		t.Errorf("FinishReason.Reason = %q", resp.FinishReason.Reason)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].ID != "call_abc" {
		t.Errorf("ID = %q", calls[0].ID)
	}
	if calls[0].Name != "get_weather" {
		t.Errorf("Name = %q", calls[0].Name)
	}
}

func TestOpenAIParseResponse_WithReasoning(t *testing.T) {
	a := NewOpenAIAdapter()
	body := loadGolden(t, "openai/response_with_reasoning.json")
	resp, err := a.ParseResponse(body, &Request{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.ReasoningTokens != 80 {
		t.Errorf("ReasoningTokens = %d", resp.Usage.ReasoningTokens)
	}
	if resp.Usage.OutputTokens != 100 {
		t.Errorf("OutputTokens = %d", resp.Usage.OutputTokens)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run 'TestOpenAIParseResponse'`
Expected: FAIL — ParseResponse returns "not implemented"

**Step 3: Implement ParseResponse**

Replace the stub in `llm/openai.go`:

```go
// --- OpenAI response types ---

type openaiResponse struct {
	ID      string          `json:"id"`
	Model   string          `json:"model"`
	Choices []openaiChoice  `json:"choices"`
	Usage   openaiUsage     `json:"usage"`
}

type openaiChoice struct {
	Index        int              `json:"index"`
	Message      openaiRespMsg    `json:"message"`
	FinishReason string           `json:"finish_reason"`
}

type openaiRespMsg struct {
	Role      string           `json:"role"`
	Content   *string          `json:"content"`
	ToolCalls []openaiToolCall  `json:"tool_calls,omitempty"`
}

type openaiUsage struct {
	PromptTokens     int                    `json:"prompt_tokens"`
	CompletionTokens int                    `json:"completion_tokens"`
	PromptDetails    *openaiPromptDetails   `json:"prompt_tokens_details,omitempty"`
	CompletionDetails *openaiCompletionDetails `json:"completion_tokens_details,omitempty"`
}

type openaiPromptDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type openaiCompletionDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

func (a *OpenAIAdapter) ParseResponse(body []byte, req *Request) (*Response, error) {
	var or openaiResponse
	if err := json.Unmarshal(body, &or); err != nil {
		return nil, &Error{Kind: ErrAdapter, Provider: "openai", Message: "failed to unmarshal response", Cause: err, Raw: body}
	}

	if len(or.Choices) == 0 {
		return nil, &Error{Kind: ErrAdapter, Provider: "openai", Message: "response has no choices", Raw: body}
	}

	choice := or.Choices[0]
	msg := Message{Role: RoleAssistant}

	if choice.Message.Content != nil && *choice.Message.Content != "" {
		msg.Content = append(msg.Content, ContentPart{Kind: ContentText, Text: *choice.Message.Content})
	}

	for _, tc := range choice.Message.ToolCalls {
		msg.Content = append(msg.Content, ContentPart{
			Kind: ContentToolCall,
			ToolCall: &ToolCallData{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			},
		})
	}

	usage := Usage{
		InputTokens:  or.Usage.PromptTokens,
		OutputTokens: or.Usage.CompletionTokens,
	}
	if or.Usage.PromptDetails != nil {
		usage.CacheReadTokens = or.Usage.PromptDetails.CachedTokens
	}
	if or.Usage.CompletionDetails != nil {
		usage.ReasoningTokens = or.Usage.CompletionDetails.ReasoningTokens
	}

	return &Response{
		ID:           or.ID,
		Model:        or.Model,
		Provider:     "openai",
		Message:      msg,
		FinishReason: mapOpenAIFinishReason(choice.FinishReason),
		Usage:        usage,
		Raw:          body,
	}, nil
}

func mapOpenAIFinishReason(raw string) FinishReason {
	reason := raw // OpenAI values mostly match unified values
	switch raw {
	case "stop", "length", "tool_calls", "content_filter":
		// already correct
	default:
		reason = raw
	}
	return FinishReason{Reason: reason, Raw: raw}
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestOpenAI`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm/openai.go llm/openai_test.go llm/testdata/openai/response_*.json
git commit -m "feat: add OpenAIAdapter.ParseResponse with golden file tests"
```

---

### Task 9: Client — Construction, Provider Routing, and Complete()

**Files:**
- Create: `llm/client.go`
- Create: `llm/client_test.go`

**Step 1: Write tests for client construction and Complete()**

Create `llm/client_test.go`:

```go
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// mockInvoker is a test double for BedrockInvoker.
type mockInvoker struct {
	response []byte
	err      error
}

func (m *mockInvoker) InvokeModel(_ context.Context, params *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &bedrockruntime.InvokeModelOutput{
		Body: m.response,
	}, nil
}

func TestClientComplete_RoutesToCorrectAdapter(t *testing.T) {
	anthropicResp := `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"from anthropic"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	openaiResp := `{"id":"chat_1","model":"gpt","choices":[{"index":0,"message":{"role":"assistant","content":"from openai"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`

	tests := []struct {
		name     string
		provider string
		response string
		wantText string
	}{
		{"anthropic", "anthropic", anthropicResp, "from anthropic"},
		{"openai", "openai", openaiResp, "from openai"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(&mockInvoker{response: []byte(tt.response)},
				WithAdapter(NewAnthropicAdapter()),
				WithAdapter(NewOpenAIAdapter()),
			)
			resp, err := client.Complete(context.Background(), &Request{
				Model:    "test-model",
				Provider: tt.provider,
				Messages: []Message{UserMessage("hello")},
			})
			if err != nil {
				t.Fatal(err)
			}
			if resp.Text() != tt.wantText {
				t.Errorf("Text = %q, want %q", resp.Text(), tt.wantText)
			}
		})
	}
}

func TestClientComplete_UsesDefaultProvider(t *testing.T) {
	resp := `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	client := NewClient(&mockInvoker{response: []byte(resp)},
		WithAdapter(NewAnthropicAdapter()),
		WithDefaultProvider("anthropic"),
	)
	result, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{UserMessage("hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text() != "ok" {
		t.Errorf("Text = %q", result.Text())
	}
}

func TestClientComplete_NoProviderReturnsConfigError(t *testing.T) {
	client := NewClient(&mockInvoker{},
		WithAdapter(NewAnthropicAdapter()),
	)
	_, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{UserMessage("hello")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) || llmErr.Kind != ErrConfig {
		t.Errorf("expected ErrConfig, got %v", err)
	}
}

func TestClientComplete_UnknownProviderReturnsConfigError(t *testing.T) {
	client := NewClient(&mockInvoker{},
		WithAdapter(NewAnthropicAdapter()),
		WithDefaultProvider("anthropic"),
	)
	_, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Provider: "gemini",
		Messages: []Message{UserMessage("hello")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) || llmErr.Kind != ErrConfig {
		t.Errorf("expected ErrConfig, got %v", err)
	}
}

func TestClientComplete_MiddlewareExecutionOrder(t *testing.T) {
	var order []string
	mw1 := func(ctx context.Context, req *Request, next CompleteFunc) (*Response, error) {
		order = append(order, "mw1-before")
		resp, err := next(ctx, req)
		order = append(order, "mw1-after")
		return resp, err
	}
	mw2 := func(ctx context.Context, req *Request, next CompleteFunc) (*Response, error) {
		order = append(order, "mw2-before")
		resp, err := next(ctx, req)
		order = append(order, "mw2-after")
		return resp, err
	}

	resp := `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	client := NewClient(&mockInvoker{response: []byte(resp)},
		WithAdapter(NewAnthropicAdapter()),
		WithDefaultProvider("anthropic"),
		WithMiddleware(mw1, mw2),
	)
	_, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{UserMessage("hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"mw1-before", "mw2-before", "mw2-after", "mw1-after"}
	if len(order) != len(want) {
		t.Fatalf("order = %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Errorf("order[%d] = %q, want %q", i, order[i], want[i])
		}
	}
}

func TestClientComplete_MiddlewareCanModifyRequest(t *testing.T) {
	// Middleware that injects a provider option
	mw := func(ctx context.Context, req *Request, next CompleteFunc) (*Response, error) {
		if req.ProviderOptions == nil {
			req.ProviderOptions = make(map[string]any)
		}
		req.ProviderOptions["test"] = "injected"
		return next(ctx, req)
	}

	resp := `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	client := NewClient(&mockInvoker{response: []byte(resp)},
		WithAdapter(NewAnthropicAdapter()),
		WithDefaultProvider("anthropic"),
		WithMiddleware(mw),
	)
	_, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{UserMessage("hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	// If we got here without error, middleware ran successfully
}

// Suppress unused import
var _ = json.Marshal
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestClient`
Expected: FAIL — NewClient not defined

**Step 3: Implement client.go**

Create `llm/client.go`:

```go
package llm

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// CompleteFunc is the signature for the core completion call and middleware next functions.
type CompleteFunc func(ctx context.Context, req *Request) (*Response, error)

// Middleware wraps a Complete call.
type Middleware func(ctx context.Context, req *Request, next CompleteFunc) (*Response, error)

// Client routes requests to adapters and calls Bedrock InvokeModel.
type Client struct {
	bedrock         BedrockInvoker
	adapters        map[string]Adapter
	defaultProvider string
	middleware      []Middleware
}

type clientConfig struct {
	adapters        []Adapter
	defaultProvider string
	middleware      []Middleware
}

// ClientOption configures a Client.
type ClientOption func(*clientConfig)

// WithAdapter registers an adapter with the client.
func WithAdapter(a Adapter) ClientOption {
	return func(c *clientConfig) {
		c.adapters = append(c.adapters, a)
	}
}

// WithDefaultProvider sets the default provider for requests that don't specify one.
func WithDefaultProvider(provider string) ClientOption {
	return func(c *clientConfig) {
		c.defaultProvider = provider
	}
}

// WithMiddleware adds middleware to the client.
func WithMiddleware(m ...Middleware) ClientOption {
	return func(c *clientConfig) {
		c.middleware = append(c.middleware, m...)
	}
}

// NewClient creates a new Client with the given Bedrock invoker and options.
func NewClient(bedrock BedrockInvoker, opts ...ClientOption) *Client {
	cfg := &clientConfig{}
	for _, o := range opts {
		o(cfg)
	}

	adapters := make(map[string]Adapter, len(cfg.adapters))
	for _, a := range cfg.adapters {
		adapters[a.Provider()] = a
	}

	return &Client{
		bedrock:         bedrock,
		adapters:        adapters,
		defaultProvider: cfg.defaultProvider,
		middleware:      cfg.middleware,
	}
}

// Complete sends a request to the appropriate provider and returns the response.
func (c *Client) Complete(ctx context.Context, req *Request) (*Response, error) {
	// Resolve provider
	provider := req.Provider
	if provider == "" {
		provider = c.defaultProvider
	}
	if provider == "" {
		return nil, &Error{Kind: ErrConfig, Message: "no provider specified and no default provider set"}
	}

	adapter, ok := c.adapters[provider]
	if !ok {
		return nil, &Error{Kind: ErrConfig, Provider: provider, Message: "no adapter registered for provider"}
	}

	// Build the core function
	core := func(ctx context.Context, req *Request) (*Response, error) {
		input, err := adapter.BuildInvokeInput(req)
		if err != nil {
			return nil, err
		}

		output, err := c.bedrock.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
			ModelId:     &input.ModelID,
			Body:        input.Body,
			ContentType: &input.ContentType,
			Accept:      &input.Accept,
		})
		if err != nil {
			return nil, classifyBedrockError(provider, err)
		}

		return adapter.ParseResponse(output.Body, req)
	}

	// Wrap with middleware (first registered = outermost)
	fn := core
	for i := len(c.middleware) - 1; i >= 0; i-- {
		mw := c.middleware[i]
		next := fn
		fn = func(ctx context.Context, req *Request) (*Response, error) {
			return mw(ctx, req, next)
		}
	}

	return fn(ctx, req)
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestClient`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm/client.go llm/client_test.go
git commit -m "feat: add Client with provider routing, middleware, and Complete()"
```

---

### Task 10: Bedrock Error Classification

**Files:**
- Modify: `llm/client.go` (add `classifyBedrockError`)
- Create: `llm/errors_classify_test.go`

**Step 1: Write tests for error classification**

Create `llm/errors_classify_test.go`:

```go
package llm

import (
	"errors"
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func TestClassifyBedrockError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantKind ErrorKind
	}{
		{
			name:     "AccessDeniedException",
			err:      &types.AccessDeniedException{Message: strPtr("access denied")},
			wantKind: ErrAuthentication,
		},
		{
			name:     "ValidationException",
			err:      &types.ValidationException{Message: strPtr("invalid input")},
			wantKind: ErrInvalidRequest,
		},
		{
			name:     "ResourceNotFoundException",
			err:      &types.ResourceNotFoundException{Message: strPtr("model not found")},
			wantKind: ErrNotFound,
		},
		{
			name:     "ThrottlingException",
			err:      &types.ThrottlingException{Message: strPtr("rate limited")},
			wantKind: ErrRateLimit,
		},
		{
			name:     "ModelTimeoutException",
			err:      &types.ModelTimeoutException{Message: strPtr("timeout")},
			wantKind: ErrServer,
		},
		{
			name:     "InternalServerException",
			err:      &types.InternalServerException{Message: strPtr("internal error")},
			wantKind: ErrServer,
		},
		{
			name:     "ModelErrorException",
			err:      &types.ModelErrorException{Message: strPtr("model error")},
			wantKind: ErrServer,
		},
		{
			name:     "context length error via message",
			err:      fmt.Errorf("context length exceeded: too many tokens"),
			wantKind: ErrContextLength,
		},
		{
			name:     "content filter error via message",
			err:      fmt.Errorf("blocked by content filter"),
			wantKind: ErrContentFilter,
		},
		{
			name:     "guardrail error via message",
			err:      fmt.Errorf("blocked by guardrail policy"),
			wantKind: ErrContentFilter,
		},
		{
			name:     "unknown error",
			err:      fmt.Errorf("something unexpected"),
			wantKind: ErrServer,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := classifyBedrockError("anthropic", tt.err)
			var llmErr *Error
			if !errors.As(result, &llmErr) {
				t.Fatalf("expected *Error, got %T", result)
			}
			if llmErr.Kind != tt.wantKind {
				t.Errorf("Kind = %v, want %v", llmErr.Kind, tt.wantKind)
			}
			if llmErr.Cause != tt.err {
				t.Error("Cause should be the original error")
			}
			if llmErr.Provider != "anthropic" {
				t.Errorf("Provider = %q", llmErr.Provider)
			}
		})
	}
}

func strPtr(s string) *string { return &s }
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestClassifyBedrockError`
Expected: FAIL — classifyBedrockError not defined (or returns nil)

**Step 3: Implement classifyBedrockError**

Add to `llm/client.go`:

```go
import "strings"

func classifyBedrockError(provider string, err error) error {
	var kind ErrorKind
	msg := err.Error()

	// Check for specific Bedrock exception types
	var accessDenied *types.AccessDeniedException
	var validation *types.ValidationException
	var notFound *types.ResourceNotFoundException
	var throttling *types.ThrottlingException
	var timeout *types.ModelTimeoutException
	var internal *types.InternalServerException
	var modelErr *types.ModelErrorException

	switch {
	case errors.As(err, &accessDenied):
		kind = ErrAuthentication
	case errors.As(err, &validation):
		kind = ErrInvalidRequest
	case errors.As(err, &notFound):
		kind = ErrNotFound
	case errors.As(err, &throttling):
		kind = ErrRateLimit
	case errors.As(err, &timeout):
		kind = ErrServer
	case errors.As(err, &internal):
		kind = ErrServer
	case errors.As(err, &modelErr):
		kind = ErrServer
	default:
		// Check message content for additional classification
		lower := strings.ToLower(msg)
		switch {
		case strings.Contains(lower, "context length") || strings.Contains(lower, "too many tokens"):
			kind = ErrContextLength
		case strings.Contains(lower, "content filter") || strings.Contains(lower, "guardrail"):
			kind = ErrContentFilter
		default:
			kind = ErrServer
		}
	}

	return &Error{
		Kind:     kind,
		Provider: provider,
		Message:  msg,
		Cause:    err,
	}
}
```

This requires adding imports to `llm/client.go`:

```go
import (
	"context"
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v -run TestClassifyBedrockError`
Expected: All PASS

**Step 5: Run full test suite**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add llm/client.go llm/errors_classify_test.go
git commit -m "feat: add Bedrock error classification"
```

---

### Task 11: Final Verification

**Files:**
- None (verification only)

**Step 1: Run the full test suite**

Run: `cd /home/sprite/unified-llm && go test ./... -v -count=1`
Expected: All tests PASS

**Step 2: Run go vet**

Run: `cd /home/sprite/unified-llm && go vet ./...`
Expected: No issues

**Step 3: Verify the package is importable**

Run: `cd /home/sprite/unified-llm && go build ./...`
Expected: Clean build

**Step 4: Review test coverage**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -coverprofile=coverage.out && go tool cover -func=coverage.out`
Expected: Coverage report showing adapter and client code well-covered

**Step 5: Commit any cleanup**

If any issues were found, fix and commit. Otherwise, no action needed.
