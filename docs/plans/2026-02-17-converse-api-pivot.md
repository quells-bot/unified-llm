# Converse API Pivot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the InvokeModel + per-provider Adapter pattern with a Conversation-centric wrapper around the Bedrock Converse API.

**Architecture:** Pure-data `Conversation` struct holds all state. `Client.Send()` translates to/from Converse SDK types via internal `toConverseInput`/`fromConverseOutput` functions, calls `bedrock.Converse()`, and returns the updated conversation + per-turn response. Adapters are deleted entirely.

**Tech Stack:** Go, AWS SDK v2 (`bedrockruntime` v1.49.0), `document` package for JSON→document conversion.

---

### Task 1: Add JSON tags to existing types and add Conversation/Config

**Files:**
- Modify: `llm/types.go`
- Modify: `llm/types_test.go`

**Step 1: Write the failing test for Conversation JSON round-trip**

Add to `llm/types_test.go`:

```go
func TestConversationJSONRoundTrip(t *testing.T) {
	maxTok := 4096
	temp := 0.7
	conv := Conversation{
		Model:  "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		System: []string{"You are helpful."},
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Kind: ContentText, Text: "hello"}}},
			{Role: RoleAssistant, Content: []ContentPart{{Kind: ContentText, Text: "hi"}}},
		},
		Config: Config{
			MaxTokens:   &maxTok,
			Temperature: &temp,
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 5},
	}
	data, err := json.Marshal(conv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var restored Conversation
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if restored.Model != conv.Model {
		t.Errorf("Model = %q, want %q", restored.Model, conv.Model)
	}
	if len(restored.System) != 1 || restored.System[0] != "You are helpful." {
		t.Errorf("System = %v", restored.System)
	}
	if len(restored.Messages) != 2 {
		t.Errorf("Messages len = %d", len(restored.Messages))
	}
	if restored.Messages[0].Content[0].Text != "hello" {
		t.Errorf("first message text = %q", restored.Messages[0].Content[0].Text)
	}
	if *restored.Config.MaxTokens != 4096 {
		t.Errorf("MaxTokens = %d", *restored.Config.MaxTokens)
	}
	if *restored.Config.Temperature != 0.7 {
		t.Errorf("Temperature = %f", *restored.Config.Temperature)
	}
	if restored.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d", restored.Usage.InputTokens)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run TestConversationJSONRoundTrip -v`
Expected: FAIL — `Conversation` and `Config` types do not exist.

**Step 3: Add JSON tags to all existing types and add Conversation/Config**

In `llm/types.go`, add json tags to every struct field that needs serialization, and add the new types. Here's what changes:

```go
// ContentPart — add json tags
type ContentPart struct {
	Kind       ContentKind     `json:"kind"`
	Text       string          `json:"text,omitempty"`
	Image      *ImageData      `json:"image,omitempty"`
	ToolCall   *ToolCallData   `json:"tool_call,omitempty"`
	ToolResult *ToolResultData `json:"tool_result,omitempty"`
	Thinking   *ThinkingData   `json:"thinking,omitempty"`
}

type ImageData struct {
	URL       string `json:"url,omitempty"`
	Data      []byte `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
}

type ToolCallData struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

type ToolResultData struct {
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
	IsError    bool   `json:"is_error,omitempty"`
}

type ThinkingData struct {
	Text      string `json:"text"`
	Signature string `json:"signature,omitempty"`
}

type Message struct {
	Role       Role          `json:"role"`
	Content    []ContentPart `json:"content"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

type ToolChoice struct {
	Mode     ToolChoiceMode `json:"mode"`
	ToolName string         `json:"tool_name,omitempty"`
}

type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	params      []Param         // unexported, not serialized
}

type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`
}
```

Add new types:

```go
// Config holds inference configuration for a conversation.
type Config struct {
	MaxTokens     *int        `json:"max_tokens,omitempty"`
	Temperature   *float64    `json:"temperature,omitempty"`
	TopP          *float64    `json:"top_p,omitempty"`
	StopSequences []string    `json:"stop_sequences,omitempty"`
	ToolChoice    *ToolChoice `json:"tool_choice,omitempty"`
}

// Conversation holds the entire state of a multi-turn conversation.
// It is pure data with no client reference, designed for JSON serialization.
type Conversation struct {
	Model    string           `json:"model"`
	System   []string         `json:"system,omitempty"`
	Messages []Message        `json:"messages"`
	Tools    []ToolDefinition `json:"tools,omitempty"`
	Config   Config           `json:"config,omitempty"`
	Usage    Usage            `json:"usage"`
}
```

Add `NewConversation` constructor and option functions:

```go
// ConversationOption configures a new Conversation.
type ConversationOption func(*Conversation)

// WithSystem adds system prompt strings to the conversation.
func WithSystem(texts ...string) ConversationOption {
	return func(c *Conversation) {
		c.System = append(c.System, texts...)
	}
}

// WithTools sets the available tools for the conversation.
func WithTools(tools ...ToolDefinition) ConversationOption {
	return func(c *Conversation) {
		c.Tools = tools
	}
}

// WithMaxTokens sets the maximum tokens for responses.
func WithMaxTokens(n int) ConversationOption {
	return func(c *Conversation) {
		c.Config.MaxTokens = &n
	}
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) ConversationOption {
	return func(c *Conversation) {
		c.Config.Temperature = &t
	}
}

// WithTopP sets the nucleus sampling parameter.
func WithTopP(p float64) ConversationOption {
	return func(c *Conversation) {
		c.Config.TopP = &p
	}
}

// WithStopSequences sets stop sequences.
func WithStopSequences(seqs ...string) ConversationOption {
	return func(c *Conversation) {
		c.Config.StopSequences = seqs
	}
}

// WithToolChoice sets the tool choice mode.
func WithToolChoice(tc ToolChoice) ConversationOption {
	return func(c *Conversation) {
		c.Config.ToolChoice = &tc
	}
}

// NewConversation creates a new Conversation with the given model and options.
func NewConversation(model string, opts ...ConversationOption) Conversation {
	c := Conversation{Model: model}
	for _, o := range opts {
		o(&c)
	}
	return c
}
```

Slim down `Response` — remove `ID`, `Model`, `Provider`, `Raw`:

```go
// Response is the per-turn result from Send.
type Response struct {
	Message      Message      `json:"message"`
	FinishReason FinishReason `json:"finish_reason"`
	Usage        Usage        `json:"usage"`
}
```

Change `FinishReason` from struct to string:

```go
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonToolUse       FinishReason = "tool_use"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonError         FinishReason = "error"
)
```

Note: rename `FinishReasonToolCalls` → `FinishReasonToolUse` to match the Converse API's native `StopReason` values.

Move `ToolCalls()` helper from `Response` to `Message`:

```go
// ToolCalls returns all tool call content parts from the message.
func (m Message) ToolCalls() []ToolCallData {
	var calls []ToolCallData
	for _, p := range m.Content {
		if p.Kind == ContentToolCall && p.ToolCall != nil {
			calls = append(calls, *p.ToolCall)
		}
	}
	return calls
}
```

Remove the old `Request` struct entirely. Remove `Response.Text()` (callers use `resp.Message.Text()`). Remove `Response.ToolCalls()` (callers use `resp.Message.ToolCalls()`).

**Step 4: Run test to verify it passes**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run TestConversationJSONRoundTrip -v`
Expected: PASS

Note: existing tests will be broken at this point — that's expected. We fix them in later tasks.

**Step 5: Write test for NewConversation with options**

Add to `llm/types_test.go`:

```go
func TestNewConversation(t *testing.T) {
	tool := NewTool("greet", "Say hello", StringParam("name"))
	conv := NewConversation("my-model",
		WithSystem("Be helpful."),
		WithTools(tool),
		WithMaxTokens(1024),
		WithTemperature(0.5),
	)
	if conv.Model != "my-model" {
		t.Errorf("Model = %q", conv.Model)
	}
	if len(conv.System) != 1 || conv.System[0] != "Be helpful." {
		t.Errorf("System = %v", conv.System)
	}
	if len(conv.Tools) != 1 || conv.Tools[0].Name != "greet" {
		t.Errorf("Tools = %v", conv.Tools)
	}
	if *conv.Config.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %v", conv.Config.MaxTokens)
	}
	if *conv.Config.Temperature != 0.5 {
		t.Errorf("Temperature = %v", conv.Config.Temperature)
	}
}
```

**Step 6: Run test to verify it passes**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run TestNewConversation -v`
Expected: PASS (types were added in step 3)

**Step 7: Commit**

```bash
git add llm/types.go llm/types_test.go
git commit -m "feat: add Conversation/Config types, JSON tags, slim Response/FinishReason"
```

---

### Task 2: Write the Converse translation layer

**Files:**
- Create: `llm/converse.go`
- Create: `llm/converse_test.go`

This task implements the internal translation functions between our types and the Converse SDK types.

**Step 1: Write failing tests for `toConverseInput`**

Create `llm/converse_test.go`:

```go
package llm

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func TestToConverseInput_SimpleText(t *testing.T) {
	conv := NewConversation("us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		WithSystem("Be helpful."),
		WithMaxTokens(1024),
	)
	conv.Messages = []Message{
		{Role: RoleUser, Content: []ContentPart{{Kind: ContentText, Text: "hello"}}},
	}

	input := toConverseInput(&conv)

	if *input.ModelId != "us.anthropic.claude-sonnet-4-5-20250929-v1:0" {
		t.Errorf("ModelId = %q", *input.ModelId)
	}
	if len(input.System) != 1 {
		t.Fatalf("System len = %d", len(input.System))
	}
	sysText, ok := input.System[0].(*types.SystemContentBlockMemberText)
	if !ok {
		t.Fatalf("System[0] type = %T", input.System[0])
	}
	if sysText.Value != "Be helpful." {
		t.Errorf("System text = %q", sysText.Value)
	}
	if len(input.Messages) != 1 {
		t.Fatalf("Messages len = %d", len(input.Messages))
	}
	if input.Messages[0].Role != types.ConversationRoleUser {
		t.Errorf("Role = %v", input.Messages[0].Role)
	}
	textBlock, ok := input.Messages[0].Content[0].(*types.ContentBlockMemberText)
	if !ok {
		t.Fatalf("Content[0] type = %T", input.Messages[0].Content[0])
	}
	if textBlock.Value != "hello" {
		t.Errorf("Text = %q", textBlock.Value)
	}
	if *input.InferenceConfig.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %d", *input.InferenceConfig.MaxTokens)
	}
}

func TestToConverseInput_WithTools(t *testing.T) {
	tool := NewTool("get_weather", "Get weather", StringParam("location"))
	conv := NewConversation("us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		WithTools(tool),
	)
	conv.Messages = []Message{
		{Role: RoleUser, Content: []ContentPart{{Kind: ContentText, Text: "weather?"}}},
	}

	input := toConverseInput(&conv)

	if input.ToolConfig == nil {
		t.Fatal("ToolConfig is nil")
	}
	if len(input.ToolConfig.Tools) != 1 {
		t.Fatalf("Tools len = %d", len(input.ToolConfig.Tools))
	}
	spec, ok := input.ToolConfig.Tools[0].(*types.ToolMemberToolSpec)
	if !ok {
		t.Fatalf("Tool type = %T", input.ToolConfig.Tools[0])
	}
	if *spec.Value.Name != "get_weather" {
		t.Errorf("Tool name = %q", *spec.Value.Name)
	}
}

func TestToConverseInput_ToolChoice(t *testing.T) {
	tool := NewTool("my_tool", "A tool")
	tests := []struct {
		name       string
		choice     ToolChoice
		wantType   string
	}{
		{"auto", ToolChoice{Mode: ToolChoiceAuto}, "auto"},
		{"required", ToolChoice{Mode: ToolChoiceRequired}, "any"},
		{"named", ToolChoice{Mode: ToolChoiceNamed, ToolName: "my_tool"}, "tool"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv := NewConversation("model",
				WithTools(tool),
				WithToolChoice(tt.choice),
			)
			conv.Messages = []Message{{Role: RoleUser, Content: []ContentPart{{Kind: ContentText, Text: "go"}}}}
			input := toConverseInput(&conv)
			if input.ToolConfig == nil {
				t.Fatal("ToolConfig is nil")
			}
			switch tt.wantType {
			case "auto":
				if _, ok := input.ToolConfig.ToolChoice.(*types.ToolChoiceMemberAuto); !ok {
					t.Errorf("expected Auto, got %T", input.ToolConfig.ToolChoice)
				}
			case "any":
				if _, ok := input.ToolConfig.ToolChoice.(*types.ToolChoiceMemberAny); !ok {
					t.Errorf("expected Any, got %T", input.ToolConfig.ToolChoice)
				}
			case "tool":
				tc, ok := input.ToolConfig.ToolChoice.(*types.ToolChoiceMemberTool)
				if !ok {
					t.Errorf("expected Tool, got %T", input.ToolConfig.ToolChoice)
				} else if *tc.Value.Name != "my_tool" {
					t.Errorf("tool name = %q", *tc.Value.Name)
				}
			}
		})
	}
}

func TestToConverseInput_ToolResultMessage(t *testing.T) {
	conv := Conversation{
		Model: "model",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Kind: ContentText, Text: "go"}}},
			{Role: RoleAssistant, Content: []ContentPart{{
				Kind:     ContentToolCall,
				ToolCall: &ToolCallData{ID: "call-1", Name: "foo", Arguments: []byte(`{"x":1}`)},
			}}},
			ToolResultMessage("call-1", "result-data", false),
		},
	}
	input := toConverseInput(&conv)
	// Tool result should be in a user-role message
	if len(input.Messages) != 3 {
		t.Fatalf("Messages len = %d", len(input.Messages))
	}
	trMsg := input.Messages[2]
	if trMsg.Role != types.ConversationRoleUser {
		t.Errorf("tool result role = %v", trMsg.Role)
	}
	trBlock, ok := trMsg.Content[0].(*types.ContentBlockMemberToolResult)
	if !ok {
		t.Fatalf("content type = %T", trMsg.Content[0])
	}
	if *trBlock.Value.ToolUseId != "call-1" {
		t.Errorf("ToolUseId = %q", *trBlock.Value.ToolUseId)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run TestToConverseInput -v`
Expected: FAIL — `toConverseInput` does not exist.

**Step 3: Write failing tests for `fromConverseOutput`**

Add to `llm/converse_test.go`:

```go
func TestFromConverseOutput_SimpleText(t *testing.T) {
	out := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role: types.ConversationRoleAssistant,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: "Hello!"},
				},
			},
		},
		StopReason: types.StopReasonEndTurn,
		Usage: &types.TokenUsage{
			InputTokens:  int32Ptr(10),
			OutputTokens: int32Ptr(5),
			TotalTokens:  int32Ptr(15),
		},
	}
	msg, usage, reason, err := fromConverseOutput(out)
	if err != nil {
		t.Fatal(err)
	}
	if msg.Role != RoleAssistant {
		t.Errorf("Role = %v", msg.Role)
	}
	if msg.Text() != "Hello!" {
		t.Errorf("Text = %q", msg.Text())
	}
	if reason != FinishReasonStop {
		t.Errorf("FinishReason = %q", reason)
	}
	if usage.InputTokens != 10 || usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v", usage)
	}
}

func TestFromConverseOutput_ToolUse(t *testing.T) {
	out := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role: types.ConversationRoleAssistant,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: "Let me check."},
					&types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
						ToolUseId: strPtr("call-1"),
						Name:      strPtr("get_weather"),
						Input:     makeDocument(map[string]any{"location": "SF"}),
					}},
				},
			},
		},
		StopReason: types.StopReasonToolUse,
		Usage:      &types.TokenUsage{InputTokens: int32Ptr(5), OutputTokens: int32Ptr(10), TotalTokens: int32Ptr(15)},
	}
	msg, _, reason, err := fromConverseOutput(out)
	if err != nil {
		t.Fatal(err)
	}
	if reason != FinishReasonToolUse {
		t.Errorf("FinishReason = %q", reason)
	}
	calls := msg.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("ToolCalls len = %d", len(calls))
	}
	if calls[0].ID != "call-1" || calls[0].Name != "get_weather" {
		t.Errorf("ToolCall = %+v", calls[0])
	}
}

func TestFromConverseOutput_WithCacheTokens(t *testing.T) {
	out := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role:    types.ConversationRoleAssistant,
				Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "ok"}},
			},
		},
		StopReason: types.StopReasonEndTurn,
		Usage: &types.TokenUsage{
			InputTokens:          int32Ptr(100),
			OutputTokens:         int32Ptr(50),
			TotalTokens:          int32Ptr(150),
			CacheReadInputTokens:  int32Ptr(80),
			CacheWriteInputTokens: int32Ptr(20),
		},
	}
	_, usage, _, err := fromConverseOutput(out)
	if err != nil {
		t.Fatal(err)
	}
	if usage.CacheReadTokens != 80 {
		t.Errorf("CacheReadTokens = %d", usage.CacheReadTokens)
	}
	if usage.CacheWriteTokens != 20 {
		t.Errorf("CacheWriteTokens = %d", usage.CacheWriteTokens)
	}
}

func int32Ptr(v int32) *int32 { return &v }
```

Note: `makeDocument` is a test helper that creates a `document.Interface` from a map. You'll need to import `"github.com/aws/smithy-go/document"` and figure out the correct way to create document values. The smithy-go document package uses interface types — in the SDK, `document.NewLazyDocument(value)` can wrap arbitrary Go values. Check the SDK source if needed, but the pattern is:

```go
func makeDocument(v any) document.Interface {
	// smithy-go document wraps Go values for JSON
	return document.NewLazyDocument(v)
}
```

If `NewLazyDocument` isn't available, use `smithydocument.NewLazyDocument` from `github.com/aws/smithy-go/document`. Verify the exact import at implementation time.

**Step 4: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run TestFromConverseOutput -v`
Expected: FAIL

**Step 5: Implement `toConverseInput` and `fromConverseOutput`**

Create `llm/converse.go`:

```go
package llm

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithydocument "github.com/aws/smithy-go/document"
)

// toConverseInput translates a Conversation into a Bedrock ConverseInput.
func toConverseInput(conv *Conversation) *bedrockruntime.ConverseInput {
	input := &bedrockruntime.ConverseInput{
		ModelId: &conv.Model,
	}

	// System prompts
	for _, s := range conv.System {
		input.System = append(input.System, &types.SystemContentBlockMemberText{Value: s})
	}

	// Messages
	for _, m := range conv.Messages {
		input.Messages = append(input.Messages, toConverseMessage(m))
	}

	// Inference config
	if conv.Config.MaxTokens != nil || conv.Config.Temperature != nil || conv.Config.TopP != nil || len(conv.Config.StopSequences) > 0 {
		ic := &types.InferenceConfiguration{}
		if conv.Config.MaxTokens != nil {
			v := int32(*conv.Config.MaxTokens)
			ic.MaxTokens = &v
		}
		if conv.Config.Temperature != nil {
			v := float32(*conv.Config.Temperature)
			ic.Temperature = &v
		}
		if conv.Config.TopP != nil {
			v := float32(*conv.Config.TopP)
			ic.TopP = &v
		}
		if len(conv.Config.StopSequences) > 0 {
			ic.StopSequences = conv.Config.StopSequences
		}
		input.InferenceConfig = ic
	}

	// Tools
	if len(conv.Tools) > 0 {
		tc := &types.ToolConfiguration{}
		for _, td := range conv.Tools {
			var schema types.ToolInputSchema
			var doc any
			json.Unmarshal(td.Parameters, &doc)
			schema = &types.ToolInputSchemaMemberJson{Value: smithydocument.NewLazyDocument(doc)}
			spec := types.ToolSpecification{
				Name:        &td.Name,
				InputSchema: schema,
			}
			if td.Description != "" {
				spec.Description = &td.Description
			}
			tc.Tools = append(tc.Tools, &types.ToolMemberToolSpec{Value: spec})
		}
		// Tool choice
		if conv.Config.ToolChoice != nil {
			switch conv.Config.ToolChoice.Mode {
			case ToolChoiceAuto:
				tc.ToolChoice = &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}
			case ToolChoiceRequired:
				tc.ToolChoice = &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}
			case ToolChoiceNamed:
				tc.ToolChoice = &types.ToolChoiceMemberTool{
					Value: types.SpecificToolChoice{Name: &conv.Config.ToolChoice.ToolName},
				}
			case ToolChoiceNone:
				// Omit tools entirely
				tc = nil
			}
		}
		input.ToolConfig = tc
	}

	// Anthropic-specific: add cache points for prompt caching
	if isAnthropicModel(conv.Model) && len(input.System) > 0 {
		input.System = append(input.System, &types.SystemContentBlockMemberCachePoint{
			Value: types.CachePointBlock{},
		})
	}

	return input
}

func toConverseMessage(m Message) types.Message {
	msg := types.Message{}

	switch m.Role {
	case RoleUser:
		msg.Role = types.ConversationRoleUser
	case RoleAssistant:
		msg.Role = types.ConversationRoleAssistant
	case RoleTool:
		msg.Role = types.ConversationRoleUser
	}

	for _, p := range m.Content {
		switch p.Kind {
		case ContentText:
			msg.Content = append(msg.Content, &types.ContentBlockMemberText{Value: p.Text})
		case ContentToolCall:
			var doc any
			json.Unmarshal(p.ToolCall.Arguments, &doc)
			msg.Content = append(msg.Content, &types.ContentBlockMemberToolUse{
				Value: types.ToolUseBlock{
					ToolUseId: &p.ToolCall.ID,
					Name:      &p.ToolCall.Name,
					Input:     smithydocument.NewLazyDocument(doc),
				},
			})
		case ContentToolResult:
			status := types.ToolResultStatusSuccess
			if p.ToolResult.IsError {
				status = types.ToolResultStatusError
			}
			msg.Content = append(msg.Content, &types.ContentBlockMemberToolResult{
				Value: types.ToolResultBlock{
					ToolUseId: &p.ToolResult.ToolCallID,
					Content: []types.ToolResultContentBlock{
						&types.ToolResultContentBlockMemberText{Value: p.ToolResult.Content},
					},
					Status: status,
				},
			})
		case ContentImage:
			if p.Image != nil && len(p.Image.Data) > 0 {
				msg.Content = append(msg.Content, &types.ContentBlockMemberImage{
					Value: types.ImageBlock{
						Format: types.ImageFormat(p.Image.MediaType),
						Source: &types.ImageSourceMemberBytes{Value: p.Image.Data},
					},
				})
			}
		case ContentThinking:
			// Only include thinking blocks for Anthropic models (handled by caller context)
			msg.Content = append(msg.Content, &types.ContentBlockMemberReasoningContent{
				Value: types.ReasoningContentBlock{
					// ReasoningContent uses a union — check SDK for exact member
				},
			})
		}
	}

	return msg
}

// fromConverseOutput translates a Bedrock ConverseOutput into our types.
func fromConverseOutput(out *bedrockruntime.ConverseOutput) (*Message, *Usage, FinishReason, error) {
	msgOut, ok := out.Output.(*types.ConverseOutputMemberMessage)
	if !ok {
		return nil, nil, "", fmt.Errorf("unexpected output type: %T", out.Output)
	}

	msg := &Message{Role: RoleAssistant}
	for _, block := range msgOut.Value.Content {
		switch b := block.(type) {
		case *types.ContentBlockMemberText:
			msg.Content = append(msg.Content, ContentPart{Kind: ContentText, Text: b.Value})
		case *types.ContentBlockMemberToolUse:
			args, _ := json.Marshal(b.Value.Input)
			msg.Content = append(msg.Content, ContentPart{
				Kind: ContentToolCall,
				ToolCall: &ToolCallData{
					ID:        derefStr(b.Value.ToolUseId),
					Name:      derefStr(b.Value.Name),
					Arguments: args,
				},
			})
		case *types.ContentBlockMemberReasoningContent:
			// Map reasoning content to our thinking type
			// The exact field access depends on the SDK union structure
		}
	}

	usage := &Usage{}
	if out.Usage != nil {
		if out.Usage.InputTokens != nil {
			usage.InputTokens = int(*out.Usage.InputTokens)
		}
		if out.Usage.OutputTokens != nil {
			usage.OutputTokens = int(*out.Usage.OutputTokens)
		}
		if out.Usage.CacheReadInputTokens != nil {
			usage.CacheReadTokens = int(*out.Usage.CacheReadInputTokens)
		}
		if out.Usage.CacheWriteInputTokens != nil {
			usage.CacheWriteTokens = int(*out.Usage.CacheWriteInputTokens)
		}
	}

	reason := mapStopReason(out.StopReason)

	return msg, usage, reason, nil
}

func mapStopReason(sr types.StopReason) FinishReason {
	switch sr {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return FinishReasonStop
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return FinishReasonLength
	case types.StopReasonToolUse:
		return FinishReasonToolUse
	case types.StopReasonContentFiltered, types.StopReasonGuardrailIntervened:
		return FinishReasonContentFilter
	default:
		return FinishReason(string(sr))
	}
}

func isAnthropicModel(model string) bool {
	return strings.Contains(model, "anthropic.")
}

func derefStr(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
```

**Important implementation notes:**
- The `document.Interface` / `smithydocument.NewLazyDocument` import path needs to be verified at implementation time. Check `go doc github.com/aws/smithy-go/document` in the module cache.
- The `ReasoningContentBlock` union structure needs to be checked — it may have sub-members like `ReasoningContentBlockMemberReasoningText` with `Text` and `Signature` fields. Verify at implementation time by reading the SDK types.
- For tool use `Input`, marshaling a `document.Interface` back to `json.RawMessage` — `json.Marshal` should work on `LazyDocument` since it wraps a Go value.

**Step 6: Run all converse tests**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run "TestToConverseInput|TestFromConverseOutput" -v`
Expected: PASS

**Step 7: Commit**

```bash
git add llm/converse.go llm/converse_test.go
git commit -m "feat: add Converse API translation layer (toConverseInput/fromConverseOutput)"
```

---

### Task 3: Rewrite Client around Conversation + Send

**Files:**
- Modify: `llm/client.go`
- Create: `llm/client_test.go` (rewrite)

**Step 1: Write failing test for `Client.Send`**

Replace `llm/client_test.go` with:

```go
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// mockConverser is a test double for BedrockConverser.
type mockConverser struct {
	output *bedrockruntime.ConverseOutput
	err    error
}

func (m *mockConverser) Converse(ctx context.Context, params *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.output, nil
}

func simpleConverseOutput(text string) *bedrockruntime.ConverseOutput {
	return &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role: types.ConversationRoleAssistant,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: text},
				},
			},
		},
		StopReason: types.StopReasonEndTurn,
		Usage: &types.TokenUsage{
			InputTokens:  int32Ptr(10),
			OutputTokens: int32Ptr(5),
			TotalTokens:  int32Ptr(15),
		},
	}
}

func TestClientSend_SimpleText(t *testing.T) {
	client := NewClient(&mockConverser{output: simpleConverseOutput("Hello!")})

	conv := NewConversation("us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		WithSystem("Be helpful."),
		WithMaxTokens(1024),
	)

	conv, resp, err := client.Send(ctx(t), conv, UserMessage("hi"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Text() != "Hello!" {
		t.Errorf("Text = %q", resp.Message.Text())
	}
	if resp.FinishReason != FinishReasonStop {
		t.Errorf("FinishReason = %q", resp.FinishReason)
	}
	// Conversation should have 2 messages: user + assistant
	if len(conv.Messages) != 2 {
		t.Errorf("Messages len = %d", len(conv.Messages))
	}
	if conv.Messages[0].Text() != "hi" {
		t.Errorf("Messages[0] = %q", conv.Messages[0].Text())
	}
	if conv.Messages[1].Text() != "Hello!" {
		t.Errorf("Messages[1] = %q", conv.Messages[1].Text())
	}
	// Usage should be accumulated
	if conv.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d", conv.Usage.InputTokens)
	}
}

func TestClientSend_AccumulatesUsage(t *testing.T) {
	mock := &mockConverser{output: simpleConverseOutput("reply")}
	client := NewClient(mock)

	conv := NewConversation("model")
	conv, _, err := client.Send(ctx(t), conv, UserMessage("first"))
	if err != nil {
		t.Fatal(err)
	}
	conv, _, err = client.Send(ctx(t), conv, UserMessage("second"))
	if err != nil {
		t.Fatal(err)
	}

	// 2 calls × 10 input tokens each
	if conv.Usage.InputTokens != 20 {
		t.Errorf("InputTokens = %d, want 20", conv.Usage.InputTokens)
	}
	// 4 messages: user, assistant, user, assistant
	if len(conv.Messages) != 4 {
		t.Errorf("Messages len = %d, want 4", len(conv.Messages))
	}
}

func TestClientSend_BedrockError(t *testing.T) {
	mock := &mockConverser{err: &types.ThrottlingException{Message: strPtr("slow down")}}
	client := NewClient(mock)

	conv := NewConversation("model")
	_, _, err := client.Send(ctx(t), conv, UserMessage("hi"))
	if err == nil {
		t.Fatal("expected error")
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) {
		t.Fatalf("expected *Error, got %T", err)
	}
	if llmErr.Kind != ErrRateLimit {
		t.Errorf("Kind = %v, want ErrRateLimit", llmErr.Kind)
	}
}

func TestClientSend_ImmutableConversation(t *testing.T) {
	client := NewClient(&mockConverser{output: simpleConverseOutput("reply")})

	original := NewConversation("model")
	updated, _, err := client.Send(ctx(t), original, UserMessage("hi"))
	if err != nil {
		t.Fatal(err)
	}

	// Original should be unchanged
	if len(original.Messages) != 0 {
		t.Errorf("original Messages len = %d, want 0", len(original.Messages))
	}
	// Updated should have messages
	if len(updated.Messages) != 2 {
		t.Errorf("updated Messages len = %d, want 2", len(updated.Messages))
	}
}

func ctx(t *testing.T) context.Context {
	t.Helper()
	return context.Background()
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run TestClientSend -v`
Expected: FAIL — `Send` method, `BedrockConverser`, `NewClient` signature don't exist yet.

**Step 3: Rewrite `client.go`**

```go
package llm

import (
	"context"
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// SendFunc is the signature for the core Send call and middleware next functions.
type SendFunc func(ctx context.Context, conv *Conversation) (*Response, error)

// Middleware wraps a Send call.
type Middleware func(ctx context.Context, conv *Conversation, next SendFunc) (*Response, error)

// BedrockConverser abstracts the Bedrock Converse call for testing.
type BedrockConverser interface {
	Converse(ctx context.Context, params *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
}

// Client calls the Bedrock Converse API.
type Client struct {
	bedrock    BedrockConverser
	middleware []Middleware
}

// ClientOption configures a Client.
type ClientOption func(*Client)

// WithMiddleware adds middleware to the client.
func WithMiddleware(m ...Middleware) ClientOption {
	return func(c *Client) {
		c.middleware = append(c.middleware, m...)
	}
}

// NewClient creates a new Client with the given Bedrock converser and options.
func NewClient(bedrock BedrockConverser, opts ...ClientOption) *Client {
	c := &Client{bedrock: bedrock}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Send appends the provided messages to a copy of the conversation,
// calls Bedrock Converse, appends the assistant response, accumulates usage,
// and returns the updated conversation and per-turn response.
func (c *Client) Send(ctx context.Context, conv Conversation, messages ...Message) (Conversation, *Response, error) {
	// Copy messages slice so caller's conversation is not mutated
	conv.Messages = append(append([]Message(nil), conv.Messages...), messages...)

	core := func(ctx context.Context, conv *Conversation) (*Response, error) {
		input := toConverseInput(conv)
		output, err := c.bedrock.Converse(ctx, input)
		if err != nil {
			return nil, classifyBedrockError(err)
		}
		msg, usage, reason, err := fromConverseOutput(output)
		if err != nil {
			return nil, err
		}
		return &Response{
			Message:      *msg,
			FinishReason: reason,
			Usage:        *usage,
		}, nil
	}

	// Wrap with middleware
	fn := core
	for i := len(c.middleware) - 1; i >= 0; i-- {
		mw := c.middleware[i]
		next := fn
		fn = func(ctx context.Context, conv *Conversation) (*Response, error) {
			return mw(ctx, conv, next)
		}
	}

	resp, err := fn(ctx, &conv)
	if err != nil {
		return conv, nil, err
	}

	// Append assistant response and accumulate usage
	conv.Messages = append(conv.Messages, resp.Message)
	conv.Usage = conv.Usage.Add(resp.Usage)

	return conv, resp, nil
}
```

Update `classifyBedrockError` to remove the `provider` parameter (no longer needed):

```go
func classifyBedrockError(err error) error {
	var kind ErrorKind
	msg := err.Error()

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
		Kind:    kind,
		Message: msg,
		Cause:   err,
	}
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run TestClientSend -v`
Expected: PASS

**Step 5: Commit**

```bash
git add llm/client.go llm/client_test.go
git commit -m "feat: rewrite Client around Conversation + Send using Converse API"
```

---

### Task 4: Update middleware signature and tests

**Files:**
- Modify: `llm/client.go` (already done in Task 3 — middleware types are there)
- Modify: `llm/client_test.go`

**Step 1: Write middleware tests**

Add to `llm/client_test.go`:

```go
func TestClientSend_MiddlewareOrder(t *testing.T) {
	var order []string
	mw1 := func(ctx context.Context, conv *Conversation, next SendFunc) (*Response, error) {
		order = append(order, "mw1-before")
		resp, err := next(ctx, conv)
		order = append(order, "mw1-after")
		return resp, err
	}
	mw2 := func(ctx context.Context, conv *Conversation, next SendFunc) (*Response, error) {
		order = append(order, "mw2-before")
		resp, err := next(ctx, conv)
		order = append(order, "mw2-after")
		return resp, err
	}

	client := NewClient(
		&mockConverser{output: simpleConverseOutput("ok")},
		WithMiddleware(mw1, mw2),
	)
	conv := NewConversation("model")
	_, _, err := client.Send(ctx(t), conv, UserMessage("hi"))
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
```

**Step 2: Run test**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run TestClientSend_Middleware -v`
Expected: PASS (middleware is already implemented in Task 3)

**Step 3: Commit**

```bash
git add llm/client_test.go
git commit -m "test: add middleware tests for Send-based client"
```

---

### Task 5: Update error types and tests

**Files:**
- Modify: `llm/errors.go`
- Modify: `llm/errors_test.go`
- Modify: `llm/errors_classify_test.go`

**Step 1: Update errors.go**

Remove `ErrAdapter` from `ErrorKind` (no more adapters). Remove `Provider` and `Raw` fields from `Error` struct. Update the error message format.

Actually — keeping `Provider` on `Error` is still useful for debugging ("which model caused this?"). The design doc says to remove `Raw` from Response but doesn't mention Error. Let's keep `Error` as-is except remove `ErrAdapter`. Update `Error.Provider` to be set from the conversation's model ID instead.

Actually, let's keep `errors.go` mostly as-is. The only change:
- Remove `ErrAdapter` (renumber or leave gap)
- Remove `Raw` field from `Error` (Converse API gives structured data, not raw bytes)

```go
// Remove ErrAdapter, rename it or remove
// Remove Raw field from Error
```

**Step 2: Update `classifyBedrockError` to not take provider param**

Already done in Task 3. Update the test accordingly.

**Step 3: Update `errors_classify_test.go`**

Change `classifyBedrockError("anthropic", tt.err)` → `classifyBedrockError(tt.err)`. Remove provider assertion.

**Step 4: Update `errors_test.go`**

Keep as-is or update if `Provider` field behavior changed.

**Step 5: Run tests**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run "TestError|TestClassify" -v`
Expected: PASS

**Step 6: Commit**

```bash
git add llm/errors.go llm/errors_test.go llm/errors_classify_test.go
git commit -m "refactor: remove ErrAdapter, clean up error types for Converse API"
```

---

### Task 6: Delete adapter files

**Files:**
- Delete: `llm/adapter.go`
- Delete: `llm/anthropic.go`
- Delete: `llm/openai.go`
- Delete: `llm/anthropic_test.go`
- Delete: `llm/openai_test.go`
- Delete: `llm/testdata/anthropic/` (all golden files)
- Delete: `llm/testdata/openai/` (all golden files)

**Step 1: Delete the files**

```bash
rm llm/adapter.go llm/anthropic.go llm/openai.go
rm llm/anthropic_test.go llm/openai_test.go
rm -rf llm/testdata/
```

**Step 2: Run all tests to verify nothing is broken**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v`
Expected: PASS — all remaining tests should work against the new types.

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: delete adapter layer (anthropic.go, openai.go, adapter.go)"
```

---

### Task 7: Update doc.go

**Files:**
- Modify: `llm/doc.go`

**Step 1: Update the package doc comment**

```go
// Package llm provides a Conversation-centric wrapper around the AWS Bedrock Converse API.
//
// The Conversation type holds the entire conversation state as serializable data,
// making it a natural fit for Temporal workflow payloads and other persistence mechanisms.
package llm
```

**Step 2: Commit**

```bash
git add llm/doc.go
git commit -m "docs: update package doc for Converse API"
```

---

### Task 8: Update the example

**Files:**
- Modify: `examples/tools/main.go`

**Step 1: Rewrite the example to use the new API**

```go
package main

import (
	"context"
	"log"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/quells-bot/unified-llm/llm"
)

const haiku = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

func main() {
	ctx := context.Background()
	conf, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Printf("failed to load aws config: %v", err)
		return
	}

	bd := bedrockruntime.NewFromConfig(conf)

	tools := []llm.ToolDefinition{
		llm.NewTool("get_user", "Get details of the user you are conversing with"),
		llm.NewTool("list_user_orders", "Get summary of a user's orders", llm.IntegerParam("user_id")),
		llm.NewTool("get_user_order", "Get details of a user's order", llm.IntegerParam("user_id"), llm.IntegerParam("order_id")),
	}
	toolsByName := make(map[string]llm.ToolDefinition, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	client := llm.NewClient(bd)

	conv := llm.NewConversation(haiku,
		llm.WithSystem("You are a service representative engaged in a polite conversation with a customer. "+
			"Anything you say to the user will be written to a simple chat interface, so respond with plain text. "+
			"Do not write any Markdown, code, or ASCII art. "+
			"Be as concise and straightforward as possible."),
		llm.WithTools(tools...),
		llm.WithMaxTokens(4096),
	)

	var resp *llm.Response
	userMsg := llm.UserMessage("How much do I usually spend on orders?")

	for {
		conv, resp, err = client.Send(ctx, conv, userMsg)
		if err != nil {
			log.Printf("failed to send: %v", err)
			return
		}

		log.Printf("< %s", resp.Message.Text())
		log.Printf("u %+v", resp.Usage)

		if resp.FinishReason != llm.FinishReasonToolUse {
			break
		}

		// Build tool result messages
		var results []llm.Message
		for _, tc := range resp.Message.ToolCalls() {
			tool, ok := toolsByName[tc.Name]
			if !ok {
				results = append(results, tc.ErrorResult(`{"error":"unknown tool"}`))
				continue
			}
			args, parseErr := tool.ParseArgs(tc)
			if parseErr != nil {
				results = append(results, tc.ErrorResult(`{"error":"`+parseErr.Error()+`"}`))
				continue
			}
			log.Printf("t %s %+v", tc.Name, args)

			switch tc.Name {
			case "get_user":
				results = append(results, tc.Result(`{"user_id":123}`))
			case "list_user_orders":
				userID, _ := args.Int("user_id")
				if userID != 123 {
					results = append(results, tc.ErrorResult(`{"error":"unknown user_id"}`))
					continue
				}
				results = append(results, tc.Result(`{"orders":[{"id":1000},{"id":1001},{"id":1002}]}`))
			case "get_user_order":
				userID, _ := args.Int("user_id")
				if userID != 123 {
					results = append(results, tc.ErrorResult(`{"error":"unknown user_id"}`))
					continue
				}
				orderID, _ := args.Int("order_id")
				switch orderID {
				case 1000:
					results = append(results, tc.Result(`{"amount":12.34}`))
				case 1001:
					results = append(results, tc.Result(`{"amount":23.45}`))
				case 1002:
					results = append(results, tc.Result(`{"amount":34.56}`))
				default:
					results = append(results, tc.ErrorResult(`{"error":"unknown order_id"}`))
				}
			}
		}

		// Send tool results as the next messages — no separate userMsg needed
		conv, resp, err = client.Send(ctx, conv, results...)
		if err != nil {
			log.Printf("failed to send tool results: %v", err)
			return
		}

		log.Printf("< %s", resp.Message.Text())
		log.Printf("u %+v", resp.Usage)

		if resp.FinishReason != llm.FinishReasonToolUse {
			break
		}
		// If still tool_use, loop continues — need to set userMsg to nothing
		// Actually the loop structure needs adjustment. Let's use a simpler pattern:
		userMsg = llm.Message{} // empty — won't be used since we break or loop with results
	}

	log.Printf("Total usage: %+v", conv.Usage)
}
```

Actually, the loop above is awkward. A cleaner pattern:

```go
	// Initial send
	conv, resp, err = client.Send(ctx, conv, llm.UserMessage("How much do I usually spend on orders?"))
	if err != nil {
		log.Printf("failed to send: %v", err)
		return
	}

	for resp.FinishReason == llm.FinishReasonToolUse {
		log.Printf("< %s", resp.Message.Text())

		var results []llm.Message
		for _, tc := range resp.Message.ToolCalls() {
			// ... execute tools, build results ...
		}

		conv, resp, err = client.Send(ctx, conv, results...)
		if err != nil {
			log.Printf("failed to send: %v", err)
			return
		}
	}

	log.Printf("< %s", resp.Message.Text())
	log.Printf("Total usage: %+v", conv.Usage)
```

Use this cleaner pattern in the actual implementation.

**Step 2: Verify it compiles**

Run: `cd /home/sprite/unified-llm && go build ./examples/tools/`
Expected: Compiles successfully.

**Step 3: Commit**

```bash
git add examples/tools/main.go
git commit -m "refactor: update tools example for Conversation/Send API"
```

---

### Task 9: Final verification

**Step 1: Run all tests**

Run: `cd /home/sprite/unified-llm && go test ./llm/ -v`
Expected: All tests PASS.

**Step 2: Run go vet and go build**

Run: `cd /home/sprite/unified-llm && go vet ./... && go build ./...`
Expected: Clean.

**Step 3: Verify JSON round-trip works end-to-end**

This was tested in Task 1, but verify it's still passing:

Run: `cd /home/sprite/unified-llm && go test ./llm/ -run TestConversationJSONRoundTrip -v`
Expected: PASS.

**Step 4: Final commit if any cleanup needed**

```bash
# Only if there are remaining changes
git add -A
git commit -m "chore: final cleanup after Converse API pivot"
```

---

## Implementation Notes

### Key SDK type mappings to verify at implementation time

1. **`document.Interface` creation:** The `smithy-go/document` package should have `NewLazyDocument(v any)` — verify the exact import path and function name. It may be `github.com/aws/smithy-go/document`.

2. **`ReasoningContentBlock` union:** The thinking/reasoning content block is a union type in the SDK. Check for members like `ReasoningContentBlockMemberReasoningText` with fields `Text string` and `Signature *string`. This maps to our `ThinkingData`.

3. **`ImageFormat`:** The SDK's `ImageFormat` is a string enum (`"png"`, `"jpeg"`, `"gif"`, `"webp"`). Our `ImageData.MediaType` uses MIME types like `"image/png"`. Need to strip the `image/` prefix when converting.

4. **Tool result content blocks:** `ToolResultContentBlock` is an interface with member `ToolResultContentBlockMemberText{Value: string}`. Used inside `ToolResultBlock.Content`.

5. **`ConverseInput.Messages` must be non-empty** — the API requires at least one message.

6. **Anthropic cache points:** Add `SystemContentBlockMemberCachePoint` after the last system block, and optionally `ToolMemberCachePoint` after the last tool, for Anthropic models only.
