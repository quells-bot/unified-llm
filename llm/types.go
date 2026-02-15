package llm

import (
	"encoding/json"
	"fmt"
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

// ParseArgs unmarshals the tool call's JSON arguments into a ToolCallArgs map.
func (tc ToolCallData) ParseArgs() (ToolCallArgs, error) {
	args := make(ToolCallArgs)
	if len(tc.Arguments) > 0 {
		if err := json.Unmarshal(tc.Arguments, &args); err != nil {
			return nil, err
		}
	}
	return args, nil
}

// Result creates a successful tool result message for this call.
func (tc ToolCallData) Result(content string) Message {
	return ToolResultMessage(tc.ID, content, false)
}

// ErrorResult creates an error tool result message for this call.
func (tc ToolCallData) ErrorResult(content string) Message {
	return ToolResultMessage(tc.ID, content, true)
}

// ToolCallArgs provides typed access to parsed tool call arguments.
type ToolCallArgs map[string]any

// String returns the string value for the given key.
func (a ToolCallArgs) String(name string) (string, bool) {
	v, ok := a[name]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// Float64 returns the float64 value for the given key.
func (a ToolCallArgs) Float64(name string) (float64, bool) {
	v, ok := a[name]
	if !ok {
		return 0, false
	}
	f, ok := v.(float64)
	return f, ok
}

// Int returns the value for the given key as an int (truncating any decimal).
func (a ToolCallArgs) Int(name string) (int, bool) {
	f, ok := a.Float64(name)
	if !ok {
		return 0, false
	}
	return int(f), ok
}

// Bool returns the boolean value for the given key.
func (a ToolCallArgs) Bool(name string) (bool, bool) {
	v, ok := a[name]
	if !ok {
		return false, false
	}
	b, ok := v.(bool)
	return b, ok
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
		Role:    RoleAssistant,
		Content: []ContentPart{{Kind: ContentText, Text: text}},
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
	params      []Param         // retained from NewTool for runtime validation
}

// ParseArgs unmarshals a tool call's arguments and validates them against
// the parameter definitions (required checks, type checks).
func (td ToolDefinition) ParseArgs(tc ToolCallData) (ToolCallArgs, error) {
	args := make(ToolCallArgs)
	if len(tc.Arguments) > 0 {
		if err := json.Unmarshal(tc.Arguments, &args); err != nil {
			return nil, err
		}
	}
	for _, p := range td.params {
		v, ok := args[p.Name]
		if !ok {
			if p.Required {
				return nil, fmt.Errorf("missing required parameter %q", p.Name)
			}
			continue
		}
		switch p.Type {
		case "string":
			if _, ok := v.(string); !ok {
				return nil, fmt.Errorf("parameter %q: expected string, got %T", p.Name, v)
			}
		case "number", "integer":
			if _, ok := v.(float64); !ok {
				return nil, fmt.Errorf("parameter %q: expected number, got %T", p.Name, v)
			}
		case "boolean":
			if _, ok := v.(bool); !ok {
				return nil, fmt.Errorf("parameter %q: expected boolean, got %T", p.Name, v)
			}
		}
	}
	return args, nil
}

// Param describes a single tool input parameter.
type Param struct {
	Name        string
	Type        string // "string", "number", "integer", "boolean"
	Description string
	Required    bool
}

func newParam(name, typ string, required bool, desc []string) Param {
	p := Param{Name: name, Type: typ, Required: required}
	if len(desc) > 0 {
		p.Description = desc[0]
	}
	return p
}

// StringParam creates a required string parameter.
func StringParam(name string, desc ...string) Param { return newParam(name, "string", true, desc) }

// OptionalStringParam creates an optional string parameter.
func OptionalStringParam(name string, desc ...string) Param {
	return newParam(name, "string", false, desc)
}

// NumberParam creates a required number parameter.
func NumberParam(name string, desc ...string) Param { return newParam(name, "number", true, desc) }

// OptionalNumberParam creates an optional number parameter.
func OptionalNumberParam(name string, desc ...string) Param {
	return newParam(name, "number", false, desc)
}

// IntegerParam creates a required integer parameter.
func IntegerParam(name string, desc ...string) Param { return newParam(name, "integer", true, desc) }

// OptionalIntegerParam creates an optional integer parameter.
func OptionalIntegerParam(name string, desc ...string) Param {
	return newParam(name, "integer", false, desc)
}

// BoolParam creates a required boolean parameter.
func BoolParam(name string, desc ...string) Param { return newParam(name, "boolean", true, desc) }

// OptionalBoolParam creates an optional boolean parameter.
func OptionalBoolParam(name string, desc ...string) Param {
	return newParam(name, "boolean", false, desc)
}

// NewTool creates a ToolDefinition with JSON Schema built from params.
func NewTool(name, description string, params ...Param) ToolDefinition {
	properties := make(map[string]map[string]string, len(params))
	required := make([]string, 0, len(params))
	for _, p := range params {
		prop := map[string]string{"type": p.Type}
		if p.Description != "" {
			prop["description"] = p.Description
		}
		properties[p.Name] = prop
		if p.Required {
			required = append(required, p.Name)
		}
	}
	schema := map[string]any{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}
	raw, _ := json.Marshal(schema)
	return ToolDefinition{
		Name:        name,
		Description: description,
		Parameters:  raw,
		params:      params,
	}
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

const (
	FinishReasonStop          = "stop"
	FinishReasonLength        = "length"
	FinishReasonToolCalls     = "tool_calls"
	FinishReasonContentFilter = "content_filter"
	FinishReasonError         = "error"
)

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
