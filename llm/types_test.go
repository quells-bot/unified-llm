package llm

import (
	"encoding/json"
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

func TestStringParam(t *testing.T) {
	p := StringParam("location", "The city")
	if p.Name != "location" || p.Type != "string" || p.Description != "The city" || !p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestStringParamNoDescription(t *testing.T) {
	p := StringParam("location")
	if p.Description != "" {
		t.Errorf("expected empty description, got %q", p.Description)
	}
}

func TestOptionalStringParam(t *testing.T) {
	p := OptionalStringParam("name")
	if p.Required {
		t.Error("expected Required=false")
	}
	if p.Type != "string" {
		t.Errorf("expected type string, got %q", p.Type)
	}
}

func TestNumberParam(t *testing.T) {
	p := NumberParam("count")
	if p.Type != "number" || !p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestOptionalNumberParam(t *testing.T) {
	p := OptionalNumberParam("count")
	if p.Type != "number" || p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestIntegerParam(t *testing.T) {
	p := IntegerParam("count")
	if p.Type != "integer" || !p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestOptionalIntegerParam(t *testing.T) {
	p := OptionalIntegerParam("count")
	if p.Type != "integer" || p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestBoolParam(t *testing.T) {
	p := BoolParam("verbose")
	if p.Type != "boolean" || !p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestOptionalBoolParam(t *testing.T) {
	p := OptionalBoolParam("verbose")
	if p.Type != "boolean" || p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestNewToolMixedParams(t *testing.T) {
	tool := NewTool("get_weather", "Get the current weather",
		StringParam("location", "The city"),
		OptionalNumberParam("days", "Forecast days"),
	)
	if tool.Name != "get_weather" {
		t.Errorf("Name = %q", tool.Name)
	}
	if tool.Description != "Get the current weather" {
		t.Errorf("Description = %q", tool.Description)
	}
	want := `{"type":"object","properties":{"location":{"type":"string","description":"The city"},"days":{"type":"number","description":"Forecast days"}},"required":["location"]}`
	assertJSONEqual(t, tool.Parameters, []byte(want))
}

func TestNewToolNoParams(t *testing.T) {
	tool := NewTool("noop", "Does nothing")
	want := `{"type":"object","properties":{},"required":[]}`
	assertJSONEqual(t, tool.Parameters, []byte(want))
}

func TestNewToolNoDescription(t *testing.T) {
	tool := NewTool("ping", "Ping a host", StringParam("host"))
	want := `{"type":"object","properties":{"host":{"type":"string"}},"required":["host"]}`
	assertJSONEqual(t, tool.Parameters, []byte(want))
}

func TestToolCallDataParseArgs(t *testing.T) {
	tc := ToolCallData{
		ID:        "call-1",
		Name:      "test",
		Arguments: json.RawMessage(`{"name":"alice","age":30,"score":9.5,"active":true}`),
	}
	args, err := tc.ParseArgs()
	if err != nil {
		t.Fatalf("ParseArgs: %v", err)
	}
	if s, ok := args.String("name"); !ok || s != "alice" {
		t.Errorf("String(name) = %q, %v", s, ok)
	}
	if f, ok := args.Float64("age"); !ok || f != 30 {
		t.Errorf("Float64(age) = %v, %v", f, ok)
	}
	if i, ok := args.Int("age"); !ok || i != 30 {
		t.Errorf("Int(age) = %v, %v", i, ok)
	}
	if f, ok := args.Float64("score"); !ok || f != 9.5 {
		t.Errorf("Float64(score) = %v, %v", f, ok)
	}
	if b, ok := args.Bool("active"); !ok || !b {
		t.Errorf("Bool(active) = %v, %v", b, ok)
	}
}

func TestToolCallDataParseArgsEmpty(t *testing.T) {
	tc := ToolCallData{ID: "call-2", Name: "noop"}
	args, err := tc.ParseArgs()
	if err != nil {
		t.Fatalf("ParseArgs: %v", err)
	}
	if len(args) != 0 {
		t.Errorf("expected empty args, got %v", args)
	}
}

func TestToolCallDataParseArgsInvalid(t *testing.T) {
	tc := ToolCallData{
		ID:        "call-3",
		Name:      "bad",
		Arguments: json.RawMessage(`not json`),
	}
	_, err := tc.ParseArgs()
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestToolCallArgsMissingKey(t *testing.T) {
	args := ToolCallArgs{"x": float64(1)}
	if _, ok := args.String("missing"); ok {
		t.Error("expected ok=false for missing key")
	}
	if _, ok := args.Float64("missing"); ok {
		t.Error("expected ok=false for missing key")
	}
	if _, ok := args.Int("missing"); ok {
		t.Error("expected ok=false for missing key")
	}
	if _, ok := args.Bool("missing"); ok {
		t.Error("expected ok=false for missing key")
	}
}

func TestToolCallArgsWrongType(t *testing.T) {
	args := ToolCallArgs{"val": "text"}
	if _, ok := args.Float64("val"); ok {
		t.Error("expected ok=false for wrong type")
	}
	if _, ok := args.Int("val"); ok {
		t.Error("expected ok=false for wrong type")
	}
	if _, ok := args.Bool("val"); ok {
		t.Error("expected ok=false for wrong type")
	}
}

func TestToolCallDataResult(t *testing.T) {
	tc := ToolCallData{ID: "call-10", Name: "test"}
	m := tc.Result(`{"ok":true}`)
	if m.Role != RoleTool {
		t.Errorf("got role %q, want %q", m.Role, RoleTool)
	}
	tr := m.Content[0].ToolResult
	if tr.ToolCallID != "call-10" || tr.Content != `{"ok":true}` || tr.IsError {
		t.Errorf("unexpected result: %+v", tr)
	}
}

func TestToolCallDataErrorResult(t *testing.T) {
	tc := ToolCallData{ID: "call-11", Name: "test"}
	m := tc.ErrorResult(`{"error":"boom"}`)
	tr := m.Content[0].ToolResult
	if tr.ToolCallID != "call-11" || !tr.IsError {
		t.Errorf("unexpected error result: %+v", tr)
	}
}

func TestToolDefinitionParseArgs(t *testing.T) {
	tool := NewTool("order", "Get order",
		IntegerParam("user_id"),
		StringParam("note"),
		OptionalBoolParam("verbose"),
	)
	tc := ToolCallData{
		ID:        "call-20",
		Name:      "order",
		Arguments: json.RawMessage(`{"user_id":42,"note":"rush"}`),
	}
	args, err := tool.ParseArgs(tc)
	if err != nil {
		t.Fatalf("ParseArgs: %v", err)
	}
	if id, ok := args.Int("user_id"); !ok || id != 42 {
		t.Errorf("Int(user_id) = %v, %v", id, ok)
	}
	if s, ok := args.String("note"); !ok || s != "rush" {
		t.Errorf("String(note) = %v, %v", s, ok)
	}
}

func TestToolDefinitionParseArgsMissingRequired(t *testing.T) {
	tool := NewTool("order", "Get order",
		IntegerParam("user_id"),
		IntegerParam("order_id"),
	)
	tc := ToolCallData{
		ID:        "call-21",
		Name:      "order",
		Arguments: json.RawMessage(`{"user_id":1}`),
	}
	_, err := tool.ParseArgs(tc)
	if err == nil {
		t.Fatal("expected error for missing required param")
	}
	want := `missing required parameter "order_id"`
	if err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

func TestToolDefinitionParseArgsWrongType(t *testing.T) {
	tool := NewTool("order", "Get order", IntegerParam("user_id"))
	tc := ToolCallData{
		ID:        "call-22",
		Name:      "order",
		Arguments: json.RawMessage(`{"user_id":"not_a_number"}`),
	}
	_, err := tool.ParseArgs(tc)
	if err == nil {
		t.Fatal("expected error for wrong type")
	}
}

func TestToolDefinitionParseArgsMissingOptionalOK(t *testing.T) {
	tool := NewTool("order", "Get order",
		IntegerParam("user_id"),
		OptionalStringParam("note"),
	)
	tc := ToolCallData{
		ID:        "call-23",
		Name:      "order",
		Arguments: json.RawMessage(`{"user_id":1}`),
	}
	args, err := tool.ParseArgs(tc)
	if err != nil {
		t.Fatalf("ParseArgs: %v", err)
	}
	if _, ok := args.String("note"); ok {
		t.Error("expected ok=false for absent optional param")
	}
}

func TestToolDefinitionParseArgsNoParams(t *testing.T) {
	tool := NewTool("ping", "Ping")
	tc := ToolCallData{ID: "call-24", Name: "ping"}
	args, err := tool.ParseArgs(tc)
	if err != nil {
		t.Fatalf("ParseArgs: %v", err)
	}
	if len(args) != 0 {
		t.Errorf("expected empty args, got %v", args)
	}
}
