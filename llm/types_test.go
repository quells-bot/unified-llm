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
