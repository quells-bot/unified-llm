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
