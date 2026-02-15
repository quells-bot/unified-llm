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
