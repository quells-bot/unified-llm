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
		Tools: []ToolDefinition{NewTool("get_weather", "Get the current weather", StringParam("location"))},
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
		Tools: []ToolDefinition{NewTool("get_weather", "Get the current weather", StringParam("location"))},
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
		Tools: []ToolDefinition{NewTool("my_tool", "A tool")},
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
		Tools: []ToolDefinition{NewTool("my_tool", "A tool")},
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
