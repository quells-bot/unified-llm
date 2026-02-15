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
		Tools:    []ToolDefinition{NewTool("get_weather", "Get weather", StringParam("location"))},
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
		Tools: []ToolDefinition{NewTool("get_weather", "Get weather", StringParam("location"))},
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
		Model:      "us.amazon.nova-pro-v1:0",
		Messages:   []Message{UserMessage("Do something")},
		Tools:      []ToolDefinition{NewTool("my_tool", "A tool")},
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
		Model:      "us.amazon.nova-pro-v1:0",
		Messages:   []Message{UserMessage("Do something")},
		Tools:      []ToolDefinition{NewTool("my_tool", "A tool")},
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
