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
