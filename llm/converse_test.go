package llm

import (
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
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
	// System: text + cache point (anthropic model)
	if len(input.System) != 2 {
		t.Fatalf("System len = %d, want 2", len(input.System))
	}
	sysText, ok := input.System[0].(*types.SystemContentBlockMemberText)
	if !ok {
		t.Fatalf("System[0] type = %T", input.System[0])
	}
	if sysText.Value != "Be helpful." {
		t.Errorf("System text = %q", sysText.Value)
	}
	if _, ok := input.System[1].(*types.SystemContentBlockMemberCachePoint); !ok {
		t.Errorf("System[1] should be CachePoint, got %T", input.System[1])
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

func TestToConverseInput_NonAnthropicNoCachePoints(t *testing.T) {
	conv := NewConversation("us.amazon.nova-pro-v1:0",
		WithSystem("Be helpful."),
	)
	conv.Messages = []Message{
		{Role: RoleUser, Content: []ContentPart{{Kind: ContentText, Text: "hello"}}},
	}

	input := toConverseInput(&conv)

	// Non-anthropic: no cache point
	if len(input.System) != 1 {
		t.Fatalf("System len = %d, want 1", len(input.System))
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
	// tool spec + cache point (anthropic)
	if len(input.ToolConfig.Tools) != 2 {
		t.Fatalf("Tools len = %d, want 2", len(input.ToolConfig.Tools))
	}
	spec, ok := input.ToolConfig.Tools[0].(*types.ToolMemberToolSpec)
	if !ok {
		t.Fatalf("Tool[0] type = %T", input.ToolConfig.Tools[0])
	}
	if *spec.Value.Name != "get_weather" {
		t.Errorf("Tool name = %q", *spec.Value.Name)
	}
	if _, ok := input.ToolConfig.Tools[1].(*types.ToolMemberCachePoint); !ok {
		t.Errorf("Tool[1] should be CachePoint, got %T", input.ToolConfig.Tools[1])
	}
}

func TestToConverseInput_ToolChoice(t *testing.T) {
	tool := NewTool("my_tool", "A tool")
	tests := []struct {
		name     string
		choice   ToolChoice
		wantType string
	}{
		{"auto", ToolChoice{Mode: ToolChoiceAuto}, "auto"},
		{"required", ToolChoice{Mode: ToolChoiceRequired}, "any"},
		{"named", ToolChoice{Mode: ToolChoiceNamed, ToolName: "my_tool"}, "tool"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv := NewConversation("us.amazon.nova-pro-v1:0",
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

func TestToConverseInput_ToolChoiceNone(t *testing.T) {
	tool := NewTool("my_tool", "A tool")
	conv := NewConversation("us.amazon.nova-pro-v1:0",
		WithTools(tool),
		WithToolChoice(ToolChoice{Mode: ToolChoiceNone}),
	)
	conv.Messages = []Message{{Role: RoleUser, Content: []ContentPart{{Kind: ContentText, Text: "go"}}}}
	input := toConverseInput(&conv)
	if input.ToolConfig != nil {
		t.Error("expected nil ToolConfig for ToolChoiceNone")
	}
}

func TestToConverseInput_ToolResultMessage(t *testing.T) {
	conv := Conversation{
		Model: "us.amazon.nova-pro-v1:0",
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
	if trBlock.Value.Status != types.ToolResultStatusSuccess {
		t.Errorf("Status = %v", trBlock.Value.Status)
	}
}

func TestToConverseInput_ToolResultError(t *testing.T) {
	conv := Conversation{
		Model: "us.amazon.nova-pro-v1:0",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Kind: ContentText, Text: "go"}}},
			{Role: RoleAssistant, Content: []ContentPart{{
				Kind:     ContentToolCall,
				ToolCall: &ToolCallData{ID: "call-1", Name: "foo", Arguments: []byte(`{}`)},
			}}},
			ToolResultMessage("call-1", "error happened", true),
		},
	}
	input := toConverseInput(&conv)
	trBlock, ok := input.Messages[2].Content[0].(*types.ContentBlockMemberToolResult)
	if !ok {
		t.Fatalf("content type = %T", input.Messages[2].Content[0])
	}
	if trBlock.Value.Status != types.ToolResultStatusError {
		t.Errorf("Status = %v, want error", trBlock.Value.Status)
	}
}

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
						Input:     document.NewLazyDocument(map[string]any{"location": "SF"}),
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
	// Check arguments
	var args map[string]any
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if args["location"] != "SF" {
		t.Errorf("location = %v", args["location"])
	}
}

func TestFromConverseOutput_CacheTokens(t *testing.T) {
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

func TestFromConverseOutput_StopReasons(t *testing.T) {
	tests := []struct {
		stop types.StopReason
		want FinishReason
	}{
		{types.StopReasonEndTurn, FinishReasonStop},
		{types.StopReasonStopSequence, FinishReasonStop},
		{types.StopReasonMaxTokens, FinishReasonLength},
		{types.StopReasonToolUse, FinishReasonToolUse},
		{types.StopReasonContentFiltered, FinishReasonContentFilter},
		{types.StopReasonGuardrailIntervened, FinishReasonContentFilter},
	}
	for _, tt := range tests {
		t.Run(string(tt.stop), func(t *testing.T) {
			out := &bedrockruntime.ConverseOutput{
				Output: &types.ConverseOutputMemberMessage{
					Value: types.Message{
						Role:    types.ConversationRoleAssistant,
						Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "x"}},
					},
				},
				StopReason: tt.stop,
				Usage:      &types.TokenUsage{InputTokens: int32Ptr(1), OutputTokens: int32Ptr(1), TotalTokens: int32Ptr(2)},
			}
			_, _, reason, err := fromConverseOutput(out)
			if err != nil {
				t.Fatal(err)
			}
			if reason != tt.want {
				t.Errorf("got %q, want %q", reason, tt.want)
			}
		})
	}
}
