package llm

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// toConverseInput translates a Conversation into a Bedrock ConverseInput.
func toConverseInput(conv *Conversation) *bedrockruntime.ConverseInput {
	input := &bedrockruntime.ConverseInput{
		ModelId: strPtr(conv.Model),
	}

	// System prompts
	for _, s := range conv.System {
		input.System = append(input.System, &types.SystemContentBlockMemberText{Value: s})
	}
	// Anthropic: add cache point after last system block
	if isAnthropicModel(conv.Model) && len(input.System) > 0 {
		input.System = append(input.System, &types.SystemContentBlockMemberCachePoint{
			Value: types.CachePointBlock{},
		})
	}

	// Messages
	for _, m := range conv.Messages {
		input.Messages = append(input.Messages, toConverseMessage(m, isAnthropicModel(conv.Model)))
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
			_ = json.Unmarshal(td.Parameters, &doc)
			schema = &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(doc)}
			spec := types.ToolSpecification{
				Name:        strPtr(td.Name),
				InputSchema: schema,
			}
			if td.Description != "" {
				spec.Description = strPtr(td.Description)
			}
			tc.Tools = append(tc.Tools, &types.ToolMemberToolSpec{Value: spec})
		}
		// Anthropic: add cache point after last tool
		if isAnthropicModel(conv.Model) {
			tc.Tools = append(tc.Tools, &types.ToolMemberCachePoint{Value: types.CachePointBlock{}})
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
					Value: types.SpecificToolChoice{Name: strPtr(conv.Config.ToolChoice.ToolName)},
				}
			case ToolChoiceNone:
				tc = nil
			}
		}
		input.ToolConfig = tc
	}

	return input
}

func toConverseMessage(m Message, isAnthropic bool) types.Message {
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
			_ = json.Unmarshal(p.ToolCall.Arguments, &doc)
			msg.Content = append(msg.Content, &types.ContentBlockMemberToolUse{
				Value: types.ToolUseBlock{
					ToolUseId: strPtr(p.ToolCall.ID),
					Name:      strPtr(p.ToolCall.Name),
					Input:     document.NewLazyDocument(doc),
				},
			})
		case ContentToolResult:
			status := types.ToolResultStatusSuccess
			if p.ToolResult.IsError {
				status = types.ToolResultStatusError
			}
			msg.Content = append(msg.Content, &types.ContentBlockMemberToolResult{
				Value: types.ToolResultBlock{
					ToolUseId: strPtr(p.ToolResult.ToolCallID),
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
						Format: types.ImageFormat(strings.TrimPrefix(p.Image.MediaType, "image/")),
						Source: &types.ImageSourceMemberBytes{Value: p.Image.Data},
					},
				})
			}
		case ContentThinking:
			if isAnthropic && p.Thinking != nil {
				msg.Content = append(msg.Content, &types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberReasoningText{
						Value: types.ReasoningTextBlock{
							Text:      strPtr(p.Thinking.Text),
							Signature: strPtr(p.Thinking.Signature),
						},
					},
				})
			}
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
			// Marshal the document Input back to json.RawMessage
			var args json.RawMessage
			if b.Value.Input != nil {
				data, err := b.Value.Input.MarshalSmithyDocument()
				if err == nil {
					args = data
				}
			}
			msg.Content = append(msg.Content, ContentPart{
				Kind: ContentToolCall,
				ToolCall: &ToolCallData{
					ID:        derefStr(b.Value.ToolUseId),
					Name:      derefStr(b.Value.Name),
					Arguments: args,
				},
			})
		case *types.ContentBlockMemberReasoningContent:
			if rt, ok := b.Value.(*types.ReasoningContentBlockMemberReasoningText); ok {
				msg.Content = append(msg.Content, ContentPart{
					Kind: ContentThinking,
					Thinking: &ThinkingData{
						Text:      derefStr(rt.Value.Text),
						Signature: derefStr(rt.Value.Signature),
					},
				})
			}
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

func strPtr(s string) *string { return &s }
