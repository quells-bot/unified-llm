package llm

import "encoding/json"

// OpenAIAdapter translates between unified types and the OpenAI Chat Completions format.
type OpenAIAdapter struct{}

// NewOpenAIAdapter creates a new OpenAIAdapter.
func NewOpenAIAdapter() *OpenAIAdapter {
	return &OpenAIAdapter{}
}

func (a *OpenAIAdapter) Provider() string { return "openai" }

// --- OpenAI request types ---

type openaiRequest struct {
	Model           string          `json:"model"`
	Messages        []openaiMessage `json:"messages"`
	Tools           []openaiTool    `json:"tools,omitempty"`
	ToolChoice      any             `json:"tool_choice,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	MaxTokens       *int            `json:"max_tokens,omitempty"`
	Stop            []string        `json:"stop,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
}

type openaiMessage struct {
	Role       string           `json:"role"`
	Content    any              `json:"content"` // string or null
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openaiToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openaiToolFunction `json:"function"`
}

type openaiToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openaiTool struct {
	Type     string        `json:"type"`
	Function openaiToolDef `json:"function"`
}

type openaiToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

func (a *OpenAIAdapter) BuildInvokeInput(req *Request) (*InvokeInput, error) {
	or := openaiRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
	}
	if len(req.StopSequences) > 0 {
		or.Stop = req.StopSequences
	}
	if req.ReasoningEffort != "" {
		or.ReasoningEffort = req.ReasoningEffort
	}

	// Translate messages
	for _, m := range req.Messages {
		om := a.translateMessage(m)
		or.Messages = append(or.Messages, om)
	}

	// Translate tools
	if len(req.Tools) > 0 {
		for _, td := range req.Tools {
			or.Tools = append(or.Tools, openaiTool{
				Type: "function",
				Function: openaiToolDef{
					Name:        td.Name,
					Description: td.Description,
					Parameters:  td.Parameters,
				},
			})
		}
	}

	// Translate tool choice
	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case ToolChoiceAuto:
			or.ToolChoice = "auto"
		case ToolChoiceNone:
			or.ToolChoice = "none"
		case ToolChoiceRequired:
			or.ToolChoice = "required"
		case ToolChoiceNamed:
			or.ToolChoice = map[string]any{
				"type":     "function",
				"function": map[string]string{"name": req.ToolChoice.ToolName},
			}
		}
	}

	body, err := json.Marshal(or)
	if err != nil {
		return nil, &Error{Kind: ErrAdapter, Provider: "openai", Message: "failed to marshal request", Cause: err}
	}

	return &InvokeInput{
		ModelID:     req.Model,
		Body:        body,
		ContentType: "application/json",
		Accept:      "application/json",
	}, nil
}

func (a *OpenAIAdapter) translateMessage(m Message) openaiMessage {
	om := openaiMessage{}

	switch m.Role {
	case RoleSystem:
		om.Role = "system"
		om.Content = m.Text()
	case RoleUser:
		om.Role = "user"
		om.Content = m.Text()
	case RoleAssistant:
		om.Role = "assistant"
		// Check for tool calls
		var toolCalls []openaiToolCall
		text := m.Text()
		for _, p := range m.Content {
			if p.Kind == ContentToolCall && p.ToolCall != nil {
				toolCalls = append(toolCalls, openaiToolCall{
					ID:   p.ToolCall.ID,
					Type: "function",
					Function: openaiToolFunction{
						Name:      p.ToolCall.Name,
						Arguments: string(p.ToolCall.Arguments),
					},
				})
			}
		}
		if len(toolCalls) > 0 {
			om.ToolCalls = toolCalls
			// content is null when there are tool calls and no text
			if text == "" {
				om.Content = nil
			} else {
				om.Content = text
			}
		} else {
			om.Content = text
		}
	case RoleTool:
		om.Role = "tool"
		om.ToolCallID = m.ToolCallID
		// Get the content from tool result
		for _, p := range m.Content {
			if p.Kind == ContentToolResult && p.ToolResult != nil {
				om.Content = p.ToolResult.Content
				break
			}
		}
	}

	return om
}

// --- OpenAI response types ---

type openaiResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Choices []openaiChoice `json:"choices"`
	Usage   openaiUsage    `json:"usage"`
}

type openaiChoice struct {
	Index        int           `json:"index"`
	Message      openaiRespMsg `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openaiRespMsg struct {
	Role      string           `json:"role"`
	Content   *string          `json:"content"`
	ToolCalls []openaiToolCall `json:"tool_calls,omitempty"`
}

type openaiUsage struct {
	PromptTokens      int                      `json:"prompt_tokens"`
	CompletionTokens  int                      `json:"completion_tokens"`
	PromptDetails     *openaiPromptDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionDetails *openaiCompletionDetails `json:"completion_tokens_details,omitempty"`
}

type openaiPromptDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type openaiCompletionDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

func (a *OpenAIAdapter) ParseResponse(body []byte, req *Request) (*Response, error) {
	var or openaiResponse
	if err := json.Unmarshal(body, &or); err != nil {
		return nil, &Error{Kind: ErrAdapter, Provider: "openai", Message: "failed to unmarshal response", Cause: err, Raw: body}
	}

	if len(or.Choices) == 0 {
		return nil, &Error{Kind: ErrAdapter, Provider: "openai", Message: "response has no choices", Raw: body}
	}

	choice := or.Choices[0]
	msg := Message{Role: RoleAssistant}

	if choice.Message.Content != nil && *choice.Message.Content != "" {
		msg.Content = append(msg.Content, ContentPart{Kind: ContentText, Text: *choice.Message.Content})
	}

	for _, tc := range choice.Message.ToolCalls {
		msg.Content = append(msg.Content, ContentPart{
			Kind: ContentToolCall,
			ToolCall: &ToolCallData{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			},
		})
	}

	usage := Usage{
		InputTokens:  or.Usage.PromptTokens,
		OutputTokens: or.Usage.CompletionTokens,
	}
	if or.Usage.PromptDetails != nil {
		usage.CacheReadTokens = or.Usage.PromptDetails.CachedTokens
	}
	if or.Usage.CompletionDetails != nil {
		usage.ReasoningTokens = or.Usage.CompletionDetails.ReasoningTokens
	}

	return &Response{
		ID:           or.ID,
		Model:        or.Model,
		Provider:     "openai",
		Message:      msg,
		FinishReason: mapOpenAIFinishReason(choice.FinishReason),
		Usage:        usage,
		Raw:          body,
	}, nil
}

func mapOpenAIFinishReason(raw string) FinishReason {
	reason := raw // OpenAI values mostly match unified values
	switch raw {
	case FinishReasonStop,
		FinishReasonLength,
		FinishReasonToolCalls,
		FinishReasonContentFilter:
		// already correct
	default:
		reason = raw
	}
	return FinishReason{Reason: reason, Raw: raw}
}
