package llm

import "encoding/json"

// AnthropicAdapter translates between unified types and the Anthropic Messages API format.
type AnthropicAdapter struct{}

// NewAnthropicAdapter creates a new AnthropicAdapter.
func NewAnthropicAdapter() *AnthropicAdapter {
	return &AnthropicAdapter{}
}

func (a *AnthropicAdapter) Provider() string { return "anthropic" }

// --- Anthropic request types ---

type anthropicRequest struct {
	AnthropicVersion string              `json:"anthropic_version"`
	MaxTokens        int                 `json:"max_tokens"`
	System           []anthropicContent  `json:"system,omitempty"`
	Messages         []anthropicMessage  `json:"messages"`
	Tools            []anthropicTool     `json:"tools,omitempty"`
	ToolChoice       any                 `json:"tool_choice,omitempty"`
	Temperature      *float64            `json:"temperature,omitempty"`
	TopP             *float64            `json:"top_p,omitempty"`
	StopSequences    []string            `json:"stop_sequences,omitempty"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type         string            `json:"type"`
	Text         string            `json:"text,omitempty"`
	ID           string            `json:"id,omitempty"`
	Name         string            `json:"name,omitempty"`
	Input        json.RawMessage   `json:"input,omitempty"`
	ToolUseID    string            `json:"tool_use_id,omitempty"`
	Content      string            `json:"content,omitempty"`
	IsError      *bool             `json:"is_error,omitempty"`
	Thinking     string            `json:"thinking,omitempty"`
	Signature    string            `json:"signature,omitempty"`
	CacheControl *cacheControl     `json:"cache_control,omitempty"`
}

type cacheControl struct {
	Type string `json:"type"`
}

type anthropicTool struct {
	Name         string          `json:"name"`
	Description  string          `json:"description"`
	InputSchema  json.RawMessage `json:"input_schema"`
	CacheControl *cacheControl   `json:"cache_control,omitempty"`
}

func (a *AnthropicAdapter) BuildInvokeInput(req *Request) (*InvokeInput, error) {
	ar := anthropicRequest{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        4096,
	}

	if req.MaxTokens != nil {
		ar.MaxTokens = *req.MaxTokens
	}
	ar.Temperature = req.Temperature
	ar.TopP = req.TopP
	if len(req.StopSequences) > 0 {
		ar.StopSequences = req.StopSequences
	}

	// Extract system messages
	var nonSystem []Message
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			for _, p := range m.Content {
				if p.Kind == ContentText {
					ar.System = append(ar.System, anthropicContent{Type: "text", Text: p.Text})
				}
			}
		} else {
			nonSystem = append(nonSystem, m)
		}
	}

	// Auto-inject cache_control on last system block
	if len(ar.System) > 0 {
		ar.System[len(ar.System)-1].CacheControl = &cacheControl{Type: "ephemeral"}
	}

	// Translate messages
	for _, m := range nonSystem {
		am := a.translateMessage(m)
		// Enforce strict user/assistant alternation: merge consecutive same-role messages
		if len(ar.Messages) > 0 && ar.Messages[len(ar.Messages)-1].Role == am.Role {
			ar.Messages[len(ar.Messages)-1].Content = append(ar.Messages[len(ar.Messages)-1].Content, am.Content...)
		} else {
			ar.Messages = append(ar.Messages, am)
		}
	}

	// Translate tools
	if len(req.Tools) > 0 {
		for _, td := range req.Tools {
			ar.Tools = append(ar.Tools, anthropicTool{
				Name:        td.Name,
				Description: td.Description,
				InputSchema: td.Parameters,
			})
		}
		// Auto-inject cache_control on last tool
		ar.Tools[len(ar.Tools)-1].CacheControl = &cacheControl{Type: "ephemeral"}
	}

	// Translate tool choice
	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case ToolChoiceAuto:
			ar.ToolChoice = map[string]string{"type": "auto"}
		case ToolChoiceNone:
			// Omit tools entirely
			ar.Tools = nil
			ar.ToolChoice = nil
		case ToolChoiceRequired:
			ar.ToolChoice = map[string]string{"type": "any"}
		case ToolChoiceNamed:
			ar.ToolChoice = map[string]string{"type": "tool", "name": req.ToolChoice.ToolName}
		}
	}

	// Merge provider options
	if opts, ok := req.ProviderOptions["anthropic"]; ok {
		if m, ok := opts.(map[string]any); ok {
			_ = m // provider options are merged at JSON level below
		}
	}

	body, err := json.Marshal(ar)
	if err != nil {
		return nil, &Error{Kind: ErrAdapter, Provider: "anthropic", Message: "failed to marshal request", Cause: err}
	}

	return &InvokeInput{
		ModelID:     req.Model,
		Body:        body,
		ContentType: "application/json",
		Accept:      "application/json",
	}, nil
}

func (a *AnthropicAdapter) translateMessage(m Message) anthropicMessage {
	am := anthropicMessage{}

	switch m.Role {
	case RoleUser:
		am.Role = "user"
	case RoleAssistant:
		am.Role = "assistant"
	case RoleTool:
		am.Role = "user" // Tool results are sent as user messages in Anthropic format
	}

	for _, p := range m.Content {
		switch p.Kind {
		case ContentText:
			am.Content = append(am.Content, anthropicContent{Type: "text", Text: p.Text})
		case ContentToolCall:
			am.Content = append(am.Content, anthropicContent{
				Type:  "tool_use",
				ID:    p.ToolCall.ID,
				Name:  p.ToolCall.Name,
				Input: p.ToolCall.Arguments,
			})
		case ContentToolResult:
			am.Content = append(am.Content, anthropicContent{
				Type:      "tool_result",
				ToolUseID: p.ToolResult.ToolCallID,
				Content:   p.ToolResult.Content,
			})
		case ContentThinking:
			am.Content = append(am.Content, anthropicContent{
				Type:      "thinking",
				Thinking:  p.Thinking.Text,
				Signature: p.Thinking.Signature,
			})
		case ContentImage:
			// Image support can be added later
		}
	}

	return am
}

// ParseResponse is implemented in the next task.
func (a *AnthropicAdapter) ParseResponse(body []byte, req *Request) (*Response, error) {
	return nil, &Error{Kind: ErrAdapter, Provider: "anthropic", Message: "not implemented"}
}
