package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAIProvider implements Provider using the OpenAI-compatible chat
// completions API (e.g. llama.cpp, vLLM, Ollama, or OpenAI itself).
type OpenAIProvider struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// OpenAIOption configures an OpenAIProvider.
type OpenAIOption func(*OpenAIProvider)

// WithAPIKey sets the Bearer token for authenticated endpoints.
func WithAPIKey(key string) OpenAIOption {
	return func(p *OpenAIProvider) { p.apiKey = key }
}

// WithHTTPClient overrides the default HTTP client.
func WithHTTPClient(c *http.Client) OpenAIOption {
	return func(p *OpenAIProvider) { p.httpClient = c }
}

// NewOpenAIProvider creates a Provider that calls POST {baseURL}/v1/chat/completions.
func NewOpenAIProvider(baseURL string, opts ...OpenAIOption) *OpenAIProvider {
	p := &OpenAIProvider{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: http.DefaultClient,
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

// Send translates the conversation to the OpenAI chat completions format,
// makes the HTTP request, and translates the response back.
func (p *OpenAIProvider) Send(ctx context.Context, conv *Conversation) (*Response, error) {
	reqBody := toOpenAIRequest(conv)
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, &Error{Kind: ErrConfig, Message: "failed to marshal request", Cause: err}
	}

	url := p.baseURL + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, &Error{Kind: ErrConfig, Message: "failed to create request", Cause: err}
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	httpResp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, &Error{Kind: ErrServer, Message: err.Error(), Cause: err}
	}
	defer httpResp.Body.Close()

	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, &Error{Kind: ErrServer, Message: "failed to read response", Cause: err}
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, classifyOpenAIError(httpResp.StatusCode, body)
	}

	var chatResp chatCompletionResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, &Error{Kind: ErrServer, Message: "failed to decode response", Cause: err}
	}

	return fromOpenAIResponse(chatResp)
}

// --- request/response wire types (unexported) ---

type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Tools       []chatTool    `json:"tools,omitempty"`
	ToolChoice  any           `json:"tool_choice,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	Stop        []string      `json:"stop,omitempty"`
}

type chatMessage struct {
	Role             string         `json:"role"`
	Content          *string        `json:"content"`                     // pointer so we can send null
	ReasoningContent string         `json:"reasoning_content,omitempty"` // llama.cpp extended field
	ToolCalls        []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatTool struct {
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type chatCompletionResponse struct {
	Choices []chatChoice `json:"choices"`
	Usage   *chatUsage   `json:"usage,omitempty"`
}

type chatChoice struct {
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type chatErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// --- translation ---

func toOpenAIRequest(conv *Conversation) chatCompletionRequest {
	req := chatCompletionRequest{
		Model:       conv.Model,
		MaxTokens:   conv.Config.MaxTokens,
		Temperature: conv.Config.Temperature,
		TopP:        conv.Config.TopP,
		Stop:        conv.Config.StopSequences,
	}

	// System prompt as a single system message.
	if len(conv.System) > 0 {
		text := strings.Join(conv.System, "\n\n")
		req.Messages = append(req.Messages, chatMessage{
			Role:    "system",
			Content: &text,
		})
	}

	// Conversation messages.
	for _, m := range conv.Messages {
		switch m.Role {
		case RoleUser:
			text := m.Text()
			req.Messages = append(req.Messages, chatMessage{
				Role:    "user",
				Content: &text,
			})

		case RoleAssistant:
			cm := chatMessage{Role: "assistant"}
			// Collect text content.
			text := m.Text()
			if text != "" {
				cm.Content = &text
			}
			// Collect tool calls.
			for _, tc := range m.ToolCalls() {
				cm.ToolCalls = append(cm.ToolCalls, chatToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: chatFunctionCall{
						Name:      tc.Name,
						Arguments: string(tc.Arguments),
					},
				})
			}
			req.Messages = append(req.Messages, cm)

		case RoleTool:
			for _, p := range m.Content {
				if p.Kind == ContentToolResult && p.ToolResult != nil {
					content := p.ToolResult.Content
					req.Messages = append(req.Messages, chatMessage{
						Role:       "tool",
						Content:    &content,
						ToolCallID: p.ToolResult.ToolCallID,
					})
				}
			}
		}
	}

	// Tools.
	for _, td := range conv.Tools {
		req.Tools = append(req.Tools, chatTool{
			Type: "function",
			Function: chatFunction{
				Name:        td.Name,
				Description: td.Description,
				Parameters:  td.Parameters,
			},
		})
	}

	// Tool choice.
	if conv.Config.ToolChoice != nil {
		switch conv.Config.ToolChoice.Mode {
		case ToolChoiceAuto:
			req.ToolChoice = "auto"
		case ToolChoiceNone:
			req.ToolChoice = "none"
		case ToolChoiceRequired:
			req.ToolChoice = "required"
		case ToolChoiceNamed:
			req.ToolChoice = map[string]any{
				"type":     "function",
				"function": map[string]any{"name": conv.Config.ToolChoice.ToolName},
			}
		}
	}

	return req
}

func fromOpenAIResponse(resp chatCompletionResponse) (*Response, error) {
	if len(resp.Choices) == 0 {
		return nil, &Error{Kind: ErrServer, Message: "no choices in response"}
	}

	choice := resp.Choices[0]
	msg := Message{Role: RoleAssistant}

	// Reasoning content (e.g. llama.cpp reasoning_content field).
	if choice.Message.ReasoningContent != "" {
		msg.Content = append(msg.Content, ContentPart{
			Kind:     ContentThinking,
			Thinking: &ThinkingData{Text: choice.Message.ReasoningContent},
		})
	}

	// Text content.
	if choice.Message.Content != nil && *choice.Message.Content != "" {
		msg.Content = append(msg.Content, ContentPart{
			Kind: ContentText,
			Text: *choice.Message.Content,
		})
	}

	// Tool calls.
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

	// Finish reason.
	reason := mapOpenAIFinishReason(choice.FinishReason)

	// Usage.
	usage := Usage{}
	if resp.Usage != nil {
		usage.InputTokens = resp.Usage.PromptTokens
		usage.OutputTokens = resp.Usage.CompletionTokens
	}

	return &Response{
		Message:      msg,
		FinishReason: reason,
		Usage:        usage,
	}, nil
}

func mapOpenAIFinishReason(reason string) FinishReason {
	switch reason {
	case "stop":
		return FinishReasonStop
	case "length":
		return FinishReasonLength
	case "tool_calls":
		return FinishReasonToolUse
	case "content_filter":
		return FinishReasonContentFilter
	default:
		return FinishReason(reason)
	}
}

func classifyOpenAIError(statusCode int, body []byte) error {
	var errResp chatErrorResponse
	_ = json.Unmarshal(body, &errResp) // best-effort parse
	msg := errResp.Error.Message
	if msg == "" {
		msg = fmt.Sprintf("HTTP %d", statusCode)
	}

	var kind ErrorKind
	switch statusCode {
	case 400:
		lower := strings.ToLower(msg)
		switch {
		case strings.Contains(lower, "context length") || strings.Contains(lower, "too many tokens"):
			kind = ErrContextLength
		default:
			kind = ErrInvalidRequest
		}
	case 401, 403:
		kind = ErrAuthentication
	case 404:
		kind = ErrNotFound
	case 429:
		kind = ErrRateLimit
	default:
		kind = ErrServer
	}

	return &Error{
		Kind:    kind,
		Message: msg,
		Cause:   fmt.Errorf("HTTP %d: %s", statusCode, msg),
	}
}
