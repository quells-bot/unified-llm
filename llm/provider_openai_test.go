package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// newTestOpenAIServer creates an httptest server that captures the request body
// and returns the given response JSON with the given status code.
func newTestOpenAIServer(t *testing.T, statusCode int, respBody any) (*httptest.Server, *[]byte) {
	t.Helper()
	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured = body
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		json.NewEncoder(w).Encode(respBody)
	}))
	t.Cleanup(srv.Close)
	return srv, &captured
}

func TestOpenAIProvider_SimpleText(t *testing.T) {
	resp := chatCompletionResponse{
		Choices: []chatChoice{{
			Message:      chatMessage{Role: "assistant", Content: strPtr("Hello!")},
			FinishReason: "stop",
		}},
		Usage: &chatUsage{PromptTokens: 8, CompletionTokens: 3},
	}
	srv, _ := newTestOpenAIServer(t, 200, resp)

	provider := NewOpenAIProvider(srv.URL)
	conv := NewConversation("llama3",
		WithSystem("Be helpful."),
		WithMaxTokens(256),
	)
	conv.Messages = []Message{UserMessage("hi")}

	result, err := provider.Send(context.Background(), &conv)
	if err != nil {
		t.Fatal(err)
	}
	if result.Message.Text() != "Hello!" {
		t.Errorf("Text = %q", result.Message.Text())
	}
	if result.FinishReason != FinishReasonStop {
		t.Errorf("FinishReason = %q", result.FinishReason)
	}
	if result.Usage.InputTokens != 8 {
		t.Errorf("InputTokens = %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 3 {
		t.Errorf("OutputTokens = %d", result.Usage.OutputTokens)
	}
}

func TestOpenAIProvider_ToolCallResponse(t *testing.T) {
	resp := chatCompletionResponse{
		Choices: []chatChoice{{
			Message: chatMessage{
				Role: "assistant",
				ToolCalls: []chatToolCall{{
					ID:   "call_123",
					Type: "function",
					Function: chatFunctionCall{
						Name:      "get_weather",
						Arguments: `{"city":"Portland"}`,
					},
				}},
			},
			FinishReason: "tool_calls",
		}},
		Usage: &chatUsage{PromptTokens: 20, CompletionTokens: 10},
	}
	srv, _ := newTestOpenAIServer(t, 200, resp)

	provider := NewOpenAIProvider(srv.URL)
	conv := NewConversation("llama3")
	conv.Messages = []Message{UserMessage("what's the weather?")}

	result, err := provider.Send(context.Background(), &conv)
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != FinishReasonToolUse {
		t.Errorf("FinishReason = %q, want tool_use", result.FinishReason)
	}

	calls := result.Message.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(calls))
	}
	if calls[0].ID != "call_123" {
		t.Errorf("ID = %q", calls[0].ID)
	}
	if calls[0].Name != "get_weather" {
		t.Errorf("Name = %q", calls[0].Name)
	}
	args, err := calls[0].ParseArgs()
	if err != nil {
		t.Fatal(err)
	}
	city, _ := args.String("city")
	if city != "Portland" {
		t.Errorf("city = %q", city)
	}
}

func TestOpenAIProvider_RequestFormat(t *testing.T) {
	resp := chatCompletionResponse{
		Choices: []chatChoice{{
			Message:      chatMessage{Role: "assistant", Content: strPtr("ok")},
			FinishReason: "stop",
		}},
	}
	srv, captured := newTestOpenAIServer(t, 200, resp)

	provider := NewOpenAIProvider(srv.URL)
	weatherTool := NewTool("get_weather", "Get weather for a city", StringParam("city", "The city name"))
	conv := NewConversation("llama3",
		WithSystem("You are helpful."),
		WithTools(weatherTool),
		WithMaxTokens(100),
		WithTemperature(0.7),
	)
	conv.Messages = []Message{UserMessage("hello")}

	_, err := provider.Send(context.Background(), &conv)
	if err != nil {
		t.Fatal(err)
	}

	var req map[string]any
	if err := json.Unmarshal(*captured, &req); err != nil {
		t.Fatal(err)
	}

	if req["model"] != "llama3" {
		t.Errorf("model = %v", req["model"])
	}
	if req["max_tokens"] != float64(100) {
		t.Errorf("max_tokens = %v", req["max_tokens"])
	}

	msgs, ok := req["messages"].([]any)
	if !ok {
		t.Fatal("messages not an array")
	}
	// Should have system + user = 2 messages
	if len(msgs) != 2 {
		t.Fatalf("messages len = %d, want 2", len(msgs))
	}
	sysMsg := msgs[0].(map[string]any)
	if sysMsg["role"] != "system" {
		t.Errorf("messages[0].role = %v", sysMsg["role"])
	}
	if sysMsg["content"] != "You are helpful." {
		t.Errorf("messages[0].content = %v", sysMsg["content"])
	}

	tools, ok := req["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", req["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tool.type = %v", tool["type"])
	}
}

func TestOpenAIProvider_ToolResultRequest(t *testing.T) {
	resp := chatCompletionResponse{
		Choices: []chatChoice{{
			Message:      chatMessage{Role: "assistant", Content: strPtr("It's sunny.")},
			FinishReason: "stop",
		}},
	}
	srv, captured := newTestOpenAIServer(t, 200, resp)

	provider := NewOpenAIProvider(srv.URL)
	conv := NewConversation("llama3")
	conv.Messages = []Message{
		UserMessage("weather?"),
		{
			Role: RoleAssistant,
			Content: []ContentPart{{
				Kind: ContentToolCall,
				ToolCall: &ToolCallData{
					ID:        "call_1",
					Name:      "get_weather",
					Arguments: json.RawMessage(`{"city":"Portland"}`),
				},
			}},
		},
		ToolResultMessage("call_1", "72F and sunny", false),
	}

	_, err := provider.Send(context.Background(), &conv)
	if err != nil {
		t.Fatal(err)
	}

	var req map[string]any
	if err := json.Unmarshal(*captured, &req); err != nil {
		t.Fatal(err)
	}
	msgs := req["messages"].([]any)
	// user, assistant, tool = 3
	if len(msgs) != 3 {
		t.Fatalf("messages len = %d, want 3", len(msgs))
	}
	toolMsg := msgs[2].(map[string]any)
	if toolMsg["role"] != "tool" {
		t.Errorf("tool message role = %v", toolMsg["role"])
	}
	if toolMsg["tool_call_id"] != "call_1" {
		t.Errorf("tool_call_id = %v", toolMsg["tool_call_id"])
	}
	if toolMsg["content"] != "72F and sunny" {
		t.Errorf("content = %v", toolMsg["content"])
	}
}

func TestOpenAIProvider_APIKey(t *testing.T) {
	var authHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authHeader = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		resp := chatCompletionResponse{
			Choices: []chatChoice{{
				Message:      chatMessage{Role: "assistant", Content: strPtr("ok")},
				FinishReason: "stop",
			}},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(srv.Close)

	provider := NewOpenAIProvider(srv.URL, WithAPIKey("sk-test-123"))
	conv := NewConversation("model")
	conv.Messages = []Message{UserMessage("hi")}

	_, err := provider.Send(context.Background(), &conv)
	if err != nil {
		t.Fatal(err)
	}
	if authHeader != "Bearer sk-test-123" {
		t.Errorf("Authorization = %q", authHeader)
	}
}

func TestOpenAIProvider_ErrorClassification(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		body     string
		wantKind ErrorKind
	}{
		{
			name:     "400 bad request",
			status:   400,
			body:     `{"error":{"message":"invalid param","type":"invalid_request_error"}}`,
			wantKind: ErrInvalidRequest,
		},
		{
			name:     "400 context length",
			status:   400,
			body:     `{"error":{"message":"maximum context length exceeded","type":"invalid_request_error"}}`,
			wantKind: ErrContextLength,
		},
		{
			name:     "401 unauthorized",
			status:   401,
			body:     `{"error":{"message":"invalid api key","type":"authentication_error"}}`,
			wantKind: ErrAuthentication,
		},
		{
			name:     "404 not found",
			status:   404,
			body:     `{"error":{"message":"model not found","type":"not_found_error"}}`,
			wantKind: ErrNotFound,
		},
		{
			name:     "429 rate limit",
			status:   429,
			body:     `{"error":{"message":"rate limit reached","type":"rate_limit_error"}}`,
			wantKind: ErrRateLimit,
		},
		{
			name:     "500 server error",
			status:   500,
			body:     `{"error":{"message":"internal error","type":"server_error"}}`,
			wantKind: ErrServer,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.status)
				w.Write([]byte(tt.body))
			}))
			t.Cleanup(srv.Close)

			provider := NewOpenAIProvider(srv.URL)
			conv := NewConversation("model")
			conv.Messages = []Message{UserMessage("hi")}

			_, err := provider.Send(context.Background(), &conv)
			if err == nil {
				t.Fatal("expected error")
			}
			var llmErr *Error
			if !errors.As(err, &llmErr) {
				t.Fatalf("expected *Error, got %T", err)
			}
			if llmErr.Kind != tt.wantKind {
				t.Errorf("Kind = %v, want %v", llmErr.Kind, tt.wantKind)
			}
		})
	}
}

func TestOpenAIProvider_NoChoicesError(t *testing.T) {
	resp := chatCompletionResponse{Choices: []chatChoice{}}
	srv, _ := newTestOpenAIServer(t, 200, resp)

	provider := NewOpenAIProvider(srv.URL)
	conv := NewConversation("model")
	conv.Messages = []Message{UserMessage("hi")}

	_, err := provider.Send(context.Background(), &conv)
	if err == nil {
		t.Fatal("expected error")
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) {
		t.Fatalf("expected *Error, got %T", err)
	}
	if llmErr.Kind != ErrServer {
		t.Errorf("Kind = %v, want ErrServer", llmErr.Kind)
	}
}

func TestOpenAIProvider_FinishReasons(t *testing.T) {
	tests := []struct {
		openai string
		want   FinishReason
	}{
		{"stop", FinishReasonStop},
		{"length", FinishReasonLength},
		{"tool_calls", FinishReasonToolUse},
		{"content_filter", FinishReasonContentFilter},
	}

	for _, tt := range tests {
		t.Run(tt.openai, func(t *testing.T) {
			resp := chatCompletionResponse{
				Choices: []chatChoice{{
					Message:      chatMessage{Role: "assistant", Content: strPtr("ok")},
					FinishReason: tt.openai,
				}},
			}
			srv, _ := newTestOpenAIServer(t, 200, resp)

			provider := NewOpenAIProvider(srv.URL)
			conv := NewConversation("model")
			conv.Messages = []Message{UserMessage("hi")}

			result, err := provider.Send(context.Background(), &conv)
			if err != nil {
				t.Fatal(err)
			}
			if result.FinishReason != tt.want {
				t.Errorf("FinishReason = %q, want %q", result.FinishReason, tt.want)
			}
		})
	}
}
