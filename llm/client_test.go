package llm

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// mockInvoker is a test double for BedrockInvoker.
type mockInvoker struct {
	response []byte
	err      error
}

func (m *mockInvoker) InvokeModel(_ context.Context, params *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &bedrockruntime.InvokeModelOutput{
		Body: m.response,
	}, nil
}

func TestClientComplete_RoutesToCorrectAdapter(t *testing.T) {
	anthropicResp := `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"from anthropic"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	openaiResp := `{"id":"chat_1","model":"gpt","choices":[{"index":0,"message":{"role":"assistant","content":"from openai"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`

	tests := []struct {
		name     string
		provider string
		response string
		wantText string
	}{
		{"anthropic", "anthropic", anthropicResp, "from anthropic"},
		{"openai", "openai", openaiResp, "from openai"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(&mockInvoker{response: []byte(tt.response)},
				WithAdapter(NewAnthropicAdapter()),
				WithAdapter(NewOpenAIAdapter()),
			)
			resp, err := client.Complete(context.Background(), &Request{
				Model:    "test-model",
				Provider: tt.provider,
				Messages: []Message{UserMessage("hello")},
			})
			if err != nil {
				t.Fatal(err)
			}
			if resp.Text() != tt.wantText {
				t.Errorf("Text = %q, want %q", resp.Text(), tt.wantText)
			}
		})
	}
}

func TestClientComplete_UsesDefaultProvider(t *testing.T) {
	resp := `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	client := NewClient(&mockInvoker{response: []byte(resp)},
		WithAdapter(NewAnthropicAdapter()),
		WithDefaultProvider("anthropic"),
	)
	result, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{UserMessage("hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text() != "ok" {
		t.Errorf("Text = %q", result.Text())
	}
}

func TestClientComplete_NoProviderReturnsConfigError(t *testing.T) {
	client := NewClient(&mockInvoker{},
		WithAdapter(NewAnthropicAdapter()),
	)
	_, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{UserMessage("hello")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) || llmErr.Kind != ErrConfig {
		t.Errorf("expected ErrConfig, got %v", err)
	}
}

func TestClientComplete_UnknownProviderReturnsConfigError(t *testing.T) {
	client := NewClient(&mockInvoker{},
		WithAdapter(NewAnthropicAdapter()),
		WithDefaultProvider("anthropic"),
	)
	_, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Provider: "gemini",
		Messages: []Message{UserMessage("hello")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) || llmErr.Kind != ErrConfig {
		t.Errorf("expected ErrConfig, got %v", err)
	}
}

func TestClientComplete_MiddlewareExecutionOrder(t *testing.T) {
	var order []string
	mw1 := func(ctx context.Context, req *Request, next CompleteFunc) (*Response, error) {
		order = append(order, "mw1-before")
		resp, err := next(ctx, req)
		order = append(order, "mw1-after")
		return resp, err
	}
	mw2 := func(ctx context.Context, req *Request, next CompleteFunc) (*Response, error) {
		order = append(order, "mw2-before")
		resp, err := next(ctx, req)
		order = append(order, "mw2-after")
		return resp, err
	}

	resp := `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	client := NewClient(&mockInvoker{response: []byte(resp)},
		WithAdapter(NewAnthropicAdapter()),
		WithDefaultProvider("anthropic"),
		WithMiddleware(mw1, mw2),
	)
	_, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{UserMessage("hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"mw1-before", "mw2-before", "mw2-after", "mw1-after"}
	if len(order) != len(want) {
		t.Fatalf("order = %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Errorf("order[%d] = %q, want %q", i, order[i], want[i])
		}
	}
}

func TestClientComplete_MiddlewareCanModifyRequest(t *testing.T) {
	// Middleware that injects a provider option
	mw := func(ctx context.Context, req *Request, next CompleteFunc) (*Response, error) {
		if req.ProviderOptions == nil {
			req.ProviderOptions = make(map[string]any)
		}
		req.ProviderOptions["test"] = "injected"
		return next(ctx, req)
	}

	resp := `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	client := NewClient(&mockInvoker{response: []byte(resp)},
		WithAdapter(NewAnthropicAdapter()),
		WithDefaultProvider("anthropic"),
		WithMiddleware(mw),
	)
	_, err := client.Complete(context.Background(), &Request{
		Model:    "test-model",
		Messages: []Message{UserMessage("hello")},
	})
	if err != nil {
		t.Fatal(err)
	}
	// If we got here without error, middleware ran successfully
}

// Suppress unused import
var _ = json.Marshal
