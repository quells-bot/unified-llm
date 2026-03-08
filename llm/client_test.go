package llm

import (
	"context"
	"testing"
)

// mockProvider is a test double for Provider.
type mockProvider struct {
	resp *Response
	err  error
}

func (m *mockProvider) Send(_ context.Context, _ *Conversation) (*Response, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.resp, nil
}

func simpleResponse(text string) *Response {
	return &Response{
		Message: Message{
			Role:    RoleAssistant,
			Content: []ContentPart{{Kind: ContentText, Text: text}},
		},
		FinishReason: FinishReasonStop,
		Usage:        Usage{InputTokens: 10, OutputTokens: 5},
	}
}

func TestClientSend_SimpleText(t *testing.T) {
	client := NewClientWithProvider(&mockProvider{resp: simpleResponse("Hello!")})

	conv := NewConversation("test-model",
		WithSystem("Be helpful."),
		WithMaxTokens(1024),
	)

	conv, resp, err := client.Send(context.Background(), conv, UserMessage("hi"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Text() != "Hello!" {
		t.Errorf("Text = %q", resp.Message.Text())
	}
	if resp.FinishReason != FinishReasonStop {
		t.Errorf("FinishReason = %q", resp.FinishReason)
	}
	if len(conv.Messages) != 2 {
		t.Errorf("Messages len = %d", len(conv.Messages))
	}
	if conv.Messages[0].Text() != "hi" {
		t.Errorf("Messages[0] = %q", conv.Messages[0].Text())
	}
	if conv.Messages[1].Text() != "Hello!" {
		t.Errorf("Messages[1] = %q", conv.Messages[1].Text())
	}
	if conv.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d", conv.Usage.InputTokens)
	}
}

func TestClientSend_AccumulatesUsage(t *testing.T) {
	client := NewClientWithProvider(&mockProvider{resp: simpleResponse("reply")})

	conv := NewConversation("model")
	conv, _, err := client.Send(context.Background(), conv, UserMessage("first"))
	if err != nil {
		t.Fatal(err)
	}
	conv, _, err = client.Send(context.Background(), conv, UserMessage("second"))
	if err != nil {
		t.Fatal(err)
	}

	if conv.Usage.InputTokens != 20 {
		t.Errorf("InputTokens = %d, want 20", conv.Usage.InputTokens)
	}
	if len(conv.Messages) != 4 {
		t.Errorf("Messages len = %d, want 4", len(conv.Messages))
	}
}

func TestClientSend_ProviderError(t *testing.T) {
	client := NewClientWithProvider(&mockProvider{
		err: &Error{Kind: ErrRateLimit, Message: "slow down"},
	})

	conv := NewConversation("model")
	_, _, err := client.Send(context.Background(), conv, UserMessage("hi"))
	if err == nil {
		t.Fatal("expected error")
	}
	llmErr, ok := err.(*Error)
	if !ok {
		t.Fatalf("expected *Error, got %T", err)
	}
	if llmErr.Kind != ErrRateLimit {
		t.Errorf("Kind = %v, want ErrRateLimit", llmErr.Kind)
	}
}

func TestClientSend_ImmutableConversation(t *testing.T) {
	client := NewClientWithProvider(&mockProvider{resp: simpleResponse("reply")})

	original := NewConversation("model")
	updated, _, err := client.Send(context.Background(), original, UserMessage("hi"))
	if err != nil {
		t.Fatal(err)
	}

	if len(original.Messages) != 0 {
		t.Errorf("original Messages len = %d, want 0", len(original.Messages))
	}
	if len(updated.Messages) != 2 {
		t.Errorf("updated Messages len = %d, want 2", len(updated.Messages))
	}
}

func TestClientSend_MiddlewareOrder(t *testing.T) {
	var order []string
	mw1 := func(ctx context.Context, conv *Conversation, next SendFunc) (*Response, error) {
		order = append(order, "mw1-before")
		resp, err := next(ctx, conv)
		order = append(order, "mw1-after")
		return resp, err
	}
	mw2 := func(ctx context.Context, conv *Conversation, next SendFunc) (*Response, error) {
		order = append(order, "mw2-before")
		resp, err := next(ctx, conv)
		order = append(order, "mw2-after")
		return resp, err
	}

	client := NewClientWithProvider(
		&mockProvider{resp: simpleResponse("ok")},
		WithMiddleware(mw1, mw2),
	)
	conv := NewConversation("model")
	_, _, err := client.Send(context.Background(), conv, UserMessage("hi"))
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
