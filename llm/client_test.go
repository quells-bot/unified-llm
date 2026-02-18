package llm

import (
	"context"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// mockConverser is a test double for BedrockConverser.
type mockConverser struct {
	output *bedrockruntime.ConverseOutput
	err    error
}

func (m *mockConverser) Converse(_ context.Context, _ *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.output, nil
}

func simpleConverseOutput(text string) *bedrockruntime.ConverseOutput {
	return &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role: types.ConversationRoleAssistant,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: text},
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
}

func int32Ptr(v int32) *int32 { return &v }

func TestClientSend_SimpleText(t *testing.T) {
	client := NewClient(&mockConverser{output: simpleConverseOutput("Hello!")})

	conv := NewConversation("us.anthropic.claude-sonnet-4-5-20250929-v1:0",
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
	mock := &mockConverser{output: simpleConverseOutput("reply")}
	client := NewClient(mock)

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

func TestClientSend_BedrockError(t *testing.T) {
	mock := &mockConverser{err: &types.ThrottlingException{Message: strPtr("slow down")}}
	client := NewClient(mock)

	conv := NewConversation("model")
	_, _, err := client.Send(context.Background(), conv, UserMessage("hi"))
	if err == nil {
		t.Fatal("expected error")
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) {
		t.Fatalf("expected *Error, got %T", err)
	}
	if llmErr.Kind != ErrRateLimit {
		t.Errorf("Kind = %v, want ErrRateLimit", llmErr.Kind)
	}
}

func TestClientSend_ImmutableConversation(t *testing.T) {
	client := NewClient(&mockConverser{output: simpleConverseOutput("reply")})

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

	client := NewClient(
		&mockConverser{output: simpleConverseOutput("ok")},
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
