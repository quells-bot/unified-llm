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

func TestBedrockProvider_Send(t *testing.T) {
	provider := NewBedrockProvider(&mockConverser{output: simpleConverseOutput("Hello!")})

	conv := NewConversation("us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		WithSystem("Be helpful."),
		WithMaxTokens(1024),
	)
	conv.Messages = []Message{UserMessage("hi")}

	resp, err := provider.Send(context.Background(), &conv)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Text() != "Hello!" {
		t.Errorf("Text = %q", resp.Message.Text())
	}
	if resp.FinishReason != FinishReasonStop {
		t.Errorf("FinishReason = %q", resp.FinishReason)
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d", resp.Usage.InputTokens)
	}
}

func TestBedrockProvider_Error(t *testing.T) {
	provider := NewBedrockProvider(&mockConverser{
		err: &types.ThrottlingException{Message: strPtr("slow down")},
	})

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
	if llmErr.Kind != ErrRateLimit {
		t.Errorf("Kind = %v, want ErrRateLimit", llmErr.Kind)
	}
}

// TestBedrockProvider_BackwardCompat ensures NewClient still works with BedrockConverser.
func TestBedrockProvider_BackwardCompat(t *testing.T) {
	client := NewClient(&mockConverser{output: simpleConverseOutput("ok")})

	conv := NewConversation("model")
	conv, resp, err := client.Send(context.Background(), conv, UserMessage("hi"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Text() != "ok" {
		t.Errorf("Text = %q", resp.Message.Text())
	}
	if len(conv.Messages) != 2 {
		t.Errorf("Messages len = %d, want 2", len(conv.Messages))
	}
}
