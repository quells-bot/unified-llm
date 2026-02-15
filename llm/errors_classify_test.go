package llm

import (
	"errors"
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func TestClassifyBedrockError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantKind ErrorKind
	}{
		{
			name:     "AccessDeniedException",
			err:      &types.AccessDeniedException{Message: strPtr("access denied")},
			wantKind: ErrAuthentication,
		},
		{
			name:     "ValidationException",
			err:      &types.ValidationException{Message: strPtr("invalid input")},
			wantKind: ErrInvalidRequest,
		},
		{
			name:     "ResourceNotFoundException",
			err:      &types.ResourceNotFoundException{Message: strPtr("model not found")},
			wantKind: ErrNotFound,
		},
		{
			name:     "ThrottlingException",
			err:      &types.ThrottlingException{Message: strPtr("rate limited")},
			wantKind: ErrRateLimit,
		},
		{
			name:     "ModelTimeoutException",
			err:      &types.ModelTimeoutException{Message: strPtr("timeout")},
			wantKind: ErrServer,
		},
		{
			name:     "InternalServerException",
			err:      &types.InternalServerException{Message: strPtr("internal error")},
			wantKind: ErrServer,
		},
		{
			name:     "ModelErrorException",
			err:      &types.ModelErrorException{Message: strPtr("model error")},
			wantKind: ErrServer,
		},
		{
			name:     "context length error via message",
			err:      fmt.Errorf("context length exceeded: too many tokens"),
			wantKind: ErrContextLength,
		},
		{
			name:     "content filter error via message",
			err:      fmt.Errorf("blocked by content filter"),
			wantKind: ErrContentFilter,
		},
		{
			name:     "guardrail error via message",
			err:      fmt.Errorf("blocked by guardrail policy"),
			wantKind: ErrContentFilter,
		},
		{
			name:     "unknown error",
			err:      fmt.Errorf("something unexpected"),
			wantKind: ErrServer,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := classifyBedrockError("anthropic", tt.err)
			var llmErr *Error
			if !errors.As(result, &llmErr) {
				t.Fatalf("expected *Error, got %T", result)
			}
			if llmErr.Kind != tt.wantKind {
				t.Errorf("Kind = %v, want %v", llmErr.Kind, tt.wantKind)
			}
			if llmErr.Cause != tt.err {
				t.Error("Cause should be the original error")
			}
			if llmErr.Provider != "anthropic" {
				t.Errorf("Provider = %q", llmErr.Provider)
			}
		})
	}
}

func strPtr(s string) *string { return &s }
