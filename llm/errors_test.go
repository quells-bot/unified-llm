package llm

import (
	"errors"
	"fmt"
	"testing"
)

func TestErrorImplementsError(t *testing.T) {
	e := &Error{
		Kind:    ErrConfig,
		Message: "no default provider",
	}
	want := "llm [config]: no default provider"
	if got := e.Error(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestErrorUnwrap(t *testing.T) {
	cause := fmt.Errorf("underlying issue")
	e := &Error{
		Kind:    ErrServer,
		Message: "bedrock failed",
		Cause:   cause,
	}
	if !errors.Is(e, cause) {
		t.Error("Unwrap should return the cause")
	}
}

func TestErrorKindString(t *testing.T) {
	tests := []struct {
		kind ErrorKind
		want string
	}{
		{ErrConfig, "config"},
		{ErrAuthentication, "authentication"},
		{ErrNotFound, "not_found"},
		{ErrInvalidRequest, "invalid_request"},
		{ErrRateLimit, "rate_limit"},
		{ErrServer, "server"},
		{ErrContextLength, "context_length"},
		{ErrContentFilter, "content_filter"},
	}
	for _, tt := range tests {
		if got := tt.kind.String(); got != tt.want {
			t.Errorf("ErrorKind(%d).String() = %q, want %q", tt.kind, got, tt.want)
		}
	}
}
