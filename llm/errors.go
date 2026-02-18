package llm

import "fmt"

// ErrorKind classifies LLM errors.
type ErrorKind int

const (
	ErrConfig        ErrorKind = iota // misconfiguration
	ErrAuthentication                 // 401/403
	ErrNotFound                       // 404
	ErrInvalidRequest                 // 400
	ErrRateLimit                      // 429
	ErrServer                         // 500+
	ErrContextLength                  // input too large
	ErrContentFilter                  // blocked by safety guardrails
)

var errorKindNames = [...]string{
	ErrConfig:         "config",
	ErrAuthentication: "authentication",
	ErrNotFound:       "not_found",
	ErrInvalidRequest: "invalid_request",
	ErrRateLimit:      "rate_limit",
	ErrServer:         "server",
	ErrContextLength:  "context_length",
	ErrContentFilter:  "content_filter",
}

func (k ErrorKind) String() string {
	if int(k) < len(errorKindNames) {
		return errorKindNames[k]
	}
	return fmt.Sprintf("unknown(%d)", k)
}

// Error is the library's error type.
type Error struct {
	Kind    ErrorKind
	Message string
	Cause   error // underlying error
}

func (e *Error) Error() string {
	return fmt.Sprintf("llm [%s]: %s", e.Kind, e.Message)
}

func (e *Error) Unwrap() error {
	return e.Cause
}
