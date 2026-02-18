package llm

import (
	"context"
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// SendFunc is the signature for the core Send call and middleware next functions.
type SendFunc func(ctx context.Context, conv *Conversation) (*Response, error)

// Middleware wraps a Send call.
type Middleware func(ctx context.Context, conv *Conversation, next SendFunc) (*Response, error)

// BedrockConverser abstracts the Bedrock Converse call for testing.
type BedrockConverser interface {
	Converse(ctx context.Context, params *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
}

// Client calls the Bedrock Converse API.
type Client struct {
	bedrock    BedrockConverser
	middleware []Middleware
}

// ClientOption configures a Client.
type ClientOption func(*Client)

// WithMiddleware adds middleware to the client.
func WithMiddleware(m ...Middleware) ClientOption {
	return func(c *Client) {
		c.middleware = append(c.middleware, m...)
	}
}

// NewClient creates a new Client with the given Bedrock converser and options.
func NewClient(bedrock BedrockConverser, opts ...ClientOption) *Client {
	c := &Client{bedrock: bedrock}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Send appends the provided messages to a copy of the conversation,
// calls Bedrock Converse, appends the assistant response, accumulates usage,
// and returns the updated conversation and per-turn response.
func (c *Client) Send(ctx context.Context, conv Conversation, messages ...Message) (Conversation, *Response, error) {
	// Copy messages slice so caller's conversation is not mutated
	conv.Messages = append(append([]Message(nil), conv.Messages...), messages...)

	core := func(ctx context.Context, conv *Conversation) (*Response, error) {
		input := toConverseInput(conv)
		output, err := c.bedrock.Converse(ctx, input)
		if err != nil {
			return nil, classifyBedrockError(err)
		}
		msg, usage, reason, err := fromConverseOutput(output)
		if err != nil {
			return nil, err
		}
		return &Response{
			Message:      *msg,
			FinishReason: reason,
			Usage:        *usage,
		}, nil
	}

	// Wrap with middleware (first registered = outermost)
	fn := core
	for i := len(c.middleware) - 1; i >= 0; i-- {
		mw := c.middleware[i]
		next := fn
		fn = func(ctx context.Context, conv *Conversation) (*Response, error) {
			return mw(ctx, conv, next)
		}
	}

	resp, err := fn(ctx, &conv)
	if err != nil {
		return conv, nil, err
	}

	// Append assistant response and accumulate usage
	conv.Messages = append(conv.Messages, resp.Message)
	conv.Usage = conv.Usage.Add(resp.Usage)

	return conv, resp, nil
}

func classifyBedrockError(err error) error {
	var kind ErrorKind
	msg := err.Error()

	var accessDenied *types.AccessDeniedException
	var validation *types.ValidationException
	var notFound *types.ResourceNotFoundException
	var throttling *types.ThrottlingException
	var timeout *types.ModelTimeoutException
	var internal *types.InternalServerException
	var modelErr *types.ModelErrorException

	switch {
	case errors.As(err, &accessDenied):
		kind = ErrAuthentication
	case errors.As(err, &validation):
		kind = ErrInvalidRequest
	case errors.As(err, &notFound):
		kind = ErrNotFound
	case errors.As(err, &throttling):
		kind = ErrRateLimit
	case errors.As(err, &timeout):
		kind = ErrServer
	case errors.As(err, &internal):
		kind = ErrServer
	case errors.As(err, &modelErr):
		kind = ErrServer
	default:
		lower := strings.ToLower(msg)
		switch {
		case strings.Contains(lower, "context length") || strings.Contains(lower, "too many tokens"):
			kind = ErrContextLength
		case strings.Contains(lower, "content filter") || strings.Contains(lower, "guardrail"):
			kind = ErrContentFilter
		default:
			kind = ErrServer
		}
	}

	return &Error{
		Kind:    kind,
		Message: msg,
		Cause:   err,
	}
}
