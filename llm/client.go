package llm

import "context"

// Provider translates a Conversation into a provider-specific API call and
// returns the result. Each implementation owns the full pipeline: type
// translation, API call, response translation, and error classification.
type Provider interface {
	Send(ctx context.Context, conv *Conversation) (*Response, error)
}

// SendFunc is the signature for the core Send call and middleware next functions.
type SendFunc func(ctx context.Context, conv *Conversation) (*Response, error)

// Middleware wraps a Send call.
type Middleware func(ctx context.Context, conv *Conversation, next SendFunc) (*Response, error)

// Client calls an LLM provider via the Provider interface.
type Client struct {
	provider   Provider
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

// NewClient creates a new Client backed by AWS Bedrock.
// This is a convenience wrapper for backward compatibility; new code may
// prefer NewClientWithProvider for other backends.
func NewClient(bedrock BedrockConverser, opts ...ClientOption) *Client {
	return NewClientWithProvider(NewBedrockProvider(bedrock), opts...)
}

// NewClientWithProvider creates a new Client with the given Provider.
func NewClientWithProvider(provider Provider, opts ...ClientOption) *Client {
	c := &Client{provider: provider}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Send appends the provided messages to a copy of the conversation,
// calls the provider, appends the assistant response, accumulates usage,
// and returns the updated conversation and per-turn response.
func (c *Client) Send(ctx context.Context, conv Conversation, messages ...Message) (Conversation, *Response, error) {
	// Copy messages slice so caller's conversation is not mutated
	conv.Messages = append(append([]Message(nil), conv.Messages...), messages...)

	core := func(ctx context.Context, conv *Conversation) (*Response, error) {
		return c.provider.Send(ctx, conv)
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
