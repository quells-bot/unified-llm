package llm

import (
	"context"
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// CompleteFunc is the signature for the core completion call and middleware next functions.
type CompleteFunc func(ctx context.Context, req *Request) (*Response, error)

// Middleware wraps a Complete call.
type Middleware func(ctx context.Context, req *Request, next CompleteFunc) (*Response, error)

// Client routes requests to adapters and calls Bedrock InvokeModel.
type Client struct {
	bedrock         BedrockInvoker
	adapters        map[string]Adapter
	defaultProvider string
	middleware      []Middleware
}

type clientConfig struct {
	adapters        []Adapter
	defaultProvider string
	middleware      []Middleware
}

// ClientOption configures a Client.
type ClientOption func(*clientConfig)

// WithAdapter registers an adapter with the client.
func WithAdapter(a Adapter) ClientOption {
	return func(c *clientConfig) {
		c.adapters = append(c.adapters, a)
	}
}

// WithDefaultProvider sets the default provider for requests that don't specify one.
func WithDefaultProvider(provider string) ClientOption {
	return func(c *clientConfig) {
		c.defaultProvider = provider
	}
}

// WithMiddleware adds middleware to the client.
func WithMiddleware(m ...Middleware) ClientOption {
	return func(c *clientConfig) {
		c.middleware = append(c.middleware, m...)
	}
}

// NewClient creates a new Client with the given Bedrock invoker and options.
func NewClient(bedrock BedrockInvoker, opts ...ClientOption) *Client {
	cfg := &clientConfig{}
	for _, o := range opts {
		o(cfg)
	}

	adapters := make(map[string]Adapter, len(cfg.adapters))
	for _, a := range cfg.adapters {
		adapters[a.Provider()] = a
	}

	return &Client{
		bedrock:         bedrock,
		adapters:        adapters,
		defaultProvider: cfg.defaultProvider,
		middleware:      cfg.middleware,
	}
}

// Complete sends a request to the appropriate provider and returns the response.
func (c *Client) Complete(ctx context.Context, req *Request) (*Response, error) {
	// Resolve provider
	provider := req.Provider
	if provider == "" {
		provider = c.defaultProvider
	}
	if provider == "" {
		return nil, &Error{Kind: ErrConfig, Message: "no provider specified and no default provider set"}
	}

	adapter, ok := c.adapters[provider]
	if !ok {
		return nil, &Error{Kind: ErrConfig, Provider: provider, Message: "no adapter registered for provider"}
	}

	// Build the core function
	core := func(ctx context.Context, req *Request) (*Response, error) {
		input, err := adapter.BuildInvokeInput(req)
		if err != nil {
			return nil, err
		}

		output, err := c.bedrock.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
			ModelId:     &input.ModelID,
			Body:        input.Body,
			ContentType: &input.ContentType,
			Accept:      &input.Accept,
		})
		if err != nil {
			return nil, classifyBedrockError(provider, err)
		}

		return adapter.ParseResponse(output.Body, req)
	}

	// Wrap with middleware (first registered = outermost)
	fn := core
	for i := len(c.middleware) - 1; i >= 0; i-- {
		mw := c.middleware[i]
		next := fn
		fn = func(ctx context.Context, req *Request) (*Response, error) {
			return mw(ctx, req, next)
		}
	}

	return fn(ctx, req)
}

func classifyBedrockError(provider string, err error) error {
	var kind ErrorKind
	msg := err.Error()

	// Check for specific Bedrock exception types
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
		// Check message content for additional classification
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
		Kind:     kind,
		Provider: provider,
		Message:  msg,
		Cause:    err,
	}
}
