package llm

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Adapter translates between unified types and a provider's native format.
type Adapter interface {
	// Provider returns the provider name (e.g., "anthropic", "openai").
	Provider() string

	// BuildInvokeInput translates a unified Request into Bedrock InvokeModel parameters.
	BuildInvokeInput(req *Request) (*InvokeInput, error)

	// ParseResponse translates a raw Bedrock response into a unified Response.
	ParseResponse(body []byte, req *Request) (*Response, error)
}

// InvokeInput carries the parameters for a Bedrock InvokeModel call.
type InvokeInput struct {
	ModelID     string // Bedrock model ID
	Body        []byte // serialized JSON in the provider's native format
	ContentType string // e.g., "application/json"
	Accept      string // e.g., "application/json"
}

// BedrockInvoker abstracts the Bedrock InvokeModel call for testing.
type BedrockInvoker interface {
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}
