package llm

import (
	"context"
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// BedrockConverser abstracts the Bedrock Converse call for testing.
type BedrockConverser interface {
	Converse(ctx context.Context, params *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
}

// BedrockProvider implements Provider using AWS Bedrock Converse.
type BedrockProvider struct {
	client BedrockConverser
}

// NewBedrockProvider creates a Provider backed by AWS Bedrock.
func NewBedrockProvider(client BedrockConverser) *BedrockProvider {
	return &BedrockProvider{client: client}
}

// Send translates the conversation to Bedrock format, calls Converse, and
// translates the response back.
func (p *BedrockProvider) Send(ctx context.Context, conv *Conversation) (*Response, error) {
	input := toConverseInput(conv)
	output, err := p.client.Converse(ctx, input)
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
