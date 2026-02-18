# unified-llm

A Go wrapper around the AWS Bedrock Converse API with an ergonomic, serialization-friendly design.

The central type is `Conversation` — a plain, JSON-serializable struct that holds the full conversation state: model, system prompts, messages, tools, config, and cumulative token usage. It is a natural fit for Temporal workflows, databases, or any system that needs to persist or pass conversation state between steps.

## Installation

```
go get github.com/quells-bot/unified-llm
```

## Usage

```go
import (
    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
    "github.com/quells-bot/unified-llm/llm"
)

cfg, _ := config.LoadDefaultConfig(ctx)
client := llm.NewClient(bedrockruntime.NewFromConfig(cfg))

conv := llm.NewConversation(
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    llm.WithSystem("You are a helpful assistant."),
    llm.WithMaxTokens(4096),
)

conv, resp, err := client.Send(ctx, conv, llm.UserMessage("Hello!"))
fmt.Println(resp.Message.Text())
```

`Send` never mutates the input conversation — it returns a new one with the assistant reply appended and usage accumulated.

## Tools

```go
tool := llm.NewTool("get_weather", "Get current weather",
    llm.StringParam("location", "City name"),
    llm.OptionalStringParam("unit", "celsius or fahrenheit"),
)

conv := llm.NewConversation(model, llm.WithTools(tool), llm.WithMaxTokens(4096))
conv, resp, err := client.Send(ctx, conv, llm.UserMessage("What's the weather in Paris?"))

for resp.FinishReason == llm.FinishReasonToolUse {
    var results []llm.Message
    for _, tc := range resp.Message.ToolCalls() {
        args, _ := tool.ParseArgs(tc)
        location, _ := args.String("location")
        results = append(results, tc.Result(`{"temp":"15°C","condition":"cloudy"}`))
        _ = location
    }
    conv, resp, err = client.Send(ctx, conv, results...)
}
```

## Middleware

```go
logger := func(ctx context.Context, conv *llm.Conversation, next llm.SendFunc) (*llm.Response, error) {
    resp, err := next(ctx, conv)
    if err == nil {
        log.Printf("tokens: %+v", resp.Usage)
    }
    return resp, err
}

client := llm.NewClient(bd, llm.WithMiddleware(logger))
```

## Error handling

All errors are `*llm.Error` with a `Kind` field for programmatic handling:

```go
var llmErr *llm.Error
if errors.As(err, &llmErr) {
    switch llmErr.Kind {
    case llm.ErrRateLimit:
        // retry with backoff
    case llm.ErrContextLength:
        // truncate conversation
    }
}
```
