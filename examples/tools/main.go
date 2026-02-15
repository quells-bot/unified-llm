package main

import (
	"context"
	"log"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/quells-bot/unified-llm/llm"
)

const (
	gptOss20B  = "openai.gpt-oss-20b-1:0"
	gptOss120B = "openai.gpt-oss-120b-1:0"
	haiku      = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
	sonnet     = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

func main() {
	ctx := context.Background()
	conf, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Printf("failed to load aws config: %v", err)
		return
	}

	bd := bedrockruntime.NewFromConfig(conf)

	systemPrompt := "You are a service representative engaged in a polite conversation with a customer. " +
		"Anything you say to the user will be written to a simple chat interface, so respond with plain text. " +
		"Do not write any Markdown, code, or ASCII art. " +
		"Be as concise and straightforward as possible."

	tools := []llm.ToolDefinition{
		llm.NewTool(
			"get_user",
			"Get details of the user you are conversing with",
		),
		llm.NewTool(
			"list_user_orders",
			"Get summary of a user's orders",
			llm.IntegerParam("user_id"),
		),
		llm.NewTool(
			"get_user_order",
			"Get details of a user's order",
			llm.IntegerParam("user_id"),
			llm.IntegerParam("order_id"),
		),
	}
	toolsByName := make(map[string]llm.ToolDefinition, len(tools))
	for _, t := range tools {
		toolsByName[t.Name] = t
	}

	client := llm.NewClient(
		bd,
		llm.WithAdapter(llm.NewOpenAIAdapter()),
		llm.WithAdapter(llm.NewAnthropicAdapter()),
	)
	var resp *llm.Response
	messages := []llm.Message{
		llm.SystemMessage(systemPrompt),
		llm.UserMessage("How much do I usually spend on orders?"),
	}
	for {
		resp, err = client.Complete(ctx, &llm.Request{
			Model:    haiku,
			Provider: "anthropic",
			Messages: messages,
			Tools:    tools,
		})
		if err != nil {
			log.Printf("failed to get completion: %v", err)
			return
		}
		messages = append(messages, resp.Message)

		log.Printf("< %s", resp.Message.Text())
		log.Printf("u %+v", resp.Usage)

		switch resp.FinishReason.Reason {
		case llm.FinishReasonToolCalls:
			for _, tc := range resp.ToolCalls() {
				tool, ok := toolsByName[tc.Name]
				if !ok {
					messages = append(messages, tc.ErrorResult(`{"error":"unknown tool"}`))
					continue
				}
				args, parseErr := tool.ParseArgs(tc)
				if parseErr != nil {
					messages = append(messages, tc.ErrorResult(`{"error":"`+parseErr.Error()+`"}`))
					continue
				}
				log.Printf("t %s %+v", tc.Name, args)

				switch tc.Name {
				case "get_user":
					messages = append(messages, tc.Result(`{"user_id":123}`))

				case "list_user_orders":
					userID, _ := args.Int("user_id")
					if userID != 123 {
						messages = append(messages, tc.ErrorResult(`{"error":"unknown user_id"}`))
						continue
					}
					messages = append(messages, tc.Result(`{"orders":[{"id":1000},{"id":1001},{"id":1002}]}`))

				case "get_user_order":
					userID, _ := args.Int("user_id")
					if userID != 123 {
						messages = append(messages, tc.ErrorResult(`{"error":"unknown user_id"}`))
						continue
					}
					orderID, _ := args.Int("order_id")
					switch orderID {
					case 1000:
						messages = append(messages, tc.Result(`{"amount":12.34}`))
					case 1001:
						messages = append(messages, tc.Result(`{"amount":23.45}`))
					case 1002:
						messages = append(messages, tc.Result(`{"amount":34.56}`))
					default:
						messages = append(messages, tc.ErrorResult(`{"error":"unknown order_id"}`))
					}
				}
			}
		default:
			return
		}
	}
}
