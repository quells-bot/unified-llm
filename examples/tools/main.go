package main

import (
	"context"
	"encoding/json"
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
			Tools: []llm.ToolDefinition{
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
			},
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
				var args map[string]any
				if len(tc.Arguments) > 0 {
					parseErr := json.Unmarshal(tc.Arguments, &args)
					if parseErr != nil {
						errMsg := `{"error":"` + parseErr.Error() + `"}`
						messages = append(messages, llm.ToolResultMessage(tc.ID, errMsg, true))
						continue
					}
				}
				log.Printf("t %s %+v", tc.Name, args)
				switch tc.Name {
				case "get_user":
					messages = append(messages, llm.ToolResultMessage(tc.ID, `{"user_id":123}`, false))
				case "list_user_orders":
					if _userID, ok := args["user_id"]; ok {
						userID := _userID.(float64)
						if userID == 123.0 {
							messages = append(messages, llm.ToolResultMessage(tc.ID, `{"orders":[{"id":1000},{"id":1001},{"id":1002}]}`, false))
						} else {
							messages = append(messages, llm.ToolResultMessage(tc.ID, `{"error":"unknown user_id"}`, true))
						}
					} else {
						messages = append(messages, llm.ToolResultMessage(tc.ID, `{"error":"missing required param user_id"}`, true))
					}
				case "get_user_order":
					if _userID, ok := args["user_id"]; ok {
						userID := _userID.(float64)
						if userID == 123.0 {
							if _orderID, ok := args["order_id"]; ok {
								orderID := _orderID.(float64)
								switch orderID {
								case 1000.:
									messages = append(messages, llm.ToolResultMessage(tc.ID, `{"amount":12.34}`, false))
								case 1001.:
									messages = append(messages, llm.ToolResultMessage(tc.ID, `{"amount":23.45}`, false))
								case 1002.:
									messages = append(messages, llm.ToolResultMessage(tc.ID, `{"amount":34.56}`, false))
								default:
									messages = append(messages, llm.ToolResultMessage(tc.ID, `{"error":"unknown order_id"}`, true))
								}
							} else {
								messages = append(messages, llm.ToolResultMessage(tc.ID, `{"error":"missing required param order_id"}`, true))
							}
						} else {
							messages = append(messages, llm.ToolResultMessage(tc.ID, `{"error":"unknown user_id"}`, true))
						}

					} else {
						messages = append(messages, llm.ToolResultMessage(tc.ID, `{"error":"missing required param user_id"}`, true))
					}
				default:
					messages = append(messages, llm.ToolResultMessage(tc.ID, `{"error":"unknown tool"}`, true))
				}
			}
		default:
			return
		}
	}
}
