# Tool Definition Builder

## Problem

Defining tools requires hand-crafting JSON Schema as `json.RawMessage`:

```go
Tools: []ToolDefinition{{
    Name:        "get_weather",
    Description: "Get the current weather",
    Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
}}
```

This is verbose, error-prone, and hard to read.

## Design

Add a `NewTool` constructor and `Param` helpers that build `ToolDefinition` values with the JSON Schema serialized automatically.

### Usage

```go
NewTool("get_weather", "Get the current weather",
    StringParam("location", "The city to check"),
    OptionalNumberParam("days", "Forecast days"),
)
```

### Types

```go
// Param describes a single tool input parameter.
type Param struct {
    Name        string
    Type        string // "string", "number", "integer", "boolean"
    Description string
    Required    bool
}
```

### Constructor

```go
func NewTool(name, description string, params ...Param) ToolDefinition
```

Builds `{"type":"object","properties":{...},"required":[...]}` from params and serializes it into `ToolDefinition.Parameters`. No error return -- all inputs are typed, nothing can fail at runtime (json.Marshal on map[string]any of strings is infallible).

### Param helpers

8 functions, one per type x required combination:

```go
func StringParam(name string, desc ...string) Param        // required
func OptionalStringParam(name string, desc ...string) Param
func NumberParam(name string, desc ...string) Param         // required
func OptionalNumberParam(name string, desc ...string) Param
func IntegerParam(name string, desc ...string) Param        // required
func OptionalIntegerParam(name string, desc ...string) Param
func BoolParam(name string, desc ...string) Param           // required
func OptionalBoolParam(name string, desc ...string) Param
```

Description is optional via variadic -- first element used if provided.

### JSON Schema output

All keys are lowercase to match the tool use API:

```json
{
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city to check"
        },
        "days": {
            "type": "number",
            "description": "Forecast days"
        }
    },
    "required": ["location"]
}
```

When a param has no description, the `"description"` key is omitted from that property.

### Location

All new code in `types.go`, next to the existing `ToolDefinition` type and message constructors.

### Tests

Unit tests in `types_test.go`:

- `NewTool` with mixed required/optional params produces correct JSON Schema
- `NewTool` with no params produces `{"type":"object","properties":{},"required":[]}`
- `NewTool` with description-less params omits `description` from schema
- Each param helper returns correct type/required/description values

Update existing adapter tests to use `NewTool` for readability.
