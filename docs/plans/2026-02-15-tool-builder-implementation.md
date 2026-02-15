# Tool Definition Builder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `NewTool` constructor and `Param` helpers so tool definitions can be built without hand-crafting JSON Schema.

**Architecture:** A `Param` struct and 8 helper functions feed into a `NewTool` constructor that serializes JSON Schema into `ToolDefinition.Parameters`. All code lives in the existing `types.go` file.

**Tech Stack:** Go standard library (`encoding/json`)

---

### Task 1: Add Param type and helpers with tests

**Files:**
- Modify: `llm/types.go:136-141` (after `ToolDefinition`)
- Modify: `llm/types_test.go` (append new tests)

**Step 1: Write the failing tests**

Append to `llm/types_test.go`:

```go
func TestStringParam(t *testing.T) {
	p := StringParam("location", "The city")
	if p.Name != "location" || p.Type != "string" || p.Description != "The city" || !p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestStringParamNoDescription(t *testing.T) {
	p := StringParam("location")
	if p.Description != "" {
		t.Errorf("expected empty description, got %q", p.Description)
	}
}

func TestOptionalStringParam(t *testing.T) {
	p := OptionalStringParam("name")
	if p.Required {
		t.Error("expected Required=false")
	}
	if p.Type != "string" {
		t.Errorf("expected type string, got %q", p.Type)
	}
}

func TestNumberParam(t *testing.T) {
	p := NumberParam("count")
	if p.Type != "number" || !p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestOptionalNumberParam(t *testing.T) {
	p := OptionalNumberParam("count")
	if p.Type != "number" || p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestIntegerParam(t *testing.T) {
	p := IntegerParam("count")
	if p.Type != "integer" || !p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestOptionalIntegerParam(t *testing.T) {
	p := OptionalIntegerParam("count")
	if p.Type != "integer" || p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestBoolParam(t *testing.T) {
	p := BoolParam("verbose")
	if p.Type != "boolean" || !p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}

func TestOptionalBoolParam(t *testing.T) {
	p := OptionalBoolParam("verbose")
	if p.Type != "boolean" || p.Required {
		t.Errorf("unexpected param: %+v", p)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./llm/ -run 'TestStringParam|TestOptionalStringParam|TestNumberParam|TestOptionalNumberParam|TestIntegerParam|TestOptionalIntegerParam|TestBoolParam|TestOptionalBoolParam' -v`

Expected: compilation error — `StringParam` not defined.

**Step 3: Write minimal implementation**

Add to `llm/types.go` after the `ToolDefinition` struct (after line 141):

```go
// Param describes a single tool input parameter.
type Param struct {
	Name        string
	Type        string // "string", "number", "integer", "boolean"
	Description string
	Required    bool
}

func newParam(name, typ string, required bool, desc []string) Param {
	p := Param{Name: name, Type: typ, Required: required}
	if len(desc) > 0 {
		p.Description = desc[0]
	}
	return p
}

// StringParam creates a required string parameter.
func StringParam(name string, desc ...string) Param { return newParam(name, "string", true, desc) }

// OptionalStringParam creates an optional string parameter.
func OptionalStringParam(name string, desc ...string) Param {
	return newParam(name, "string", false, desc)
}

// NumberParam creates a required number parameter.
func NumberParam(name string, desc ...string) Param { return newParam(name, "number", true, desc) }

// OptionalNumberParam creates an optional number parameter.
func OptionalNumberParam(name string, desc ...string) Param {
	return newParam(name, "number", false, desc)
}

// IntegerParam creates a required integer parameter.
func IntegerParam(name string, desc ...string) Param { return newParam(name, "integer", true, desc) }

// OptionalIntegerParam creates an optional integer parameter.
func OptionalIntegerParam(name string, desc ...string) Param {
	return newParam(name, "integer", false, desc)
}

// BoolParam creates a required boolean parameter.
func BoolParam(name string, desc ...string) Param { return newParam(name, "boolean", true, desc) }

// OptionalBoolParam creates an optional boolean parameter.
func OptionalBoolParam(name string, desc ...string) Param {
	return newParam(name, "boolean", false, desc)
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./llm/ -run 'TestStringParam|TestOptionalStringParam|TestNumberParam|TestOptionalNumberParam|TestIntegerParam|TestOptionalIntegerParam|TestBoolParam|TestOptionalBoolParam' -v`

Expected: all PASS.

**Step 5: Commit**

```bash
git add llm/types.go llm/types_test.go
git commit -m "feat: add Param type and helper functions"
```

---

### Task 2: Add NewTool constructor with tests

**Files:**
- Modify: `llm/types.go` (add `NewTool` function after the Param helpers)
- Modify: `llm/types_test.go` (append new tests)

**Step 1: Write the failing tests**

Append to `llm/types_test.go`:

```go
func TestNewToolMixedParams(t *testing.T) {
	tool := NewTool("get_weather", "Get the current weather",
		StringParam("location", "The city"),
		OptionalNumberParam("days", "Forecast days"),
	)
	if tool.Name != "get_weather" {
		t.Errorf("Name = %q", tool.Name)
	}
	if tool.Description != "Get the current weather" {
		t.Errorf("Description = %q", tool.Description)
	}
	want := `{"type":"object","properties":{"location":{"type":"string","description":"The city"},"days":{"type":"number","description":"Forecast days"}},"required":["location"]}`
	assertJSONEqual(t, tool.Parameters, []byte(want))
}

func TestNewToolNoParams(t *testing.T) {
	tool := NewTool("noop", "Does nothing")
	want := `{"type":"object","properties":{},"required":[]}`
	assertJSONEqual(t, tool.Parameters, []byte(want))
}

func TestNewToolNoDescription(t *testing.T) {
	tool := NewTool("ping", "Ping a host", StringParam("host"))
	want := `{"type":"object","properties":{"host":{"type":"string"}},"required":["host"]}`
	assertJSONEqual(t, tool.Parameters, []byte(want))
}
```

Note: `assertJSONEqual` is defined in `llm/anthropic_test.go` and is available to all tests in the `llm` package.

**Step 2: Run tests to verify they fail**

Run: `go test ./llm/ -run 'TestNewTool' -v`

Expected: compilation error — `NewTool` not defined.

**Step 3: Write minimal implementation**

Add to `llm/types.go` after the Param helpers:

```go
// NewTool creates a ToolDefinition with JSON Schema built from params.
func NewTool(name, description string, params ...Param) ToolDefinition {
	properties := make(map[string]map[string]string, len(params))
	required := make([]string, 0, len(params))
	for _, p := range params {
		prop := map[string]string{"type": p.Type}
		if p.Description != "" {
			prop["description"] = p.Description
		}
		properties[p.Name] = prop
		if p.Required {
			required = append(required, p.Name)
		}
	}
	schema := map[string]any{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}
	raw, _ := json.Marshal(schema)
	return ToolDefinition{
		Name:        name,
		Description: description,
		Parameters:  raw,
	}
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./llm/ -run 'TestNewTool' -v`

Expected: all PASS.

**Step 5: Commit**

```bash
git add llm/types.go llm/types_test.go
git commit -m "feat: add NewTool constructor for ergonomic tool definitions"
```

---

### Task 3: Update adapter tests to use NewTool

**Files:**
- Modify: `llm/anthropic_test.go`
- Modify: `llm/openai_test.go`

**Step 1: Update anthropic_test.go**

Replace all 5 occurrences of hand-crafted `ToolDefinition` with `NewTool`:

Line 73-77 (`TestAnthropicBuildInvokeInput_WithTools`):
```go
		// before
		Tools: []ToolDefinition{{
			Name:        "get_weather",
			Description: "Get the current weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		}},
		// after
		Tools: []ToolDefinition{NewTool("get_weather", "Get the current weather", StringParam("location"))},
```

Line 101-105 (`TestAnthropicBuildInvokeInput_ToolResult`):
```go
		// before
		Tools: []ToolDefinition{{
			Name:        "get_weather",
			Description: "Get the current weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		}},
		// after
		Tools: []ToolDefinition{NewTool("get_weather", "Get the current weather", StringParam("location"))},
```

Line 119-123 (`TestAnthropicBuildInvokeInput_ToolChoiceRequired`):
```go
		// before
		Tools: []ToolDefinition{{
			Name:        "my_tool",
			Description: "A tool",
			Parameters:  json.RawMessage(`{"type":"object","properties":{},"required":[]}`),
		}},
		// after
		Tools: []ToolDefinition{NewTool("my_tool", "A tool")},
```

Line 138-142 (`TestAnthropicBuildInvokeInput_ToolChoiceNamed`):
```go
		// before
		Tools: []ToolDefinition{{
			Name:        "my_tool",
			Description: "A tool",
			Parameters:  json.RawMessage(`{"type":"object","properties":{},"required":[]}`),
		}},
		// after
		Tools: []ToolDefinition{NewTool("my_tool", "A tool")},
```

Remove `"encoding/json"` from the import block if no other references remain. Check: `json.RawMessage` is still used on line 96 for `ToolCallData.Arguments`, so the import stays.

**Step 2: Update openai_test.go**

Replace all 4 occurrences:

Line 45-49 (`TestOpenAIBuildInvokeInput_WithTools`):
```go
		// after
		Tools: []ToolDefinition{NewTool("get_weather", "Get weather", StringParam("location"))},
```

Line 73-77 (`TestOpenAIBuildInvokeInput_ToolResult`):
```go
		// after
		Tools: []ToolDefinition{NewTool("get_weather", "Get weather", StringParam("location"))},
```

Line 91-94 (`TestOpenAIBuildInvokeInput_ToolChoiceRequired`):
```go
		// after
		Tools: []ToolDefinition{NewTool("my_tool", "A tool")},
```

Line 110-113 (`TestOpenAIBuildInvokeInput_ToolChoiceNamed`):
```go
		// after
		Tools: []ToolDefinition{NewTool("my_tool", "A tool")},
```

**Step 3: Run all tests to verify no regressions**

Run: `go test ./llm/ -v`

Expected: all tests PASS. The golden files don't change — `NewTool` produces the same JSON Schema that was previously hand-written.

**Step 4: Commit**

```bash
git add llm/anthropic_test.go llm/openai_test.go
git commit -m "refactor: use NewTool in adapter tests"
```
