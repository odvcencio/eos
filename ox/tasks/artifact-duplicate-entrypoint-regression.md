# Complete duplicate entrypoint validation with a real regression

Repository: `m31labs.dev/eos`

Exact base: `b6f8e1f9c050d08938c830e845b0a6e29177373b`

License evidence: the root Apache-2.0 `LICENSE` has SHA-256
`1e9af743932caff91b6a201de5b446b594ca9c9c0bfbb02f7a0537fc19c41e22`.

## Correction reason

The first response found the correct production guard but returned an invalid,
incomplete patch. It added a fabricated placeholder comment near
`EntryPoint`, used synthetic index hashes and overlapping hunk ranges, omitted
the required test, and had no final newline.

Its useful production logic was:

```go
if _, exists := entryByName[entry.Name]; exists {
	return fmt.Errorf("duplicate entrypoint %q", entry.Name)
}
```

Return a complete patch against the exact base. Do not add any placeholder or
unrelated comment.

## Required behavior

- Insert the duplicate check after the empty-name check and before the map
  assignment in `Module.Validate`.
- Preserve the first response's useful production logic and error wording.
- Add `TestValidateRejectsDuplicateEntrypointNames` to
  `artifact/eos/module_test.go`.
- In the test, create a module with `NewModule("duplicate-entry")`.
- Give it two `EntryPointPipeline` declarations named `embed`.
- Require a non-nil error.
- Require the error to contain `duplicate entrypoint "embed"`.
- Touch only `artifact/eos/module.go` and `artifact/eos/module_test.go`.
- Do not change any import. The test file already imports `strings` and
  `testing`.
- Do not add validation for any other declaration type.
- Do not touch syntax, terminal-return, CUDA, ABI, generated, or documentation
  paths.

## Exact production context

From `artifact/eos/module.go`:

```go
	entryByName := map[string]EntryPoint{}
	for _, entry := range m.EntryPoints {
		if entry.Name == "" {
			return fmt.Errorf("entrypoint name is required")
		}
		entryByName[entry.Name] = entry
		for _, input := range entry.Inputs {
			if input.Name == "" {
				return fmt.Errorf("entrypoint %q input name is required", entry.Name)
			}
			if err := validateValueType(input.Type); err != nil {
				return fmt.Errorf("entrypoint %q input %q: %w", entry.Name, input.Name, err)
			}
		}
		for _, output := range entry.Outputs {
			if output.Name == "" {
				return fmt.Errorf("entrypoint %q output name is required", entry.Name)
			}
			if err := validateValueType(output.Type); err != nil {
				return fmt.Errorf("entrypoint %q output %q: %w", entry.Name, output.Name, err)
			}
		}
	}
```

## Exact test context

The start of `artifact/eos/module_test.go` is:

```go
package eos

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)
```

Place the new regression after this existing test:

```go
func TestValidateRejectsUnknownStepEntry(t *testing.T) {
	module := NewModule("bad")
	module.EntryPoints = []EntryPoint{
		{
			Name: "embed",
			Kind: EntryPointPipeline,
			Inputs: []ValueBinding{
				{Name: "tokens", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "i32", Shape: []string{"T"}}}},
			},
			Outputs: []ValueBinding{
				{Name: "embeddings", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"T", "D"}}}},
			},
		},
	}
	module.Buffers = []Buffer{{Name: "x", DType: "f16", Shape: []string{"T", "D"}}}
	module.Steps = []Step{
		{Entry: "missing", Kind: StepGather, Inputs: []string{"tokens"}, Outputs: []string{"x"}},
	}

	if err := module.Validate(); err == nil {
		t.Fatal("expected validate error")
	}
}
```

## Output contract

Return only one applicable unified diff. The first bytes must be `diff --git `.
Do not use Markdown fences or commentary. Use real file context and complete
hunk ranges. End the response with a newline.
