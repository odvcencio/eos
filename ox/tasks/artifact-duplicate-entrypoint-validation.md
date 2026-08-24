# Reject duplicate artifact entrypoint names

Repository: `m31labs.dev/eos`

Exact base: `b6f8e1f9c050d08938c830e845b0a6e29177373b`

License evidence: the repository root contains an Apache-2.0 `LICENSE` file.
Its SHA-256 is
`1e9af743932caff91b6a201de5b446b594ca9c9c0bfbb02f7a0537fc19c41e22`.

## Defect

`(*Module).Validate` stores entrypoints in `entryByName` without rejecting a
duplicate name. Later step validation reads that map, so the last duplicate
silently replaces earlier entrypoint metadata. `EncodeJSON`, `DecodeJSON`, and
`EncodeMLL` all rely on `Validate` and therefore accept the ambiguous artifact.

This focused probe fails on the exact base:

```go
func TestProbeModuleValidateRejectsDuplicateEntryPoints(t *testing.T) {
	module := NewModule("duplicate-entry")
	entry := EntryPoint{Name: "embed", Kind: EntryPointPipeline}
	module.EntryPoints = []EntryPoint{entry, entry}

	if err := module.Validate(); err == nil {
		t.Fatal("Validate accepted duplicate entrypoint names")
	}
}
```

Observed result:

```text
--- FAIL: TestProbeModuleValidateRejectsDuplicateEntryPoints
    duplicate_entry_probe_test.go:11: Validate accepted duplicate entrypoint names
```

No unmerged remote branch changes `artifact/eos/module.go` or
`artifact/eos/module_test.go`. There are no open pull requests.

## Required behavior

- Make `Module.Validate` reject a second entrypoint with the same non-empty
  name.
- Return the error before the second declaration replaces the first map value.
- Include the duplicate entrypoint name in the error.
- Preserve validation order and all behavior for unique entrypoint names.
- Add one focused regression in `artifact/eos/module_test.go`.
- Use a minimal module from `NewModule` with two pipeline entrypoints named
  `embed`.
- Require a non-nil error and verify that it identifies the duplicate
  entrypoint name.
- Touch only `artifact/eos/module.go` and `artifact/eos/module_test.go`.
- Do not add validation for params, buffers, kernels, bindings, or step names.
- Do not touch `syntax/cst.go`, `syntax/parser_test.go`, terminal-return logic,
  CUDA code, ABI code, generated files, or documentation.

## Relevant production source

From `artifact/eos/module.go`:

```go
// EntryPoint is a kernel or pipeline entry exposed by the module.
type EntryPoint struct {
	Name    string         `json:"name"`
	Kind    EntryPointKind `json:"kind"`
	Inputs  []ValueBinding `json:"inputs,omitempty"`
	Outputs []ValueBinding `json:"outputs,omitempty"`
}

// Validate checks basic artifact invariants.
func (m *Module) Validate() error {
	if m == nil {
		return fmt.Errorf("nil module")
	}
	if m.Name == "" {
		return fmt.Errorf("module name is required")
	}
	if m.Version == "" {
		return fmt.Errorf("module version is required")
	}
	if m.Version != Version {
		return fmt.Errorf("module version %q is not supported, want %q", m.Version, Version)
	}
	if len(m.Requirements.SupportedBackends) == 0 {
		return fmt.Errorf("at least one supported backend is required")
	}
	seenBackends := map[BackendKind]bool{}
	for _, kind := range m.Requirements.SupportedBackends {
		if kind == "" {
			return fmt.Errorf("supported backend is required")
		}
		if seenBackends[kind] {
			return fmt.Errorf("duplicate supported backend %q", kind)
		}
		seenBackends[kind] = true
	}
	seenCapabilities := map[string]bool{}
	for _, capability := range m.Requirements.Capabilities {
		if capability == "" {
			return fmt.Errorf("capability name is required")
		}
		if seenCapabilities[capability] {
			return fmt.Errorf("duplicate capability %q", capability)
		}
		seenCapabilities[capability] = true
	}
	for _, param := range m.Params {
		if param.Name == "" {
			return fmt.Errorf("param name is required")
		}
		if param.Binding == "" {
			return fmt.Errorf("param %q binding is required", param.Name)
		}
		if err := validateValueType(param.Type); err != nil {
			return fmt.Errorf("param %q: %w", param.Name, err)
		}
	}
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
	// Buffer, kernel, and step validation follows.
}
```

`artifact/eos/module.go` already imports `fmt`.

The test file is package `eos` and already imports `strings` and `testing`.
Existing validation tests use plain `testing` checks, for example:

```go
func TestValidateRejectsUnknownStepEntry(t *testing.T) {
	module := NewModule("bad")
	// Existing setup omitted here.
	if err := module.Validate(); err == nil {
		t.Fatal("expected validate error")
	}
}
```

## Output contract

Return only one applicable unified diff. The first bytes must be `diff --git `.
Do not use Markdown fences or commentary. Use the exact existing filenames.
Do not add a new file. Include complete hunk ranges and enough unchanged
context for `git apply`.
