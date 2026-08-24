# Task: reject duplicate value-binding names within each artifact signature scope

Return one directly applicable Git unified diff and nothing else. The first
bytes must be `diff --git `, every hunk must have valid line ranges, do not use
Markdown fences, and end the response with a newline. Modify only these
existing files:

- `artifact/eos/module.go`
- `artifact/eos/module_test.go`

## Exact repository checkpoint

- Clean HEAD: `788d01b1b6da8db09b8d965d106fa9a58c7187a9`
- Root license: Apache-2.0; `LICENSE` SHA-256
  `1e9af743932caff91b6a201de5b446b594ca9c9c0bfbb02f7a0537fc19c41e22`
- This HEAD includes duplicate-entrypoint and duplicate top-level declaration
  validation from merged PRs #13 and #14.
- Baseline `go test ./artifact/eos -count=1` passes.
- Baseline `go test ./... -count=1` passes.

## Deterministic defect

`(*Module).Validate` silently accepts duplicate `ValueBinding.Name` values in
all four independently scoped signature lists:

1. one entrypoint's `Inputs`;
2. one entrypoint's `Outputs`;
3. one kernel's `Inputs`;
4. one kernel's `Outputs`.

A table-driven probe constructed otherwise-valid modules for each case. Every
subtest failed because `Validate` returned nil:

```text
--- FAIL: TestProbeValidateRejectsDuplicateValueBindingNames/entrypoint_inputs
    Validate accepted duplicate value binding names
--- FAIL: TestProbeValidateRejectsDuplicateValueBindingNames/entrypoint_outputs
    Validate accepted duplicate value binding names
--- FAIL: TestProbeValidateRejectsDuplicateValueBindingNames/kernel_inputs
    Validate accepted duplicate value binding names
--- FAIL: TestProbeValidateRejectsDuplicateValueBindingNames/kernel_outputs
    Validate accepted duplicate value binding names
```

A control with the binding name `value` in one entrypoint's input and a
different entrypoint's output passes today and must continue to pass. Duplicate
tracking must therefore be local to the individual binding list. Do not impose
module-global binding-name uniqueness and do not combine input and output
names into one namespace.

## Required behavior

- Reject the second occurrence of a name within each of the four lists above,
  before later lookup or serialization can make the declaration ambiguous.
- Preserve the existing empty-name and value-type validation behavior.
- Return contextual errors that identify the owning entrypoint/kernel, whether
  the duplicate is an input or output, and the duplicate name.
- Preserve equal binding names across different entrypoints, different
  kernels, and input-versus-output lists.
- Add one focused table-driven regression covering all four rejected scopes
  and at least one accepted cross-scope control.
- Keep the change local to module validation. Do not alter serialization,
  syntax, terminal-return handling, CUDA/ABI/runtime code, or embedding code.

## Exact production types

```go
type EntryPoint struct {
	Name    string         `json:"name"`
	Kind    EntryPointKind `json:"kind"`
	Inputs  []ValueBinding `json:"inputs,omitempty"`
	Outputs []ValueBinding `json:"outputs,omitempty"`
}

type ValueBinding struct {
	Name string    `json:"name"`
	Type ValueType `json:"type"`
}

type Kernel struct {
	Name     string          `json:"name"`
	Inputs   []ValueBinding  `json:"inputs,omitempty"`
	Outputs  []ValueBinding  `json:"outputs,omitempty"`
	Hints    ScheduleHints   `json:"hints,omitempty"`
	Body     []KernelOp      `json:"body,omitempty"`
	Variants []KernelVariant `json:"variants,omitempty"`
}
```

## Exact affected validator context: `artifact/eos/module.go`

```go
	entryByName := map[string]EntryPoint{}
	for _, entry := range m.EntryPoints {
		if entry.Name == "" {
			return fmt.Errorf("entrypoint name is required")
		}
		if _, exists := entryByName[entry.Name]; exists {
			return fmt.Errorf("duplicate entrypoint %q", entry.Name)
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

```go
	kernelByName := map[string]Kernel{}
	for _, kernel := range m.Kernels {
		if kernel.Name == "" {
			return fmt.Errorf("kernel name is required")
		}
		if _, exists := kernelByName[kernel.Name]; exists {
			return fmt.Errorf("duplicate kernel %q", kernel.Name)
		}
		kernelByName[kernel.Name] = kernel
		for _, input := range kernel.Inputs {
			if input.Name == "" {
				return fmt.Errorf("kernel %q input name is required", kernel.Name)
			}
			if err := validateValueType(input.Type); err != nil {
				return fmt.Errorf("kernel %q input %q: %w", kernel.Name, input.Name, err)
			}
		}
		for _, output := range kernel.Outputs {
			if output.Name == "" {
				return fmt.Errorf("kernel %q output name is required", kernel.Name)
			}
			if err := validateValueType(output.Type); err != nil {
				return fmt.Errorf("kernel %q output %q: %w", kernel.Name, output.Name, err)
			}
		}
		for _, op := range kernel.Body {
			if op.Op == "" {
				return fmt.Errorf("kernel %q has body op with empty op name", kernel.Name)
			}
		}
		if err := validateKernelVariants(m.Requirements.SupportedBackends, kernel); err != nil {
			return err
		}
	}
```

## Existing test conventions: `artifact/eos/module_test.go`

The file is in package `eos`, already imports `strings` and `testing`, and has
table-driven validation coverage. Use a valid tensor binding such as:

```go
valueType := ValueType{
	Kind:   ValueTensor,
	Tensor: &TensorType{DType: "f32"},
}
```

For kernel cases, keep the module otherwise valid by using only WebGPU and a
matching variant, for example:

```go
module.Requirements.SupportedBackends = []BackendKind{BackendWebGPU}
module.Kernels = []Kernel{{
	Name: "binding_kernel",
	Variants: []KernelVariant{{
		Backend: BackendWebGPU,
		Entry:   "binding_kernel",
		Source:  "@compute @workgroup_size(1) fn binding_kernel() {}",
	}},
}}
```

Assert each error contains an exact contextual substring. The accepted control
must call `Validate` and fail the test if any error is returned. Keep all
production logic and tests in the two allowed existing files.
