package backend

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

// ExecuteSymbolic runs the current Eos runtime path. The selected
// backend resolves compiled variants at load time, and plan steps execute
// through backend-owned dispatch where promoted kernels exist and through the
// host/reference path where they do not yet.
func ExecuteSymbolic(ctx context.Context, mod *eosartifact.Module, weights map[string]WeightBinding, compiled map[string]CompiledKernel, dispatch KernelDispatcher, dispatchStep StepDispatcher, kind eosartifact.BackendKind, req Request) (Result, error) {
	if mod == nil {
		return Result{}, fmt.Errorf("nil module")
	}
	entry, ok := entryPointByName(mod, req.Entry)
	if !ok {
		return Result{}, fmt.Errorf("unknown entrypoint %q", req.Entry)
	}
	if err := validateRequestInputs(entry, req.Inputs); err != nil {
		return Result{}, err
	}

	bindings := map[string]int{}
	env := map[string]Value{}
	paramUses := countEntryParamUses(stepsForEntry(mod, entry.Name), mod.Params)
	unusedEntryParams := countUnusedEntryParams(paramUses)
	eagerMaterialized := 0
	lazyMaterialized := 0
	releasedParams := 0
	for _, param := range mod.Params {
		weight, ok := weights[param.Name]
		if !ok {
			return Result{}, fmt.Errorf("missing weight binding for param %q", param.Name)
		}
		if !shouldEagerMaterializeParam(weight) {
			continue
		}
		data, concreteType, err := PreviewValueWithBindings(param.Type, weight.Data, bindings)
		if err != nil {
			return Result{}, fmt.Errorf("param %q: %w", param.Name, err)
		}
		env[param.Name] = Value{
			Type:     concreteType,
			Data:     data,
			Producer: "param:" + param.Name,
			Metadata: map[string]any{
				"materialized":        "eager",
				"weight_residency":    weight.Residency,
				"weight_access_count": weight.AccessCount,
			},
		}
		eagerMaterialized++
	}
	for _, input := range entry.Inputs {
		data, concreteType, err := MaterializeValueWithBindings(input.Type, req.Inputs[input.Name], bindings)
		if err != nil {
			return Result{}, fmt.Errorf("input %q: %w", input.Name, err)
		}
		env[input.Name] = Value{
			Type:     concreteType,
			Data:     data,
			Producer: "input:" + input.Name,
		}
	}

	steps := stepsForEntry(mod, entry.Name)
	if len(steps) == 0 {
		return Result{}, fmt.Errorf("entrypoint %q has no plan steps", entry.Name)
	}

	trace := make([]TraceStep, 0, len(steps))
	result := Result{
		Outputs: map[string]Value{},
		Metadata: map[string]string{
			"backend":                   string(kind),
			"status":                    "hybrid",
			"kernel_dispatch":           "backend_native",
			"weight_bindings":           fmt.Sprintf("%d", len(weights)),
			"compiled_kernels":          fmt.Sprintf("%d", len(compiled)),
			"entrypoint":                entry.Name,
			"step_count":                fmt.Sprintf("%d", len(steps)),
			"params_total":              fmt.Sprintf("%d", len(mod.Params)),
			"params_eager_materialized": fmt.Sprintf("%d", eagerMaterialized),
		},
		Accounting: ExecutionAccounting{
			Backend:         kind,
			FallbackReasons: map[string]int{},
		},
	}

	for _, step := range steps {
		for _, input := range step.Inputs {
			if _, ok := env[input]; !ok {
				if isParamName(mod.Params, input) {
					value, concreteType, materialized, err := materializeParamValue(input, mod.Params, weights, bindings, env)
					if err != nil {
						return Result{}, fmt.Errorf("param %q: %w", input, err)
					}
					if materialized {
						_ = concreteType
						env[input] = value
						lazyMaterialized++
					}
				}
			}
			if _, ok := env[input]; !ok {
				return Result{}, fmt.Errorf("entrypoint %q step %q missing runtime input %q", entry.Name, step.Name, input)
			}
		}
		values, variantEntry, err := executeStep(ctx, mod, entry, step, env, compiled, dispatch, dispatchStep, bindings, kind)
		if err != nil {
			return Result{}, fmt.Errorf("entrypoint %q step %q: %w", entry.Name, step.Name, err)
		}
		result.Accounting.recordStep(step, values)
		trace = append(trace, TraceStep{
			Entry:   step.Entry,
			Kind:    step.Kind,
			Name:    step.Name,
			Kernel:  step.Kernel,
			Variant: variantEntry,
			Inputs:  cloneStrings(step.Inputs),
			Outputs: cloneStrings(step.Outputs),
		})
		switch step.Kind {
		case eosartifact.StepReturn:
			for _, name := range step.Outputs {
				value, ok := env[name]
				if !ok {
					return Result{}, fmt.Errorf("entrypoint %q return references unknown value %q", entry.Name, name)
				}
				result.Outputs[name] = value
			}
		default:
			for i, name := range step.Outputs {
				if i >= len(values) {
					return Result{}, fmt.Errorf("step produced %d values for %d outputs", len(values), len(step.Outputs))
				}
				env[name] = values[i]
			}
		}
		for _, input := range step.Inputs {
			if !isParamName(mod.Params, input) {
				continue
			}
			if paramUses[input] > 0 {
				paramUses[input]--
			}
			if paramUses[input] == 0 && shouldReleaseMaterializedParam(weights[input]) {
				if _, ok := env[input]; ok {
					delete(env, input)
					releasedParams++
				}
			}
		}
	}
	if len(result.Outputs) == 0 {
		return Result{}, fmt.Errorf("entrypoint %q produced no outputs", entry.Name)
	}
	result.Metadata["params_lazy_materialized"] = fmt.Sprintf("%d", lazyMaterialized)
	result.Metadata["params_released"] = fmt.Sprintf("%d", releasedParams)
	result.Metadata["params_unused_for_entry"] = fmt.Sprintf("%d", unusedEntryParams)
	result.Metadata["param_materialization"] = paramMaterializationMode(eagerMaterialized, lazyMaterialized)
	result.Accounting.writeMetadata(result.Metadata)
	result.Trace = trace
	return result, nil
}

func (a *ExecutionAccounting) recordStep(step eosartifact.Step, values []Value) {
	if a == nil {
		return
	}
	a.TotalSteps++
	if step.Kind == eosartifact.StepLaunchKernel {
		a.KernelSteps++
		a.KernelLaunches++
	}
	meta := firstExecutionMetadata(values)
	mode := executionMetadataString(meta, "execution_mode")
	device := executionMetadataBool(meta, "device_execution") || strings.HasSuffix(mode, "_device")
	fallbackReason := executionMetadataString(meta, "fallback_reason")
	fallback := mode == "host_fallback" || fallbackReason != ""
	if device {
		a.DeviceSteps++
		if step.Kind == eosartifact.StepLaunchKernel {
			a.DeviceKernelLaunches++
		}
	} else {
		a.HostSteps++
		if step.Kind == eosartifact.StepLaunchKernel {
			a.HostKernelLaunches++
		}
	}
	if fallback {
		a.FallbackSteps++
		if fallbackReason == "" {
			fallbackReason = "host_fallback"
		}
		if a.FallbackReasons == nil {
			a.FallbackReasons = map[string]int{}
		}
		a.FallbackReasons[fallbackReason]++
	}
	a.UploadBytes += executionMetadataInt64(meta, "uploaded_bytes", "upload_bytes")
	a.DownloadBytes += executionMetadataInt64(meta, "downloaded_bytes", "download_bytes")
	a.SyncCount += executionMetadataInt64(meta, "sync_count", "syncs", "synchronizations")
	a.GraphCaptures += executionMetadataInt64(meta, "graph_captures", "graph_capture_count")
	a.GraphReplays += executionMetadataInt64(meta, "graph_replays", "graph_replay_count")
	a.ResidencyHits += executionMetadataInt64(meta, "residency_hits", "resident_cache_hits")
	a.ResidencyMisses += executionMetadataInt64(meta, "residency_misses", "resident_cache_misses")
}

func firstExecutionMetadata(values []Value) map[string]any {
	for _, value := range values {
		if len(value.Metadata) != 0 {
			return value.Metadata
		}
	}
	return nil
}

func executionMetadataString(meta map[string]any, key string) string {
	if meta == nil {
		return ""
	}
	value, ok := meta[key]
	if !ok {
		return ""
	}
	if text, ok := value.(string); ok {
		return text
	}
	return fmt.Sprint(value)
}

func executionMetadataBool(meta map[string]any, key string) bool {
	if meta == nil {
		return false
	}
	value, ok := meta[key]
	if !ok {
		return false
	}
	switch typed := value.(type) {
	case bool:
		return typed
	case string:
		parsed, err := strconv.ParseBool(typed)
		return err == nil && parsed
	default:
		return false
	}
}

func executionMetadataInt64(meta map[string]any, keys ...string) int64 {
	if meta == nil {
		return 0
	}
	for _, key := range keys {
		value, ok := meta[key]
		if !ok {
			continue
		}
		switch typed := value.(type) {
		case int:
			return int64(typed)
		case int8:
			return int64(typed)
		case int16:
			return int64(typed)
		case int32:
			return int64(typed)
		case int64:
			return typed
		case uint:
			return int64(typed)
		case uint8:
			return int64(typed)
		case uint16:
			return int64(typed)
		case uint32:
			return int64(typed)
		case uint64:
			if typed <= uint64(^uint64(0)>>1) {
				return int64(typed)
			}
		case float32:
			return int64(typed)
		case float64:
			return int64(typed)
		case string:
			if parsed, err := strconv.ParseInt(typed, 10, 64); err == nil {
				return parsed
			}
		}
	}
	return 0
}

func (a *ExecutionAccounting) writeMetadata(meta map[string]string) {
	if a == nil || meta == nil {
		return
	}
	meta["total_steps"] = strconv.Itoa(a.TotalSteps)
	meta["kernel_steps"] = strconv.Itoa(a.KernelSteps)
	meta["device_step_count"] = strconv.Itoa(a.DeviceSteps)
	meta["host_step_count"] = strconv.Itoa(a.HostSteps)
	meta["fallback_step_count"] = strconv.Itoa(a.FallbackSteps)
	meta["kernel_launch_count"] = strconv.Itoa(a.KernelLaunches)
	meta["device_kernel_launch_count"] = strconv.Itoa(a.DeviceKernelLaunches)
	meta["host_kernel_launch_count"] = strconv.Itoa(a.HostKernelLaunches)
	meta["upload_bytes"] = strconv.FormatInt(a.UploadBytes, 10)
	meta["download_bytes"] = strconv.FormatInt(a.DownloadBytes, 10)
	meta["sync_count"] = strconv.FormatInt(a.SyncCount, 10)
	meta["graph_captures"] = strconv.FormatInt(a.GraphCaptures, 10)
	meta["graph_replays"] = strconv.FormatInt(a.GraphReplays, 10)
	meta["residency_hits"] = strconv.FormatInt(a.ResidencyHits, 10)
	meta["residency_misses"] = strconv.FormatInt(a.ResidencyMisses, 10)
	meta["full_device_execution"] = strconv.FormatBool(a.FullDeviceExecution())
	meta["fallback_reasons"] = formatFallbackReasons(a.FallbackReasons)
}

func formatFallbackReasons(reasons map[string]int) string {
	if len(reasons) == 0 {
		return ""
	}
	keys := make([]string, 0, len(reasons))
	for key := range reasons {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, key+"="+strconv.Itoa(reasons[key]))
	}
	return strings.Join(parts, ",")
}

func shouldEagerMaterializeParam(weight WeightBinding) bool {
	return weight.Residency == "" || weight.Residency == "device_resident"
}

func shouldReleaseMaterializedParam(weight WeightBinding) bool {
	return weight.Residency == "lazy_staged"
}

func materializeParamValue(name string, params []eosartifact.Param, weights map[string]WeightBinding, bindings map[string]int, env map[string]Value) (Value, eosartifact.ValueType, bool, error) {
	param, ok := paramByName(params, name)
	if !ok {
		return Value{}, eosartifact.ValueType{}, false, fmt.Errorf("unknown param %q", name)
	}
	if value, ok := env[name]; ok {
		return value, value.Type, false, nil
	}
	weight, ok := weights[name]
	if !ok {
		return Value{}, eosartifact.ValueType{}, false, fmt.Errorf("missing weight binding")
	}
	data, concreteType, err := PreviewValueWithBindings(param.Type, weight.Data, bindings)
	if err != nil {
		return Value{}, eosartifact.ValueType{}, false, err
	}
	return Value{
		Type:     concreteType,
		Data:     data,
		Producer: "param:" + name,
		Metadata: map[string]any{
			"materialized":        "lazy",
			"weight_residency":    weight.Residency,
			"weight_access_count": weight.AccessCount,
		},
	}, concreteType, true, nil
}

func countEntryParamUses(steps []eosartifact.Step, params []eosartifact.Param) map[string]int {
	counts := map[string]int{}
	if len(steps) == 0 || len(params) == 0 {
		return counts
	}
	paramNames := map[string]bool{}
	for _, param := range params {
		paramNames[param.Name] = true
		counts[param.Name] = 0
	}
	for _, step := range steps {
		for _, input := range step.Inputs {
			if paramNames[input] {
				counts[input]++
			}
		}
	}
	return counts
}

func countUnusedEntryParams(uses map[string]int) int {
	unused := 0
	for _, count := range uses {
		if count == 0 {
			unused++
		}
	}
	return unused
}

func isParamName(params []eosartifact.Param, name string) bool {
	_, ok := paramByName(params, name)
	return ok
}

func paramByName(params []eosartifact.Param, name string) (eosartifact.Param, bool) {
	for _, param := range params {
		if param.Name == name {
			return param, true
		}
	}
	return eosartifact.Param{}, false
}

func paramMaterializationMode(eager, lazy int) string {
	switch {
	case lazy == 0:
		return "eager_all"
	case eager == 0:
		return "lazy_on_demand"
	default:
		return "mixed"
	}
}

func entryPointByName(mod *eosartifact.Module, name string) (eosartifact.EntryPoint, bool) {
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, true
		}
	}
	return eosartifact.EntryPoint{}, false
}

func validateRequestInputs(entry eosartifact.EntryPoint, inputs map[string]any) error {
	for _, input := range entry.Inputs {
		if _, ok := inputs[input.Name]; !ok {
			return fmt.Errorf("entrypoint %q missing input %q", entry.Name, input.Name)
		}
	}
	for name := range inputs {
		if !entryHasInput(entry, name) {
			return fmt.Errorf("entrypoint %q does not declare input %q", entry.Name, name)
		}
	}
	return nil
}

func entryHasInput(entry eosartifact.EntryPoint, name string) bool {
	for _, input := range entry.Inputs {
		if input.Name == name {
			return true
		}
	}
	return false
}

func stepsForEntry(mod *eosartifact.Module, entry string) []eosartifact.Step {
	out := make([]eosartifact.Step, 0, len(mod.Steps))
	for _, step := range mod.Steps {
		if step.Entry == entry {
			out = append(out, step)
		}
	}
	return out
}

func resolveStepOutputType(mod *eosartifact.Module, entry eosartifact.EntryPoint, step eosartifact.Step, outputIndex int, env map[string]Value) eosartifact.ValueType {
	if outputIndex < len(step.Outputs) {
		name := step.Outputs[outputIndex]
		if binding, ok := entryOutputByName(entry, name); ok {
			return binding.Type
		}
		if buf, ok := bufferByName(mod, name); ok {
			return valueTypeForBuffer(buf)
		}
	}
	if step.Kind == eosartifact.StepLaunchKernel {
		if kernel, ok := kernelByName(mod, step.Kernel); ok && outputIndex < len(kernel.Outputs) {
			return kernel.Outputs[outputIndex].Type
		}
	}
	if len(step.Inputs) > 0 {
		if value, ok := env[step.Inputs[0]]; ok {
			return value.Type
		}
	}
	return eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32"}}
}

func entryOutputByName(entry eosartifact.EntryPoint, name string) (eosartifact.ValueBinding, bool) {
	for _, output := range entry.Outputs {
		if output.Name == name {
			return output, true
		}
	}
	return eosartifact.ValueBinding{}, false
}

func bufferByName(mod *eosartifact.Module, name string) (eosartifact.Buffer, bool) {
	for _, buf := range mod.Buffers {
		if buf.Name == name {
			return buf, true
		}
	}
	return eosartifact.Buffer{}, false
}

func kernelVariantForBackend(kernel eosartifact.Kernel, kind eosartifact.BackendKind) (eosartifact.KernelVariant, bool) {
	for _, variant := range kernel.Variants {
		if variant.Backend == kind {
			return variant, true
		}
	}
	return eosartifact.KernelVariant{}, false
}

func valueTypeForBuffer(buf eosartifact.Buffer) eosartifact.ValueType {
	if buf.DType == "kv_cache" {
		return eosartifact.ValueType{Kind: eosartifact.ValueKVCache}
	}
	if buf.DType == "candidate_pack" {
		return eosartifact.ValueType{
			Kind:          eosartifact.ValueCandidatePack,
			CandidatePack: &eosartifact.CandidatePackType{Shape: cloneStrings(buf.Shape)},
		}
	}
	return eosartifact.ValueType{
		Kind:   eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{DType: buf.DType, Shape: cloneStrings(buf.Shape)},
	}
}

func producerName(step eosartifact.Step) string {
	if step.Kernel != "" {
		return "kernel:" + step.Kernel
	}
	if step.Name != "" {
		return string(step.Kind) + ":" + step.Name
	}
	return string(step.Kind)
}

func cloneStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, len(in))
	copy(out, in)
	return out
}
