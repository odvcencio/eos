package cuda

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"sync"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

type cachedLoad struct {
	compiled             map[string]backend.CompiledKernel
	native               map[string]backend.NativeKernelProgram
	device               *deviceRuntime
	residentMatMulParams map[string]bool
}

// Backend is the CUDA backend stub.
type Backend struct {
	mu          sync.Mutex
	loadCache   map[string]cachedLoad
	cacheHits   int
	cacheMisses int
}

// New returns a CUDA backend.
func New() *Backend {
	return &Backend{loadCache: map[string]cachedLoad{}}
}

// Kind reports the backend identity.
func (b *Backend) Kind() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

// Capabilities reports the runtime features the CUDA backend supports.
func (b *Backend) Capabilities() []string {
	return []string{
		eosartifact.CapabilityCandidatePack,
		eosartifact.CapabilityKVCache,
		eosartifact.CapabilityMaskedMeanPool,
		eosartifact.CapabilityHostFallback,
		eosartifact.CapabilityDeviceExecution,
		eosartifact.CapabilityImageOps,
		eosartifact.CapabilityTrainingLosses,
		eosartifact.CapabilityTurboQuant,
		eosartifact.CapabilitySparseAttention,
	}
}

// CanLoad reports whether the module allows CUDA execution.
func (b *Backend) CanLoad(mod *eosartifact.Module) bool {
	return mod != nil && mod.SupportsBackend(eosartifact.BackendCUDA)
}

// Load prepares a CUDA executor stub.
func (b *Backend) Load(_ context.Context, mod *eosartifact.Module, weights map[string]backend.WeightBinding) (backend.Executor, error) {
	return b.load(context.Background(), mod, weights, "")
}

func (b *Backend) LoadWithCacheKey(ctx context.Context, mod *eosartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	return b.load(ctx, mod, weights, cacheKey)
}

func (b *Backend) load(_ context.Context, mod *eosartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	if cacheKey != "" {
		if cached, ok := b.cachedLoad(cacheKey); ok {
			return &executor{module: mod, weights: weights, compiled: cached.compiled, native: cached.native, device: cached.device, residentMatMulParams: cloneBoolMap(cached.residentMatMulParams)}, nil
		}
	}
	compiled, err := backend.CompileVariants(mod, eosartifact.BackendCUDA)
	if err != nil {
		return nil, err
	}
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	native := map[string]backend.NativeKernelProgram{}
	for _, kernel := range mod.Kernels {
		prog, err := backend.CompileNativeKernelProgram(eosartifact.BackendCUDA, kernel, compiled[kernel.Name])
		if err != nil {
			if device != nil {
				device.close()
			}
			return nil, err
		}
		if device != nil {
			if err := device.attachDeviceExecution(&prog, kernel); err != nil {
				device.close()
				return nil, err
			}
		}
		native[kernel.Name] = prog
	}
	residentMatMulParams, err := bindResidentMatMulParams(mod, weights, device)
	if err != nil {
		if device != nil {
			device.close()
		}
		return nil, err
	}
	if cacheKey != "" {
		b.storeCachedLoad(cacheKey, cachedLoad{compiled: compiled, native: native, device: device, residentMatMulParams: cloneBoolMap(residentMatMulParams)})
	}
	return &executor{module: mod, weights: weights, compiled: compiled, native: native, device: device, residentMatMulParams: residentMatMulParams}, nil
}

func (b *Backend) cachedLoad(cacheKey string) (cachedLoad, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	cached, ok := b.loadCache[cacheKey]
	if ok {
		b.cacheHits++
	} else {
		b.cacheMisses++
	}
	return cached, ok
}

func (b *Backend) storeCachedLoad(cacheKey string, cached cachedLoad) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.loadCache[cacheKey] = cached
}

func bindResidentMatMulParams(mod *eosartifact.Module, weights map[string]backend.WeightBinding, device *deviceRuntime) (map[string]bool, error) {
	if mod == nil || device == nil {
		return nil, nil
	}
	candidates := matMulParamInputs(mod)
	if len(candidates) == 0 {
		return nil, nil
	}
	params := make(map[string]eosartifact.Param, len(mod.Params))
	for _, param := range mod.Params {
		params[param.Name] = param
	}
	bindings := map[string]int{}
	resident := make(map[string]bool, len(candidates))
	for name := range candidates {
		param, ok := params[name]
		if !ok {
			continue
		}
		weight, ok := weights[name]
		if !ok || !deviceResidentWeight(weight) {
			continue
		}
		value, _, err := backend.PreviewValueWithBindings(param.Type, weight.Data, bindings)
		if err != nil {
			return nil, fmt.Errorf("bind resident matmul param %q: %w", name, err)
		}
		tensor, ok := value.(*backend.Tensor)
		if !ok || tensor == nil || len(tensor.Shape) != 2 || !autoResidentMatMulDType(tensor.DType) {
			continue
		}
		if err := device.bindMatMulRight(name, tensor); err != nil {
			return nil, fmt.Errorf("bind resident matmul param %q: %w", name, err)
		}
		resident[name] = true
	}
	if len(resident) == 0 {
		return nil, nil
	}
	return resident, nil
}

func bindBGEPretrainedBERTResidents(device *deviceRuntime, step eosartifact.Step, inputs []*backend.Tensor, contract bertCUDASelectedContract) (bertCUDAResidentWeightPlan, bertCUDAResidentBindStats, error) {
	start := time.Now()
	stats := bertCUDAResidentBindStats{}
	if device == nil {
		return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder requires device runtime")
	}
	plan, _, err := planBGEPretrainedBERTResidentWeights(step, inputs)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	if device.bertResidentCache == nil {
		device.bertResidentCache = map[string]bertCUDAResidentBindingCache{}
	}
	for _, weight := range plan.Weights {
		tensor := inputs[weight.InputIndex]
		bytes, err := checkedResidentTensorBytes(weight.Name, tensor)
		if err != nil {
			return bertCUDAResidentWeightPlan{}, stats, err
		}
		stats.ResidentWeightBytes += bytes
		fingerprint := contract.WeightFingerprints[weight.InputIndex]
		if fingerprint == "" {
			fingerprint, err = tensorF32ContentFingerprint(tensor)
			if err != nil {
				return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("fingerprint BGE CUDA resident %q: %w", weight.Name, err)
			}
		}
		if cached, ok := device.bertResidentCache[weight.Name]; ok && cached.DType == tensor.DType && cached.Fingerprint == fingerprint && cached.WeightSetGeneration == contract.Provenance.WeightSetGeneration && equalIntSlices(cached.Shape, tensor.Shape) && cachedResidentStillBound(device, weight) {
			stats.CacheHits++
			continue
		}
		stats.CacheMisses++
		stats.UploadedBytes += bytes
		switch weight.Role {
		case bertCUDAWeightDenseMatrix:
			if err := device.bindMatMulRight(weight.Name, tensor); err != nil {
				return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("bind BGE CUDA resident matrix %q: %w", weight.Name, err)
			}
		default:
			if err := device.bindBERTResidentTensor(weight.Name, tensor); err != nil {
				return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("bind BGE CUDA resident tensor %q: %w", weight.Name, err)
			}
		}
		device.bertResidentCache[weight.Name] = bertCUDAResidentBindingCache{
			DType:               tensor.DType,
			Shape:               append([]int(nil), tensor.Shape...),
			Fingerprint:         fingerprint,
			WeightSetGeneration: contract.Provenance.WeightSetGeneration,
			Bytes:               bytes,
		}
	}
	stats.ColdBind = stats.CacheMisses > 0
	stats.BindNanos = time.Since(start).Nanoseconds()
	return plan, stats, nil
}

func validateSelectedBGECUDAContractCached(device *deviceRuntime, mod *eosartifact.Module, step eosartifact.Step, inputs []*backend.Tensor) (bertCUDASelectedContract, error) {
	if device == nil {
		return bertCUDASelectedContract{}, fmt.Errorf("cuda BGE full encoder requires device runtime")
	}
	contract, err := validateSelectedBGECUDAContract(mod, step, inputs)
	if err != nil {
		return bertCUDASelectedContract{}, err
	}
	cacheKey, err := bgeSelectedContractCacheKey(mod, step, inputs, contract.Provenance, contract.WeightFingerprint)
	if err != nil {
		return bertCUDASelectedContract{}, err
	}
	if device.bertSelectedContractCache == nil {
		device.bertSelectedContractCache = map[string]bertCUDASelectedContract{}
	}
	if cached, ok := device.bertSelectedContractCache[cacheKey]; ok {
		if cached.ContractFingerprint != contract.ContractFingerprint || cached.WeightFingerprint != contract.WeightFingerprint || cached.Provenance.WeightSetGeneration != contract.Provenance.WeightSetGeneration {
			return bertCUDASelectedContract{}, fmt.Errorf("selected BGE CUDA contract cache entry mismatch")
		}
		cached.CacheHit = true
		return cached, nil
	}
	contract.CacheKey = cacheKey
	device.bertSelectedContractCache[cacheKey] = contract
	return contract, nil
}

func checkedResidentTensorBytes(name string, tensor *backend.Tensor) (int64, error) {
	if tensor == nil {
		return 0, fmt.Errorf("BGE CUDA resident %q tensor is nil", name)
	}
	if err := checkedShapeProduct("BGE CUDA resident "+name, tensor.Shape, len(tensor.F32)); err != nil {
		return 0, err
	}
	if len(tensor.F32) > int(^uint(0)>>1)/4 {
		return 0, fmt.Errorf("BGE CUDA resident %q byte count overflows int", name)
	}
	return int64(len(tensor.F32) * 4), nil
}

func cachedResidentStillBound(device *deviceRuntime, weight bertCUDAResidentWeight) bool {
	switch weight.Role {
	case bertCUDAWeightDenseMatrix:
		_, ok := device.residentMatrices[weight.Name]
		return ok
	default:
		_, ok := device.bertResidentTensors[weight.Name]
		return ok
	}
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func matMulParamInputs(mod *eosartifact.Module) map[string]bool {
	params := make(map[string]bool, len(mod.Params))
	for _, param := range mod.Params {
		params[param.Name] = true
	}
	out := map[string]bool{}
	for _, step := range mod.Steps {
		if step.Kind != eosartifact.StepMatMul {
			continue
		}
		for _, input := range step.Inputs {
			if params[input] {
				out[input] = true
			}
		}
	}
	return out
}

func deviceResidentWeight(weight backend.WeightBinding) bool {
	return weight.Residency == "" || weight.Residency == "device_resident"
}

func autoResidentMatMulDType(dtype string) bool {
	return dtype == "f32" || dtype == "f16"
}

func cloneBoolMap(in map[string]bool) map[string]bool {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]bool, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

type executor struct {
	module               *eosartifact.Module
	weights              map[string]backend.WeightBinding
	compiled             map[string]backend.CompiledKernel
	native               map[string]backend.NativeKernelProgram
	device               *deviceRuntime
	residentMatMulParams map[string]bool
}

func (e *executor) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (e *executor) Run(ctx context.Context, req backend.Request) (backend.Result, error) {
	var before backend.MatMulAcceleratorStats
	if e.device != nil {
		before = e.device.matMulStatsSnapshot()
	}
	result, err := backend.ExecuteSymbolic(ctx, e.module, e.weights, e.compiled, e.dispatchKernel, e.dispatchStep, eosartifact.BackendCUDA, req)
	if err != nil {
		return backend.Result{}, err
	}
	if e.device != nil {
		stats := e.device.matMulStatsSnapshot()
		if result.Metadata == nil {
			result.Metadata = map[string]string{}
		}
		result.Metadata["cuda_matmul_bound_matrices"] = strconv.FormatInt(stats.BoundMatrices, 10)
		result.Metadata["cuda_matmul_bound_right_runs"] = strconv.FormatInt(stats.BoundRightCalls-before.BoundRightCalls, 10)
		result.Metadata["cuda_matmul_run_uploaded_bytes"] = strconv.FormatInt(stats.RunUploadedBytes-before.RunUploadedBytes, 10)
	}
	return result, nil
}

func (e *executor) dispatchKernel(_ context.Context, kernel eosartifact.Kernel, inputs []*backend.Tensor) (backend.KernelDispatchResult, error) {
	prog, ok := e.native[kernel.Name]
	if !ok {
		return backend.KernelDispatchResult{}, fmt.Errorf("CUDA kernel %q is not compiled", kernel.Name)
	}
	meta := cloneLaunchConfig(prog.LaunchConfig)
	runner := prog.Run
	if shouldFallbackScoreKernel(kernel, inputs) && prog.Fallback != nil {
		runner = prog.Fallback
		meta["device_execution"] = false
		meta["dispatch_mode"] = "host_fallback"
		meta["execution_mode"] = "host_fallback"
		meta["launch_api"] = "host_reference"
		meta["fallback_reason"] = "unsupported_input_shape"
	}
	out, err := runner(inputs)
	if err != nil {
		return backend.KernelDispatchResult{}, err
	}
	return backend.KernelDispatchResult{
		Outputs:      out,
		VariantEntry: prog.Compiled.Entry,
		SourceHash:   prog.Compiled.SourceHash,
		Metadata:     meta,
	}, nil
}

func (e *executor) dispatchStep(_ context.Context, step eosartifact.Step, outputType eosartifact.ValueType, inputs []*backend.Tensor) (backend.StepDispatchResult, bool, error) {
	switch step.Kind {
	case eosartifact.StepBERTEmbedder:
		if !bertCUDA12LayerHiddenGateEnabled() {
			return backend.StepDispatchResult{}, false, nil
		}
		if e.device == nil {
			return backend.StepDispatchResult{}, true, fmt.Errorf("cuda BGE full encoder hidden gate requested but device runtime is unavailable")
		}
		e.device.bgeFullEncoderMu.Lock()
		defer e.device.bgeFullEncoderMu.Unlock()
		contract, err := validateSelectedBGECUDAContractCached(e.device, e.module, step, inputs)
		if err != nil {
			return backend.StepDispatchResult{}, true, err
		}
		_, bindStats, err := bindBGEPretrainedBERTResidents(e.device, step, inputs, contract)
		if err != nil {
			return backend.StepDispatchResult{}, true, err
		}
		result, stats, err := e.device.runBGEFullEncoderHidden(step, outputType, inputs)
		if err != nil {
			return backend.StepDispatchResult{}, true, err
		}
		if result.Metadata == nil {
			result.Metadata = map[string]any{}
		}
		stats.ResidentUploadedBytes = bindStats.UploadedBytes
		stats.ResidentWeightBytesReferenced = bindStats.ResidentWeightBytes
		stats.ResidentCacheHits = bindStats.CacheHits
		stats.ResidentCacheMisses = bindStats.CacheMisses
		stats.ResidentBindNanos = bindStats.BindNanos
		stats.ColdResidentBind = bindStats.ColdBind
		stats.ContractFingerprint = contract.ContractFingerprint
		stats.WeightFingerprint = contract.WeightFingerprint
		stats.UploadedBytes += stats.ResidentUploadedBytes
		result.Metadata["resident_uploaded_bytes"] = stats.ResidentUploadedBytes
		result.Metadata["uploaded_bytes"] = stats.UploadedBytes
		result.Metadata["resident_weight_bytes_referenced"] = stats.ResidentWeightBytesReferenced
		result.Metadata["resident_cache_hits"] = stats.ResidentCacheHits
		result.Metadata["resident_cache_misses"] = stats.ResidentCacheMisses
		result.Metadata["resident_bind_nanos"] = stats.ResidentBindNanos
		result.Metadata["cold_resident_bind"] = stats.ColdResidentBind
		result.Metadata["warm_resident_cache"] = !stats.ColdResidentBind
		result.Metadata["contract_fingerprint_sha256"] = stats.ContractFingerprint
		result.Metadata["weight_fingerprint_sha256"] = stats.WeightFingerprint
		result.Metadata["contract_cache_hit"] = contract.CacheHit
		result.Metadata["package_sha256"] = contract.Provenance.PackageSHA256
		result.Metadata["package_identity_sha256"] = contract.Provenance.PackageIdentity
		result.Metadata["module_sha256"] = contract.Provenance.ModuleSHA256
		result.Metadata["weights_sha256"] = contract.Provenance.WeightsSHA256
		result.Metadata["weight_set_generation"] = contract.Provenance.WeightSetGeneration
		result.Metadata["retrieval_role_schema"] = contract.Provenance.RoleSchema
		result.Metadata["retrieval_query_role"] = contract.Provenance.QueryRole
		result.Metadata["retrieval_document_role"] = contract.Provenance.DocumentRole
		result.Metadata["retrieval_query_prefix"] = contract.Provenance.QueryPrefix
		result.Metadata["retrieval_document_prefix"] = contract.Provenance.DocumentPrefix
		result.Metadata["pooling"] = contract.Provenance.Pooling
		result.Metadata["normalization"] = contract.Provenance.Normalization
		result.Metadata["max_length"] = contract.Provenance.MaxLength
		result.Metadata["native_dim"] = contract.Provenance.NativeDim
		return result, true, nil
	case eosartifact.StepMatMul:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinMatMul(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		if len(step.Inputs) == 2 {
			if e.residentMatMulParams[step.Inputs[1]] {
				result, err := e.device.runMatMulWithBoundRight(inputs[0], step.Inputs[1], outputType, false, false)
				if err != nil {
					return backend.StepDispatchResult{}, false, err
				}
				result, err = backend.ApplyMatMulAttributesToResult(inputs[0], inputs[1], step.Attributes, result)
				if err != nil {
					return backend.StepDispatchResult{}, false, err
				}
				return result, true, nil
			}
			if e.residentMatMulParams[step.Inputs[0]] {
				result, err := e.device.runMatMulWithBoundLeft(step.Inputs[0], inputs[1], outputType, false, false)
				if err != nil {
					return backend.StepDispatchResult{}, false, err
				}
				result, err = backend.ApplyMatMulAttributesToResult(inputs[0], inputs[1], step.Attributes, result)
				if err != nil {
					return backend.StepDispatchResult{}, false, err
				}
				return result, true, nil
			}
		}
		result, err := e.device.runMatMul(inputs, outputType)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		result, err = backend.ApplyMatMulAttributesToResult(inputs[0], inputs[1], step.Attributes, result)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepConv2D:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		cfg, ok := planBuiltinConv2D(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runConv2DStep(inputs, outputType, cfg)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepConv2DTrans:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		cfg, ok := planBuiltinConv2DTranspose(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runConv2DTransposeStep(inputs, outputType, cfg)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepTurboQEncode:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		cfg, ok := planBuiltinTurboQEncode(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runTurboQEncodeStep(inputs, outputType, cfg)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepTurboQDecode:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		cfg, ok := planBuiltinTurboQDecode(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runTurboQDecodeStep(inputs, outputType, cfg)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepSparseAttention:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		cfg, ok := planBuiltinSparseAttention(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runSparseAttentionStep(inputs, outputType, cfg)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepTurboSparseAttention:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		cfg, ok := planBuiltinTurboSparseAttention(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runTurboSparseAttentionStep(inputs, outputType, cfg)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepGDN, eosartifact.StepIGDN:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinGDN(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runGDNStep(inputs, outputType, step.Kind == eosartifact.StepIGDN)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepMSELoss:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinMSELoss(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runMSELossStep(inputs, outputType)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepMSSSIMLoss:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinMSELoss(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runMSSSIMLossStep(inputs, outputType)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepScalarAdd:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		if !supportsBuiltinScalarAdd(inputs) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runScalarAddStep(inputs, outputType)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepRDLoss:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		rateWeight := stepAttrFloat32(step.Attributes, "lambda", 1) * stepAttrFloat32(step.Attributes, "rate_scale", 1)
		if !supportsBuiltinRDLoss(inputs, rateWeight) {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runRDLossStep(inputs, outputType, rateWeight)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	case eosartifact.StepCrossEntropy:
		if e.device == nil {
			return backend.StepDispatchResult{}, false, nil
		}
		plan, ok := planBuiltinCrossEntropy(step, inputs)
		if !ok {
			return backend.StepDispatchResult{}, false, nil
		}
		result, err := e.device.runCrossEntropyStep(inputs, outputType, plan)
		if err != nil {
			return backend.StepDispatchResult{}, false, err
		}
		return result, true, nil
	default:
		return backend.StepDispatchResult{}, false, nil
	}
}

func supportsBuiltinMatMul(inputs []*backend.Tensor) bool {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return false
	}
	lhs := inputs[0]
	rhs := inputs[1]
	switch len(lhs.Shape) {
	case 2:
		return len(rhs.Shape) == 2 && lhs.Shape[1] == rhs.Shape[0]
	case 3:
		switch len(rhs.Shape) {
		case 2:
			return lhs.Shape[2] == rhs.Shape[0]
		case 3:
			return lhs.Shape[0] == rhs.Shape[0] && lhs.Shape[2] == rhs.Shape[1]
		default:
			return false
		}
	default:
		return false
	}
}

func supportsBuiltinGDN(inputs []*backend.Tensor) bool {
	if len(inputs) < 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return false
	}
	input, beta, gamma := inputs[0], inputs[1], inputs[2]
	if len(input.Shape) != 4 || len(input.F32) != input.Elements() {
		return false
	}
	channels := input.Shape[1]
	if len(beta.F32) < channels {
		return false
	}
	if len(gamma.Shape) != 2 || gamma.Shape[0] != channels || gamma.Shape[1] != channels || len(gamma.F32) < channels*channels {
		return false
	}
	return true
}

type cudaConv2DConfig struct {
	batches     int
	inChannels  int
	inHeight    int
	inWidth     int
	outChannels int
	outHeight   int
	outWidth    int
	inPerGroup  int
	outPerGroup int
	kernelH     int
	kernelW     int
	groups      int
	strideH     int
	strideW     int
	padH        int
	padW        int
	dilationH   int
	dilationW   int
	hasBias     bool
}

type cudaConv2DTransposeConfig struct {
	batches     int
	inChannels  int
	inHeight    int
	inWidth     int
	outChannels int
	outHeight   int
	outWidth    int
	outPerGroup int
	inPerGroup  int
	kernelH     int
	kernelW     int
	groups      int
	strideH     int
	strideW     int
	padH        int
	padW        int
	dilationH   int
	dilationW   int
	outPadH     int
	outPadW     int
	hasBias     bool
}

func planBuiltinConv2D(step eosartifact.Step, inputs []*backend.Tensor) (cudaConv2DConfig, bool) {
	if len(inputs) < 2 || len(inputs) > 3 || inputs[0] == nil || inputs[1] == nil {
		return cudaConv2DConfig{}, false
	}
	input, weight := inputs[0], inputs[1]
	if len(input.Shape) != 4 || len(weight.Shape) != 4 || len(input.F32) != input.Elements() || len(weight.F32) != weight.Elements() {
		return cudaConv2DConfig{}, false
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	outC, inPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	groups := stepAttrInt(step.Attributes, "groups", 1)
	if groups <= 0 || inPerGroup*groups != inC || outC%groups != 0 {
		return cudaConv2DConfig{}, false
	}
	strideH := stepAttrInt(step.Attributes, "stride_h", stepAttrInt(step.Attributes, "stride", 1))
	strideW := stepAttrInt(step.Attributes, "stride_w", stepAttrInt(step.Attributes, "stride", 1))
	padH := stepAttrInt(step.Attributes, "pad_h", stepAttrInt(step.Attributes, "padding", 0))
	padW := stepAttrInt(step.Attributes, "pad_w", stepAttrInt(step.Attributes, "padding", 0))
	dilationH := stepAttrInt(step.Attributes, "dilation_h", stepAttrInt(step.Attributes, "dilation", 1))
	dilationW := stepAttrInt(step.Attributes, "dilation_w", stepAttrInt(step.Attributes, "dilation", 1))
	if strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 {
		return cudaConv2DConfig{}, false
	}
	outH := (inH+2*padH-dilationH*(kH-1)-1)/strideH + 1
	outW := (inW+2*padW-dilationW*(kW-1)-1)/strideW + 1
	if outH <= 0 || outW <= 0 {
		return cudaConv2DConfig{}, false
	}
	hasBias := len(inputs) == 3 && inputs[2] != nil
	if hasBias && len(inputs[2].F32) < outC {
		return cudaConv2DConfig{}, false
	}
	return cudaConv2DConfig{
		batches:     n,
		inChannels:  inC,
		inHeight:    inH,
		inWidth:     inW,
		outChannels: outC,
		outHeight:   outH,
		outWidth:    outW,
		inPerGroup:  inPerGroup,
		outPerGroup: outC / groups,
		kernelH:     kH,
		kernelW:     kW,
		groups:      groups,
		strideH:     strideH,
		strideW:     strideW,
		padH:        padH,
		padW:        padW,
		dilationH:   dilationH,
		dilationW:   dilationW,
		hasBias:     hasBias,
	}, true
}

func planBuiltinConv2DTranspose(step eosartifact.Step, inputs []*backend.Tensor) (cudaConv2DTransposeConfig, bool) {
	if len(inputs) < 2 || len(inputs) > 3 || inputs[0] == nil || inputs[1] == nil {
		return cudaConv2DTransposeConfig{}, false
	}
	input, weight := inputs[0], inputs[1]
	if len(input.Shape) != 4 || len(weight.Shape) != 4 || len(input.F32) != input.Elements() || len(weight.F32) != weight.Elements() {
		return cudaConv2DTransposeConfig{}, false
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	weightInC, outPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	if weightInC != inC {
		return cudaConv2DTransposeConfig{}, false
	}
	groups := stepAttrInt(step.Attributes, "groups", 1)
	if groups <= 0 || inC%groups != 0 {
		return cudaConv2DTransposeConfig{}, false
	}
	strideH := stepAttrInt(step.Attributes, "stride_h", stepAttrInt(step.Attributes, "stride", 1))
	strideW := stepAttrInt(step.Attributes, "stride_w", stepAttrInt(step.Attributes, "stride", 1))
	padH := stepAttrInt(step.Attributes, "pad_h", stepAttrInt(step.Attributes, "padding", 0))
	padW := stepAttrInt(step.Attributes, "pad_w", stepAttrInt(step.Attributes, "padding", 0))
	dilationH := stepAttrInt(step.Attributes, "dilation_h", stepAttrInt(step.Attributes, "dilation", 1))
	dilationW := stepAttrInt(step.Attributes, "dilation_w", stepAttrInt(step.Attributes, "dilation", 1))
	outPadH := stepAttrInt(step.Attributes, "output_padding_h", stepAttrInt(step.Attributes, "output_padding", 0))
	outPadW := stepAttrInt(step.Attributes, "output_padding_w", stepAttrInt(step.Attributes, "output_padding", 0))
	if strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 {
		return cudaConv2DTransposeConfig{}, false
	}
	outC := outPerGroup * groups
	outH := (inH-1)*strideH - 2*padH + dilationH*(kH-1) + outPadH + 1
	outW := (inW-1)*strideW - 2*padW + dilationW*(kW-1) + outPadW + 1
	if outH <= 0 || outW <= 0 {
		return cudaConv2DTransposeConfig{}, false
	}
	hasBias := len(inputs) == 3 && inputs[2] != nil
	if hasBias && len(inputs[2].F32) < outC {
		return cudaConv2DTransposeConfig{}, false
	}
	return cudaConv2DTransposeConfig{
		batches:     n,
		inChannels:  inC,
		inHeight:    inH,
		inWidth:     inW,
		outChannels: outC,
		outHeight:   outH,
		outWidth:    outW,
		outPerGroup: outPerGroup,
		inPerGroup:  inC / groups,
		kernelH:     kH,
		kernelW:     kW,
		groups:      groups,
		strideH:     strideH,
		strideW:     strideW,
		padH:        padH,
		padW:        padW,
		dilationH:   dilationH,
		dilationW:   dilationW,
		outPadH:     outPadH,
		outPadW:     outPadW,
		hasBias:     hasBias,
	}, true
}

type cudaTurboQConfig struct {
	bits     int
	seed     int64
	batches  int
	channels int
	height   int
	width    int
}

const (
	cudaSparseAttentionMaxTopK                = 128
	cudaTurboSparseAttentionMaxRouteTopBlocks = 128
)

type cudaSparseAttentionConfig struct {
	rank      int
	kvLayout  int
	batches   int
	queryLen  int
	keyLen    int
	queryDim  int
	valueDim  int
	topK      int
	outputLen int
}

type cudaTurboSparseAttentionConfig struct {
	sparse         cudaSparseAttentionConfig
	key            cudaTurboQConfig
	value          cudaTurboQConfig
	routeBlockSize int
	routeTopBlocks int
}

func planBuiltinTurboQEncode(step eosartifact.Step, inputs []*backend.Tensor) (cudaTurboQConfig, bool) {
	if len(inputs) != 1 || inputs[0] == nil {
		return cudaTurboQConfig{}, false
	}
	input := inputs[0]
	if len(input.Shape) != 4 || len(input.F32) != input.Elements() {
		return cudaTurboQConfig{}, false
	}
	bits := stepAttrInt(step.Attributes, "bits", 4)
	if bits != 2 && bits != 4 && bits != 8 {
		return cudaTurboQConfig{}, false
	}
	if input.Shape[1] < 2 {
		return cudaTurboQConfig{}, false
	}
	return cudaTurboQConfig{
		bits:     bits,
		seed:     stepAttrInt64(step.Attributes, "seed", 0x4d697261),
		batches:  input.Shape[0],
		channels: input.Shape[1],
		height:   input.Shape[2],
		width:    input.Shape[3],
	}, true
}

func planBuiltinTurboQDecode(step eosartifact.Step, inputs []*backend.Tensor) (cudaTurboQConfig, bool) {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return cudaTurboQConfig{}, false
	}
	coords, norms := inputs[0], inputs[1]
	if len(coords.Shape) != 4 || len(norms.Shape) != 3 || len(coords.F32) != coords.Elements() || len(norms.F32) != norms.Elements() {
		return cudaTurboQConfig{}, false
	}
	if norms.Shape[0] != coords.Shape[0] || norms.Shape[1] != coords.Shape[2] || norms.Shape[2] != coords.Shape[3] {
		return cudaTurboQConfig{}, false
	}
	bits := stepAttrInt(step.Attributes, "bits", cudaBitsForQTensor(coords))
	if bits != 2 && bits != 4 && bits != 8 {
		return cudaTurboQConfig{}, false
	}
	if coords.Shape[1] < 2 {
		return cudaTurboQConfig{}, false
	}
	return cudaTurboQConfig{
		bits:     bits,
		seed:     stepAttrInt64(step.Attributes, "seed", 0x4d697261),
		batches:  coords.Shape[0],
		channels: coords.Shape[1],
		height:   coords.Shape[2],
		width:    coords.Shape[3],
	}, true
}

func planBuiltinSparseAttention(step eosartifact.Step, inputs []*backend.Tensor) (cudaSparseAttentionConfig, bool) {
	if len(inputs) != 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return cudaSparseAttentionConfig{}, false
	}
	query, key, value := inputs[0], inputs[1], inputs[2]
	for _, tensor := range []*backend.Tensor{query, key, value} {
		if len(tensor.F32) != tensor.Elements() {
			return cudaSparseAttentionConfig{}, false
		}
	}
	if len(query.Shape) == 2 && len(key.Shape) == 2 && len(value.Shape) == 2 {
		qLen, queryDim := query.Shape[0], query.Shape[1]
		keyLen, keyDim := key.Shape[0], key.Shape[1]
		valueLen, valueDim := value.Shape[0], value.Shape[1]
		if qLen < 0 || keyLen <= 0 || queryDim <= 0 || queryDim != keyDim || keyLen != valueLen || valueDim <= 0 {
			return cudaSparseAttentionConfig{}, false
		}
		topK := sparseAttentionTopKForCUDA(step.Attributes, keyLen)
		if topK <= 0 || topK > cudaSparseAttentionMaxTopK {
			return cudaSparseAttentionConfig{}, false
		}
		return cudaSparseAttentionConfig{
			rank:      2,
			kvLayout:  0,
			batches:   1,
			queryLen:  qLen,
			keyLen:    keyLen,
			queryDim:  queryDim,
			valueDim:  valueDim,
			topK:      topK,
			outputLen: qLen * valueDim,
		}, true
	}
	if len(query.Shape) == 3 && len(key.Shape) == 3 && len(value.Shape) == 3 {
		batches, qLen, queryDim := query.Shape[0], query.Shape[1], query.Shape[2]
		keyBatches, keyLen, keyDim := key.Shape[0], key.Shape[1], key.Shape[2]
		valueBatches, valueLen, valueDim := value.Shape[0], value.Shape[1], value.Shape[2]
		if batches <= 0 || batches != keyBatches || batches != valueBatches || qLen < 0 || keyLen <= 0 || queryDim <= 0 || queryDim != keyDim || keyLen != valueLen || valueDim <= 0 {
			return cudaSparseAttentionConfig{}, false
		}
		topK := sparseAttentionTopKForCUDA(step.Attributes, keyLen)
		if topK <= 0 || topK > cudaSparseAttentionMaxTopK {
			return cudaSparseAttentionConfig{}, false
		}
		return cudaSparseAttentionConfig{
			rank:      3,
			kvLayout:  0,
			batches:   batches,
			queryLen:  qLen,
			keyLen:    keyLen,
			queryDim:  queryDim,
			valueDim:  valueDim,
			topK:      topK,
			outputLen: batches * qLen * valueDim,
		}, true
	}
	return cudaSparseAttentionConfig{}, false
}

func planBuiltinTurboSparseAttention(step eosartifact.Step, inputs []*backend.Tensor) (cudaTurboSparseAttentionConfig, bool) {
	if len(inputs) != 5 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil || inputs[3] == nil || inputs[4] == nil {
		return cudaTurboSparseAttentionConfig{}, false
	}
	query, keyCoords, keyNorms, valueCoords, valueNorms := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	for _, tensor := range []*backend.Tensor{query, keyCoords, keyNorms, valueCoords, valueNorms} {
		if len(tensor.F32) != tensor.Elements() {
			return cudaTurboSparseAttentionConfig{}, false
		}
	}
	if len(keyCoords.Shape) != 4 || len(valueCoords.Shape) != 4 || len(keyNorms.Shape) != 3 || len(valueNorms.Shape) != 3 {
		return cudaTurboSparseAttentionConfig{}, false
	}
	if keyCoords.Shape[3] != 1 || valueCoords.Shape[3] != 1 {
		return cudaTurboSparseAttentionConfig{}, false
	}
	if keyNorms.Shape[0] != keyCoords.Shape[0] || keyNorms.Shape[1] != keyCoords.Shape[2] || keyNorms.Shape[2] != keyCoords.Shape[3] {
		return cudaTurboSparseAttentionConfig{}, false
	}
	if valueNorms.Shape[0] != valueCoords.Shape[0] || valueNorms.Shape[1] != valueCoords.Shape[2] || valueNorms.Shape[2] != valueCoords.Shape[3] {
		return cudaTurboSparseAttentionConfig{}, false
	}
	if keyCoords.Shape[0] != valueCoords.Shape[0] || keyCoords.Shape[2] != valueCoords.Shape[2] {
		return cudaTurboSparseAttentionConfig{}, false
	}
	bits := stepAttrInt(step.Attributes, "bits", cudaBitsForQTensor(keyCoords))
	if bits != 2 && bits != 4 && bits != 8 {
		return cudaTurboSparseAttentionConfig{}, false
	}
	if cudaBitsForQTensor(keyCoords) != bits {
		return cudaTurboSparseAttentionConfig{}, false
	}
	if cudaBitsForQTensor(valueCoords) != bits {
		return cudaTurboSparseAttentionConfig{}, false
	}
	batches, queryDim, keyLen, valueDim := keyCoords.Shape[0], keyCoords.Shape[1], keyCoords.Shape[2], valueCoords.Shape[1]
	if batches <= 0 || queryDim <= 0 || keyLen <= 0 || valueDim <= 0 {
		return cudaTurboSparseAttentionConfig{}, false
	}
	var rank, queryLen int
	switch len(query.Shape) {
	case 2:
		rank = 2
		queryLen = query.Shape[0]
		if batches != 1 || query.Shape[1] != queryDim {
			return cudaTurboSparseAttentionConfig{}, false
		}
	case 3:
		rank = 3
		queryLen = query.Shape[1]
		if query.Shape[0] != batches || query.Shape[2] != queryDim {
			return cudaTurboSparseAttentionConfig{}, false
		}
	default:
		return cudaTurboSparseAttentionConfig{}, false
	}
	if queryLen < 0 {
		return cudaTurboSparseAttentionConfig{}, false
	}
	topK := sparseAttentionTopKForCUDA(step.Attributes, keyLen)
	if topK <= 0 || topK > cudaSparseAttentionMaxTopK {
		return cudaTurboSparseAttentionConfig{}, false
	}
	routeBlockSize := stepAttrInt(step.Attributes, "route_block_size", 0)
	routeTopBlocks := stepAttrInt(step.Attributes, "route_top_blocks", 0)
	if (routeBlockSize > 0) != (routeTopBlocks > 0) {
		return cudaTurboSparseAttentionConfig{}, false
	}
	if routeBlockSize > 0 {
		if routeBlockSize > keyLen {
			routeBlockSize = keyLen
		}
		blockCount := (keyLen + routeBlockSize - 1) / routeBlockSize
		if routeTopBlocks > blockCount {
			routeTopBlocks = blockCount
		}
		if routeTopBlocks <= 0 || routeTopBlocks > cudaTurboSparseAttentionMaxRouteTopBlocks {
			return cudaTurboSparseAttentionConfig{}, false
		}
	}
	seed := stepAttrInt64(step.Attributes, "seed", 0x4d697261)
	keyCfg := cudaTurboQConfig{
		bits:     bits,
		seed:     seed,
		batches:  batches,
		channels: queryDim,
		height:   keyLen,
		width:    1,
	}
	valueCfg := cudaTurboQConfig{
		bits:     bits,
		seed:     seed,
		batches:  batches,
		channels: valueDim,
		height:   keyLen,
		width:    1,
	}
	return cudaTurboSparseAttentionConfig{
		sparse: cudaSparseAttentionConfig{
			rank:      rank,
			kvLayout:  1,
			batches:   batches,
			queryLen:  queryLen,
			keyLen:    keyLen,
			queryDim:  queryDim,
			valueDim:  valueDim,
			topK:      topK,
			outputLen: batches * queryLen * valueDim,
		},
		key:            keyCfg,
		value:          valueCfg,
		routeBlockSize: routeBlockSize,
		routeTopBlocks: routeTopBlocks,
	}, true
}

func sparseAttentionTopKForCUDA(attrs map[string]string, keyLen int) int {
	if keyLen <= 0 {
		return 0
	}
	topK := stepAttrInt(attrs, "top_k", 0)
	if topK <= 0 {
		topK = int(math.Ceil(math.Sqrt(float64(keyLen))))
	}
	if topK < 1 {
		topK = 1
	}
	if topK > keyLen {
		topK = keyLen
	}
	return topK
}

func supportsBuiltinMSELoss(inputs []*backend.Tensor) bool {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return false
	}
	lhs, rhs := inputs[0], inputs[1]
	return lhs.EqualShape(rhs) && len(lhs.F32) == len(rhs.F32) && len(lhs.F32) == lhs.Elements()
}

func supportsBuiltinScalarAdd(inputs []*backend.Tensor) bool {
	if len(inputs) == 0 {
		return false
	}
	for _, input := range inputs {
		if input == nil || len(input.F32) != 1 {
			return false
		}
	}
	return true
}

func supportsBuiltinRDLoss(inputs []*backend.Tensor, lambda float32) bool {
	return supportsBuiltinScalarAdd(inputs) && len(inputs) == 2 && !math.IsNaN(float64(lambda)) && !math.IsInf(float64(lambda), 0) && lambda >= 0
}

type cudaCrossEntropyMode int

const (
	cudaCrossEntropyCategorical cudaCrossEntropyMode = iota
	cudaCrossEntropyBitPlane
	cudaCrossEntropyLogNormal
)

type cudaCrossEntropyLayout int

const (
	cudaCrossEntropyLayoutUniform cudaCrossEntropyLayout = iota
	cudaCrossEntropyLayoutGlobal
	cudaCrossEntropyLayoutFlat
	cudaCrossEntropyLayoutNCHW
	cudaCrossEntropyLayoutSigmoidFallback
)

type cudaSigmaMode int

const (
	cudaSigmaRaw cudaSigmaMode = iota
	cudaSigmaSoftplus
	cudaSigmaExp
)

type cudaCrossEntropyPlan struct {
	mode      cudaCrossEntropyMode
	layout    cudaCrossEntropyLayout
	levels    int
	bits      int
	sigmaMode cudaSigmaMode
}

func planBuiltinCrossEntropy(step eosartifact.Step, inputs []*backend.Tensor) (cudaCrossEntropyPlan, bool) {
	if len(inputs) < 1 || inputs[0] == nil {
		return cudaCrossEntropyPlan{}, false
	}
	codes := inputs[0]
	if len(codes.F32) != codes.Elements() {
		return cudaCrossEntropyPlan{}, false
	}
	var logits *backend.Tensor
	if len(inputs) > 1 {
		logits = inputs[1]
	}
	attrs := step.Attributes
	if attrs != nil && attrs["distribution"] == "log_normal" {
		sigmaMode, ok := cudaSigmaModeForAttrs(attrs)
		if !ok || logits == nil || len(codes.Shape) != 3 || len(logits.Shape) != 4 {
			return cudaCrossEntropyPlan{}, false
		}
		if logits.Shape[0] != codes.Shape[0] || logits.Shape[1] < 2 || logits.Shape[2] != codes.Shape[1] || logits.Shape[3] != codes.Shape[2] {
			return cudaCrossEntropyPlan{}, false
		}
		if len(logits.F32) < logits.Elements() {
			return cudaCrossEntropyPlan{}, false
		}
		return cudaCrossEntropyPlan{mode: cudaCrossEntropyLogNormal, layout: cudaCrossEntropyLayoutNCHW, levels: 256, bits: 8, sigmaMode: sigmaMode}, true
	}

	bits := stepAttrInt(attrs, "bits", cudaBitsForQTensor(codes))
	levels := stepAttrInt(attrs, "levels", 0)
	if levels <= 0 {
		if bits > 0 {
			levels = 1 << bits
		} else {
			levels = 256
		}
	}
	if levels <= 0 {
		return cudaCrossEntropyPlan{}, false
	}
	if cudaFactorizationAttr(attrs) == "bit-plane" {
		if bits <= 0 || bits > 16 {
			return cudaCrossEntropyPlan{}, false
		}
		layout, ok := cudaCrossEntropyBitLayout(codes, logits, bits)
		if !ok {
			return cudaCrossEntropyPlan{}, false
		}
		return cudaCrossEntropyPlan{mode: cudaCrossEntropyBitPlane, layout: layout, levels: levels, bits: bits}, true
	}
	layout, ok := cudaCrossEntropyCategoricalLayout(codes, logits, levels, attrs)
	if !ok {
		return cudaCrossEntropyPlan{}, false
	}
	return cudaCrossEntropyPlan{mode: cudaCrossEntropyCategorical, layout: layout, levels: levels, bits: bits}, true
}

func cudaCrossEntropyCategoricalLayout(codes, logits *backend.Tensor, levels int, attrs map[string]string) (cudaCrossEntropyLayout, bool) {
	if logits == nil || len(logits.F32) == 0 {
		return cudaCrossEntropyLayoutUniform, true
	}
	if len(logits.F32) == levels {
		return cudaCrossEntropyLayoutGlobal, true
	}
	if attrs != nil && attrs["logits_layout"] == "nchw_alphabet" && supportsNCHWAlphabet(codes, logits, levels) {
		return cudaCrossEntropyLayoutNCHW, true
	}
	if len(logits.F32) >= codes.Elements()*levels {
		return cudaCrossEntropyLayoutFlat, true
	}
	if len(logits.F32) > 0 {
		return cudaCrossEntropyLayoutSigmoidFallback, true
	}
	return cudaCrossEntropyLayoutUniform, true
}

func cudaCrossEntropyBitLayout(codes, logits *backend.Tensor, bits int) (cudaCrossEntropyLayout, bool) {
	if logits == nil || len(logits.F32) == 0 {
		return cudaCrossEntropyLayoutUniform, true
	}
	if supportsNCHWBitPair(codes, logits, bits) {
		return cudaCrossEntropyLayoutNCHW, true
	}
	if len(logits.F32) == bits*2 {
		return cudaCrossEntropyLayoutGlobal, true
	}
	if len(logits.F32) >= codes.Elements()*bits*2 {
		return cudaCrossEntropyLayoutFlat, true
	}
	if len(logits.F32) > 0 {
		return cudaCrossEntropyLayoutSigmoidFallback, true
	}
	return cudaCrossEntropyLayoutUniform, true
}

func supportsNCHWAlphabet(codes, logits *backend.Tensor, levels int) bool {
	if codes == nil || logits == nil || len(codes.Shape) != 4 || len(logits.Shape) != 4 {
		return false
	}
	return logits.Shape[0] == codes.Shape[0] &&
		logits.Shape[1] >= codes.Shape[1]*levels &&
		logits.Shape[2] == codes.Shape[2] &&
		logits.Shape[3] == codes.Shape[3] &&
		len(logits.F32) >= logits.Elements()
}

func supportsNCHWBitPair(codes, logits *backend.Tensor, bits int) bool {
	if codes == nil || logits == nil || len(codes.Shape) != 4 || len(logits.Shape) != 4 {
		return false
	}
	return logits.Shape[0] == codes.Shape[0] &&
		logits.Shape[1] >= codes.Shape[1]*bits*2 &&
		logits.Shape[2] == codes.Shape[2] &&
		logits.Shape[3] == codes.Shape[3] &&
		len(logits.F32) >= logits.Elements()
}

func cudaSigmaModeForAttrs(attrs map[string]string) (cudaSigmaMode, bool) {
	if attrs == nil {
		return cudaSigmaRaw, true
	}
	switch attrs["sigma_parameter"] {
	case "":
		return cudaSigmaRaw, true
	case "softplus":
		return cudaSigmaSoftplus, true
	case "exp":
		return cudaSigmaExp, true
	default:
		return cudaSigmaRaw, false
	}
}

func cudaFactorizationAttr(attrs map[string]string) string {
	if attrs == nil {
		return "categorical"
	}
	switch attrs["factorization"] {
	case "bit-plane", "bitplane", "bit_plane":
		return "bit-plane"
	default:
		return "categorical"
	}
}

func cudaBitsForQTensor(t *backend.Tensor) int {
	if t == nil {
		return 0
	}
	switch t.DType {
	case "q2":
		return 2
	case "q4":
		return 4
	case "q8", "q_norm":
		return 8
	default:
		return 0
	}
}

func stepAttrInt(attrs map[string]string, key string, fallback int) int {
	if attrs == nil {
		return fallback
	}
	raw := attrs[key]
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func stepAttrFloat32(attrs map[string]string, key string, fallback float32) float32 {
	if attrs == nil {
		return fallback
	}
	raw := attrs[key]
	if raw == "" {
		return fallback
	}
	value, err := strconv.ParseFloat(raw, 32)
	if err != nil {
		return fallback
	}
	return float32(value)
}

func stepAttrInt64(attrs map[string]string, key string, fallback int64) int64 {
	if attrs == nil {
		return fallback
	}
	raw := attrs[key]
	if raw == "" {
		return fallback
	}
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return fallback
	}
	return value
}

func shouldFallbackScoreKernel(kernel eosartifact.Kernel, inputs []*backend.Tensor) bool {
	if len(kernel.Body) == 0 {
		return false
	}
	switch kernel.Body[0].Op {
	case "dot", "cosine", "l2_distance":
		if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
			return false
		}
		query := inputs[0]
		docs := inputs[1]
		return !(len(query.Shape) == 1 && len(docs.Shape) == 2 && query.Shape[0] == docs.Shape[1])
	default:
		return false
	}
}

func cloneLaunchConfig(in map[string]any) map[string]any {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
