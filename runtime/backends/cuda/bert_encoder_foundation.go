package cuda

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"math"
	"os"
	"sort"
	"strconv"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const (
	bgeSmallVocabSize        = 30522
	bgeSmallHiddenSize       = 384
	bgeSmallLayers           = 12
	bgeSmallHeads            = 12
	bgeSmallIntermediateSize = 1536
	bgeSmallMaxPositions     = 512
	bgeSmallTypeVocabSize    = 2
	bgeSmallLayerNormEpsilon = 1e-12

	pretrainedBERTCUDAFoundationContract = "full_device_pretrained_bert_encoder.v1"

	selectedBGECUDAContractVersion       = "eos.cuda.bge_small_en_v1_5.hidden_contract.v1"
	selectedBGEPackageSHA256             = "841b0d851c06290daeeab4bf4d25cb1dd7bb87920316dac950e1b556a3bae763"
	selectedBGEPackageIdentitySHA256     = "a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637"
	selectedBGEModuleSHA256              = "58d96cd05005ea2a419651ab5f624c36cbe6fe0e5b9bfe23b45e90e9b4a26d02"
	selectedBGEWeightsSHA256             = "4ec787db0ed5bd2f0ec692806903e5fced0bdc38f23b0e1ac15e1d3399b80636"
	selectedBGEModelName                 = "BAAI/bge-small-en-v1.5"
	selectedBGEArchitecture              = "BertModel"
	selectedBGEQueryPrefix               = "Represent this sentence for searching relevant passages: "
	selectedBGEDocumentPrefix            = ""
	selectedBGEContractFingerprintSHA256 = "a52117d26cd67c7f67484c1181c8ce4e33b723cda6095e469ecb06d586cd606e"
)

type bertCUDAEncoderFoundationStatus struct {
	Contract          string
	ShapeValidated    bool
	WeightsPlanned    bool
	FullDeviceReady   bool
	MissingComponents []string
}

func newBERTCUDAEncoderFoundationStatus() bertCUDAEncoderFoundationStatus {
	return bertCUDAEncoderFoundationStatus{
		Contract:          pretrainedBERTCUDAFoundationContract,
		FullDeviceReady:   false,
		MissingComponents: []string{"public fail-closed device encoder claim"},
	}
}

type bertCUDAWeightRole int

const (
	bertCUDAWeightEmbedding bertCUDAWeightRole = iota
	bertCUDAWeightVector
	bertCUDAWeightDenseMatrix
	bertCUDAWeightDenseBias
)

type bertCUDAResidentWeight struct {
	Name       string
	Role       bertCUDAWeightRole
	Shape      []int
	InputIndex int
	Layer      int
	Slot       string
}

type bertCUDAResidentWeightPlan struct {
	Weights []bertCUDAResidentWeight
}

type bertCUDAFullEncoderTransferStats struct {
	ResidentUploadedBytes         int64
	InputUploadedBytes            int64
	UploadedBytes                 int64
	DownloadedBytes               int64
	FinalDownloadedBytes          int64
	StatusDownloadedBytes         int64
	IntermediateDownloadedBytes   int64
	ResidentWeightBytesReferenced int64
	WorkspaceBytes                int64
	MaxBatchTokens                int
	Layers                        int
	ResidentCacheHits             int64
	ResidentCacheMisses           int64
	ResidentBindNanos             int64
	RunNanos                      int64
	ColdResidentBind              bool
	ContractFingerprint           string
	WeightFingerprint             string
}

type bertCUDASelectedContract struct {
	ContractFingerprint string
	WeightFingerprint   string
	WeightFingerprints  map[int]string
	Provenance          bertCUDASelectedPackageProvenance
	CacheKey            string
	CacheHit            bool
}

type bertCUDASelectedPackageProvenance struct {
	PackageSHA256       string
	ModuleSHA256        string
	WeightsSHA256       string
	PackageIdentity     string
	WeightSetGeneration string
	RoleSchema          string
	QueryRole           string
	DocumentRole        string
	QueryPrefix         string
	DocumentPrefix      string
	Pooling             string
	Normalization       string
	MaxLength           int
	NativeDim           int
}

type bertCUDAResidentBindingCache struct {
	DType               string
	Shape               []int
	Fingerprint         string
	WeightSetGeneration string
	Bytes               int64
}

type bertCUDAResidentBindStats struct {
	UploadedBytes       int64
	CacheHits           int64
	CacheMisses         int64
	ResidentWeightBytes int64
	BindNanos           int64
	ColdBind            bool
}

func bertCUDA12LayerHiddenGateEnabled() bool {
	switch os.Getenv("EOS_BERT_CUDA_12LAYER_HIDDEN") {
	case "1", "true", "TRUE", "yes", "YES", "on", "ON":
		return true
	default:
		return false
	}
}

func bertCUDAMaxBatchTokensFromEnv(defaultValue int) (int, error) {
	raw := os.Getenv("EOS_BERT_CUDA_MAX_BATCH_TOKENS")
	if raw == "" {
		return defaultValue, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return 0, fmt.Errorf("EOS_BERT_CUDA_MAX_BATCH_TOKENS=%q is invalid; want positive integer", raw)
	}
	return value, nil
}

func validateSelectedBGECUDAContract(mod *eosartifact.Module, step eosartifact.Step, inputs []*backend.Tensor) (bertCUDASelectedContract, error) {
	if err := validateSelectedBGEContractShapeAndPlan(mod, step, inputs); err != nil {
		return bertCUDASelectedContract{}, err
	}
	provenance, err := validateSelectedBGEPackageProvenance(mod)
	if err != nil {
		return bertCUDASelectedContract{}, err
	}
	weightFingerprint, perWeight, err := bgeSelectedWeightFingerprint(inputs)
	if err != nil {
		return bertCUDASelectedContract{}, err
	}
	contractFingerprint := bgeSelectedContractFingerprint(mod, step, provenance, weightFingerprint)
	if selectedBGEContractFingerprintSHA256 != "" && selectedBGEContractFingerprintSHA256 != "TODO" && contractFingerprint != selectedBGEContractFingerprintSHA256 {
		return bertCUDASelectedContract{}, fmt.Errorf("selected BGE CUDA contract fingerprint mismatch: got %s want %s", contractFingerprint, selectedBGEContractFingerprintSHA256)
	}
	return bertCUDASelectedContract{
		ContractFingerprint: contractFingerprint,
		WeightFingerprint:   weightFingerprint,
		WeightFingerprints:  perWeight,
		Provenance:          provenance,
	}, nil
}

func validateSelectedBGEPackageProvenance(mod *eosartifact.Module) (bertCUDASelectedPackageProvenance, error) {
	if mod == nil {
		return bertCUDASelectedPackageProvenance{}, fmt.Errorf("selected BGE CUDA contract requires package provenance")
	}
	if got, ok := moduleBoolMetadata(mod, "pretrained_bert_package_provenance_verified"); !ok || !got {
		return bertCUDASelectedPackageProvenance{}, fmt.Errorf("selected BGE CUDA contract package provenance is not verified")
	}
	checkString := func(key, want string) (string, error) {
		got := moduleStringMetadata(mod, key)
		if got != want {
			return "", fmt.Errorf("selected BGE CUDA contract package provenance %s=%q, want %q", key, got, want)
		}
		return got, nil
	}
	checkInt := func(key string, want int) (int, error) {
		value, ok := mod.Metadata[key]
		if !ok {
			return 0, fmt.Errorf("selected BGE CUDA contract package provenance %s is missing", key)
		}
		got, err := metadataInt(value)
		if err != nil {
			return 0, fmt.Errorf("selected BGE CUDA contract package provenance %s: %w", key, err)
		}
		if got != want {
			return 0, fmt.Errorf("selected BGE CUDA contract package provenance %s=%d, want %d", key, got, want)
		}
		return got, nil
	}
	version, err := checkString("pretrained_bert_package_provenance_version", "manta/pretrained-bert-package-runtime-provenance/v1")
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	_ = version
	packageSHA, err := checkString("pretrained_bert_package_sha256", selectedBGEPackageSHA256)
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	identity, err := checkString("pretrained_bert_package_identity_sha256", selectedBGEPackageIdentitySHA256)
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	moduleSHA, err := checkString("pretrained_bert_package_module_sha256", selectedBGEModuleSHA256)
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	weightsSHA, err := checkString("pretrained_bert_package_weights_sha256", selectedBGEWeightsSHA256)
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	generation := moduleStringMetadata(mod, "pretrained_bert_package_weight_set_generation")
	if generation == "" {
		return bertCUDASelectedPackageProvenance{}, fmt.Errorf("selected BGE CUDA contract package provenance pretrained_bert_package_weight_set_generation is missing")
	}
	roleSchema, err := checkString("pretrained_bert_retrieval_role_schema", "manta.pretrained_bert_retrieval_role_contract.v1")
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	queryRole, err := checkString("pretrained_bert_retrieval_query_role", "query")
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	documentRole, err := checkString("pretrained_bert_retrieval_document_role", "document")
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	queryPrefix, err := checkString("pretrained_bert_retrieval_query_prefix", selectedBGEQueryPrefix)
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	documentPrefix, err := checkString("pretrained_bert_retrieval_document_prefix", selectedBGEDocumentPrefix)
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	pooling, err := checkString("pretrained_bert_retrieval_pooling", "cls")
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	normalization, err := checkString("pretrained_bert_retrieval_normalization", "l2")
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	maxLength, err := checkInt("pretrained_bert_retrieval_max_length", bgeSmallMaxPositions)
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	nativeDim, err := checkInt("pretrained_bert_retrieval_native_dim", bgeSmallHiddenSize)
	if err != nil {
		return bertCUDASelectedPackageProvenance{}, err
	}
	return bertCUDASelectedPackageProvenance{
		PackageSHA256:       packageSHA,
		ModuleSHA256:        moduleSHA,
		WeightsSHA256:       weightsSHA,
		PackageIdentity:     identity,
		WeightSetGeneration: generation,
		RoleSchema:          roleSchema,
		QueryRole:           queryRole,
		DocumentRole:        documentRole,
		QueryPrefix:         queryPrefix,
		DocumentPrefix:      documentPrefix,
		Pooling:             pooling,
		Normalization:       normalization,
		MaxLength:           maxLength,
		NativeDim:           nativeDim,
	}, nil
}

func validateSelectedBGEContractShapeAndPlan(mod *eosartifact.Module, step eosartifact.Step, inputs []*backend.Tensor) error {
	if _, err := validateBGEPretrainedBERTEmbedderStep(step, inputs); err != nil {
		return err
	}
	if mod == nil {
		return fmt.Errorf("selected BGE CUDA contract requires module metadata")
	}
	if mod.Name != "pretrained_bert_embedder" {
		return fmt.Errorf("selected BGE CUDA contract module name=%q, want %q", mod.Name, "pretrained_bert_embedder")
	}
	for _, check := range []struct {
		key  string
		want string
	}{
		{"model_name", selectedBGEModelName},
		{"architecture", selectedBGEArchitecture},
		{"pooling", "cls"},
		{"normalization", "l2"},
	} {
		got, _ := mod.Metadata[check.key].(string)
		if got != check.want {
			return fmt.Errorf("selected BGE CUDA contract metadata %s=%q, want %q", check.key, got, check.want)
		}
	}
	if value, ok := mod.Metadata["max_length"]; ok {
		maxLength, err := metadataInt(value)
		if err != nil {
			return fmt.Errorf("selected BGE CUDA contract metadata max_length: %w", err)
		}
		if maxLength != bgeSmallMaxPositions {
			return fmt.Errorf("selected BGE CUDA contract metadata max_length=%d, want %d", maxLength, bgeSmallMaxPositions)
		}
	}
	if step.Entry != "bert_embed" {
		return fmt.Errorf("selected BGE CUDA contract step entry=%q, want %q", step.Entry, "bert_embed")
	}
	if step.Name != "bert_embedder_reference" {
		return fmt.Errorf("selected BGE CUDA contract step name=%q, want %q", step.Name, "bert_embedder_reference")
	}
	if len(step.Outputs) != 1 || step.Outputs[0] != "embeddings" {
		return fmt.Errorf("selected BGE CUDA contract step outputs=%v, want [embeddings]", step.Outputs)
	}
	expectedAttrs := map[string]string{
		"epsilon":             "1e-12",
		"hidden_act":          "gelu",
		"num_attention_heads": "12",
		"num_hidden_layers":   "12",
		"pooling":             "cls",
		"normalization":       "l2",
	}
	if len(step.Attributes) != len(expectedAttrs) {
		return fmt.Errorf("selected BGE CUDA contract attr count=%d, want %d", len(step.Attributes), len(expectedAttrs))
	}
	for key, want := range expectedAttrs {
		if got := step.Attributes[key]; got != want {
			return fmt.Errorf("selected BGE CUDA contract attr %s=%q, want %q", key, got, want)
		}
	}
	expectedInputs := bgeExpectedStepInputs()
	if len(step.Inputs) != len(expectedInputs) {
		return fmt.Errorf("selected BGE CUDA contract step input count=%d, want %d", len(step.Inputs), len(expectedInputs))
	}
	for i := range expectedInputs {
		if step.Inputs[i] != expectedInputs[i] {
			return fmt.Errorf("selected BGE CUDA contract step input[%d]=%q, want %q", i, step.Inputs[i], expectedInputs[i])
		}
	}
	return nil
}

func bgeSelectedContractCacheKey(mod *eosartifact.Module, step eosartifact.Step, inputs []*backend.Tensor, provenance bertCUDASelectedPackageProvenance, weightFingerprint string) (string, error) {
	if err := validateSelectedBGEContractShapeAndPlan(mod, step, inputs); err != nil {
		return "", err
	}
	h := sha256.New()
	writeHashField(h, selectedBGECUDAContractVersion)
	writeHashField(h, "resident_cache_key.v1")
	writeHashField(h, bgeSelectedModuleStepFingerprint(mod, step, provenance))
	writeHashField(h, provenance.WeightSetGeneration)
	writeHashField(h, weightFingerprint)
	return hex.EncodeToString(h.Sum(nil)), nil
}

func bgeSelectedModuleStepFingerprint(mod *eosartifact.Module, step eosartifact.Step, provenance bertCUDASelectedPackageProvenance) string {
	h := sha256.New()
	writeHashField(h, selectedBGECUDAContractVersion)
	writeHashField(h, provenance.PackageSHA256)
	writeHashField(h, provenance.PackageIdentity)
	writeHashField(h, provenance.ModuleSHA256)
	writeHashField(h, provenance.WeightsSHA256)
	writeHashField(h, provenance.WeightSetGeneration)
	writeHashField(h, provenance.RoleSchema)
	writeHashField(h, provenance.QueryRole)
	writeHashField(h, provenance.DocumentRole)
	writeHashField(h, provenance.QueryPrefix)
	writeHashField(h, provenance.DocumentPrefix)
	writeHashField(h, provenance.Pooling)
	writeHashField(h, provenance.Normalization)
	writeHashField(h, strconv.Itoa(provenance.MaxLength))
	writeHashField(h, strconv.Itoa(provenance.NativeDim))
	writeHashField(h, mod.Name)
	writeHashField(h, moduleStringMetadata(mod, "model_name"))
	writeHashField(h, moduleStringMetadata(mod, "architecture"))
	writeHashField(h, moduleStringMetadata(mod, "pooling"))
	writeHashField(h, moduleStringMetadata(mod, "normalization"))
	if value, ok := mod.Metadata["max_length"]; ok {
		writeHashField(h, fmt.Sprint(value))
	}
	writeHashField(h, string(step.Kind))
	writeHashField(h, step.Entry)
	writeHashField(h, step.Name)
	for _, input := range step.Inputs {
		writeHashField(h, input)
	}
	for _, output := range step.Outputs {
		writeHashField(h, output)
	}
	keys := make([]string, 0, len(step.Attributes))
	for key := range step.Attributes {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		writeHashField(h, key)
		writeHashField(h, step.Attributes[key])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func bgeExpectedStepInputs() []string {
	specs := bgeCUDAWeightSpecs()
	inputs := make([]string, 3+len(specs))
	inputs[0] = "input_ids"
	inputs[1] = "attention_mask"
	inputs[2] = "token_type_ids"
	for _, spec := range specs {
		if spec.layer >= 0 {
			inputs[spec.inputIndex] = fmt.Sprintf("encoder_layer_%d_%s", spec.layer, spec.slot)
		} else {
			inputs[spec.inputIndex] = spec.slot
		}
	}
	return inputs
}

func bgeSelectedWeightFingerprint(inputs []*backend.Tensor) (string, map[int]string, error) {
	h := sha256.New()
	writeHashField(h, selectedBGECUDAContractVersion)
	perWeight := make(map[int]string, len(bgeCUDAWeightSpecs()))
	for _, spec := range bgeCUDAWeightSpecs() {
		if spec.inputIndex >= len(inputs) {
			return "", nil, fmt.Errorf("selected BGE CUDA contract missing weight input %d", spec.inputIndex)
		}
		tensor := inputs[spec.inputIndex]
		digest, err := tensorF32ContentFingerprint(tensor)
		if err != nil {
			return "", nil, fmt.Errorf("selected BGE CUDA contract weight %s fingerprint: %w", spec.slot, err)
		}
		perWeight[spec.inputIndex] = digest
		writeHashField(h, strconv.Itoa(spec.inputIndex))
		writeHashField(h, spec.slot)
		writeHashField(h, tensor.DType)
		for _, dim := range tensor.Shape {
			writeHashField(h, strconv.Itoa(dim))
		}
		writeHashField(h, digest)
	}
	return hex.EncodeToString(h.Sum(nil)), perWeight, nil
}

func bgeSelectedContractFingerprint(mod *eosartifact.Module, step eosartifact.Step, provenance bertCUDASelectedPackageProvenance, weightFingerprint string) string {
	h := sha256.New()
	writeHashField(h, bgeSelectedModuleStepFingerprint(mod, step, provenance))
	writeHashField(h, weightFingerprint)
	return hex.EncodeToString(h.Sum(nil))
}

func tensorF32ContentFingerprint(t *backend.Tensor) (string, error) {
	if t == nil {
		return "", fmt.Errorf("tensor is nil")
	}
	if err := checkedShapeProduct("selected BGE CUDA contract tensor", t.Shape, len(t.F32)); err != nil {
		return "", err
	}
	h := sha256.New()
	writeHashField(h, t.DType)
	for _, dim := range t.Shape {
		writeHashField(h, strconv.Itoa(dim))
	}
	var buf [4]byte
	for _, value := range t.F32 {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(value))
		if _, err := h.Write(buf[:]); err != nil {
			return "", err
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeHashField(h hash.Hash, value string) {
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(value)))
	_, _ = h.Write(lenBuf[:])
	_, _ = h.Write([]byte(value))
}

func moduleStringMetadata(mod *eosartifact.Module, key string) string {
	if mod == nil || mod.Metadata == nil {
		return ""
	}
	value, _ := mod.Metadata[key].(string)
	return value
}

func moduleBoolMetadata(mod *eosartifact.Module, key string) (bool, bool) {
	if mod == nil || mod.Metadata == nil {
		return false, false
	}
	switch value := mod.Metadata[key].(type) {
	case bool:
		return value, true
	case string:
		switch value {
		case "true", "1", "yes", "on":
			return true, true
		case "false", "0", "no", "off":
			return false, true
		default:
			return false, false
		}
	default:
		return false, false
	}
}

func metadataInt(value any) (int, error) {
	switch v := value.(type) {
	case int:
		return v, nil
	case int64:
		if v < int64(^uint(0)>>1)*-1-1 || v > int64(^uint(0)>>1) {
			return 0, fmt.Errorf("%d overflows int", v)
		}
		return int(v), nil
	case float64:
		if math.Trunc(v) != v {
			return 0, fmt.Errorf("%v is not an integer", v)
		}
		if v < float64(-int(^uint(0)>>1)-1) || v > float64(int(^uint(0)>>1)) {
			return 0, fmt.Errorf("%v overflows int", v)
		}
		return int(v), nil
	case string:
		parsed, err := strconv.Atoi(v)
		if err != nil {
			return 0, err
		}
		return parsed, nil
	default:
		return 0, fmt.Errorf("unsupported type %T", value)
	}
}

func checkedShapeProduct(label string, shape []int, backing int) error {
	product := 1
	maxInt := int(^uint(0) >> 1)
	for _, dim := range shape {
		if dim < 0 {
			return fmt.Errorf("%s dimension %d is negative", label, dim)
		}
		if dim != 0 && product > maxInt/dim {
			return fmt.Errorf("%s product overflows int for shape %v", label, shape)
		}
		product *= dim
	}
	if product != backing {
		return fmt.Errorf("%s backing length %d does not match shape %v", label, backing, shape)
	}
	return nil
}

func validateBGEPretrainedBERTEmbedderStep(step eosartifact.Step, inputs []*backend.Tensor) (bertCUDAEncoderFoundationStatus, error) {
	status := newBERTCUDAEncoderFoundationStatus()
	if step.Kind != eosartifact.StepBERTEmbedder {
		return status, fmt.Errorf("cuda bert foundation supports only StepBERTEmbedder, got %s", step.Kind)
	}
	layers, err := bertStepIntAttr(step.Attributes, "num_hidden_layers", 1)
	if err != nil {
		return status, err
	}
	heads, err := bertStepIntAttr(step.Attributes, "num_attention_heads", 1)
	if err != nil {
		return status, err
	}
	epsilon, err := bertStepFloatAttr(step.Attributes, "epsilon", bgeSmallLayerNormEpsilon)
	if err != nil {
		return status, err
	}
	hiddenAct := step.Attributes["hidden_act"]
	if hiddenAct == "" {
		hiddenAct = "gelu"
	}
	pooling := step.Attributes["pooling"]
	if pooling == "" {
		pooling = "masked_mean"
	}
	normalization := step.Attributes["normalization"]
	if normalization == "" {
		normalization = "l2"
	}
	if layers != bgeSmallLayers || heads != bgeSmallHeads || hiddenAct != "gelu" || pooling != "cls" || normalization != "l2" || epsilon != bgeSmallLayerNormEpsilon {
		return status, fmt.Errorf("unsupported BGE CUDA encoder attrs: layers=%d heads=%d hidden_act=%q pooling=%q normalization=%q epsilon=%g", layers, heads, hiddenAct, pooling, normalization, epsilon)
	}
	expected := 8 + bgeSmallLayers*16
	if len(inputs) != expected {
		return status, fmt.Errorf("BGE CUDA encoder expects %d tensors, got %d", expected, len(inputs))
	}
	if err := validateBGEInputIDs(inputs[0], "input_ids"); err != nil {
		return status, err
	}
	if err := validateBGEInputIDs(inputs[1], "attention_mask"); err != nil {
		return status, err
	}
	if err := validateBGEInputIDs(inputs[2], "token_type_ids"); err != nil {
		return status, err
	}
	if !inputs[0].EqualShape(inputs[1]) || !inputs[0].EqualShape(inputs[2]) {
		return status, fmt.Errorf("BGE CUDA encoder input_id, attention_mask, token_type_id shapes must match")
	}
	if inputs[0].Shape[1] > bgeSmallMaxPositions {
		return status, fmt.Errorf("BGE CUDA encoder token length %d exceeds max positions %d", inputs[0].Shape[1], bgeSmallMaxPositions)
	}
	if err := validateBGEInputIDValues(inputs[0], inputs[1], inputs[2]); err != nil {
		return status, err
	}
	for _, check := range bgeCUDAWeightSpecs()[:5] {
		if err := validateBGEFloatTensor(inputs[check.inputIndex], check.slot, check.shape); err != nil {
			return status, err
		}
	}
	for _, check := range bgeCUDAWeightSpecs()[5:] {
		if err := validateBGEFloatTensor(inputs[check.inputIndex], fmt.Sprintf("encoder_layer_%d_%s", check.layer, check.slot), check.shape); err != nil {
			return status, err
		}
	}
	status.ShapeValidated = true
	return status, nil
}

func planBGEPretrainedBERTResidentWeights(step eosartifact.Step, inputs []*backend.Tensor) (bertCUDAResidentWeightPlan, bertCUDAEncoderFoundationStatus, error) {
	status, err := validateBGEPretrainedBERTEmbedderStep(step, inputs)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, status, err
	}
	if len(step.Inputs) != len(inputs) {
		return bertCUDAResidentWeightPlan{}, status, fmt.Errorf("BGE CUDA encoder resident weight planning requires %d stable input names, got %d", len(inputs), len(step.Inputs))
	}
	plan := bertCUDAResidentWeightPlan{Weights: make([]bertCUDAResidentWeight, 0, len(inputs)-3)}
	seen := map[string]int{}
	for _, spec := range bgeCUDAWeightSpecs() {
		name := step.Inputs[spec.inputIndex]
		if name == "" {
			return bertCUDAResidentWeightPlan{}, status, fmt.Errorf("BGE CUDA encoder resident weight input %d (%s) is missing a stable artifact name", spec.inputIndex, spec.slot)
		}
		if previous, ok := seen[name]; ok {
			return bertCUDAResidentWeightPlan{}, status, fmt.Errorf("BGE CUDA encoder resident weight name %q is duplicated at inputs %d and %d", name, previous, spec.inputIndex)
		}
		seen[name] = spec.inputIndex
		plan.Weights = append(plan.Weights, bertCUDAResidentWeight{
			Name:       name,
			Role:       spec.role,
			Shape:      append([]int(nil), inputs[spec.inputIndex].Shape...),
			InputIndex: spec.inputIndex,
			Layer:      spec.layer,
			Slot:       spec.slot,
		})
	}
	status.WeightsPlanned = true
	return plan, status, nil
}

type bertCUDAWeightSpec struct {
	inputIndex int
	layer      int
	slot       string
	shape      []int
	role       bertCUDAWeightRole
}

func bgeCUDAWeightSpecs() []bertCUDAWeightSpec {
	specs := []bertCUDAWeightSpec{
		{3, -1, "token_embeddings", []int{bgeSmallVocabSize, bgeSmallHiddenSize}, bertCUDAWeightEmbedding},
		{4, -1, "position_embeddings", []int{bgeSmallMaxPositions, bgeSmallHiddenSize}, bertCUDAWeightEmbedding},
		{5, -1, "token_type_embeddings", []int{bgeSmallTypeVocabSize, bgeSmallHiddenSize}, bertCUDAWeightEmbedding},
		{6, -1, "embedding_layernorm_weight", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
		{7, -1, "embedding_layernorm_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
	}
	layerSlots := []struct {
		slot  string
		shape []int
		role  bertCUDAWeightRole
	}{
		{"attention_query_weight", []int{bgeSmallHiddenSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"attention_query_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"attention_key_weight", []int{bgeSmallHiddenSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"attention_key_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"attention_value_weight", []int{bgeSmallHiddenSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"attention_value_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"attention_output_weight", []int{bgeSmallHiddenSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"attention_output_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"attention_layernorm_weight", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
		{"attention_layernorm_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
		{"intermediate_weight", []int{bgeSmallIntermediateSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"intermediate_bias", []int{bgeSmallIntermediateSize}, bertCUDAWeightDenseBias},
		{"output_weight", []int{bgeSmallHiddenSize, bgeSmallIntermediateSize}, bertCUDAWeightDenseMatrix},
		{"output_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"output_layernorm_weight", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
		{"output_layernorm_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
	}
	for layer := 0; layer < bgeSmallLayers; layer++ {
		for offset, slot := range layerSlots {
			specs = append(specs, bertCUDAWeightSpec{8 + layer*16 + offset, layer, slot.slot, slot.shape, slot.role})
		}
	}
	return specs
}

func bertStepIntAttr(attrs map[string]string, name string, defaultValue int) (int, error) {
	if attrs == nil || attrs[name] == "" {
		return defaultValue, nil
	}
	value, err := strconv.Atoi(attrs[name])
	if err != nil || value <= 0 {
		return 0, fmt.Errorf("BGE CUDA encoder attr %s=%q is invalid", name, attrs[name])
	}
	return value, nil
}

func bertStepFloatAttr(attrs map[string]string, name string, defaultValue float64) (float64, error) {
	if attrs == nil || attrs[name] == "" {
		return defaultValue, nil
	}
	value, err := strconv.ParseFloat(attrs[name], 64)
	if err != nil || value < 0 {
		return 0, fmt.Errorf("BGE CUDA encoder attr %s=%q is invalid", name, attrs[name])
	}
	return value, nil
}

func validateBGEInputIDs(t *backend.Tensor, name string) error {
	if t == nil || t.DType != "i32" || len(t.Shape) != 2 || t.Shape[0] <= 0 || t.Shape[1] <= 0 || t.Elements() != len(t.I32) {
		return fmt.Errorf("BGE CUDA encoder %s must be i32 [B,T] with matching backing data", name)
	}
	return nil
}

func validateBGEInputIDValues(inputIDs, attentionMask, tokenTypeIDs *backend.Tensor) error {
	for i, tokenID := range inputIDs.I32 {
		if tokenID < 0 || int(tokenID) >= bgeSmallVocabSize {
			return fmt.Errorf("BGE CUDA encoder input_ids[%d]=%d out of range [0,%d)", i, tokenID, bgeSmallVocabSize)
		}
		positionID := i % inputIDs.Shape[1]
		if positionID < 0 || positionID >= bgeSmallMaxPositions {
			return fmt.Errorf("BGE CUDA encoder implicit position_ids[%d]=%d out of range [0,%d)", i, positionID, bgeSmallMaxPositions)
		}
		tokenTypeID := tokenTypeIDs.I32[i]
		if tokenTypeID < 0 || int(tokenTypeID) >= bgeSmallTypeVocabSize {
			return fmt.Errorf("BGE CUDA encoder token_type_ids[%d]=%d out of range [0,%d)", i, tokenTypeID, bgeSmallTypeVocabSize)
		}
		mask := attentionMask.I32[i]
		if mask != 0 && mask != 1 {
			return fmt.Errorf("BGE CUDA encoder attention_mask[%d]=%d is invalid, want 0 or 1", i, mask)
		}
	}
	return nil
}

func validateBGEFloatTensor(t *backend.Tensor, name string, shape []int) error {
	if t == nil {
		return fmt.Errorf("BGE CUDA encoder %s must be f32-compatible shape %v", name, shape)
	}
	if t.DType != "f32" && t.DType != "f16" {
		return fmt.Errorf("BGE CUDA encoder %s dtype %q is not f32-compatible", name, t.DType)
	}
	if len(t.Shape) != len(shape) || t.Elements() != len(t.F32) {
		return fmt.Errorf("BGE CUDA encoder %s must be f32-compatible shape %v", name, shape)
	}
	for i := range shape {
		if t.Shape[i] != shape[i] {
			return fmt.Errorf("BGE CUDA encoder %s shape %v, want %v", name, t.Shape, shape)
		}
	}
	return nil
}
