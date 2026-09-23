package eosruntime

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	AOQTResidualLowRankTransformSchema  = "eos.aoqt.residual_lowrank_transform.v1"
	AOQTResidualLowRankTransformVersion = "eos-d384-residual-lowrank-screen-v1"
	AOQTResidualLowRankDim              = 384
)

const (
	AOQTResidualLowRankRoleQueryOnly    = "query-only"
	AOQTResidualLowRankRoleDocumentOnly = "document-only"
	AOQTResidualLowRankRoleSymmetric    = "symmetric"
)

type AOQTResidualLowRankParameters struct {
	Schema  string      `json:"schema"`
	Version string      `json:"version"`
	Dim     int         `json:"dim"`
	Rank    int         `json:"rank"`
	Role    string      `json:"role"`
	Alpha   float64     `json:"alpha"`
	Seed    string      `json:"seed"`
	A       [][]float32 `json:"a"`
	B       [][]float32 `json:"b"`
}

type AOQTResidualLowRankCacheTransformConfig struct {
	Parameters      AOQTResidualLowRankParameters
	DocVectorPath   string
	QueryVectorPath string
	OutputDocPath   string
	OutputQueryPath string
	SidecarPath     string
	Dataset         string
	Artifact        string
}

type AOQTResidualLowRankCacheBinding struct {
	Schema          string                              `json:"schema"`
	Version         string                              `json:"version"`
	Dataset         string                              `json:"dataset,omitempty"`
	Artifact        string                              `json:"artifact,omitempty"`
	ParameterSHA256 string                              `json:"parameter_sha256"`
	Parameters      AOQTResidualLowRankParameterBinding `json:"parameters"`
	Inputs          AOQTVectorCacheTransformIO          `json:"inputs"`
	Outputs         AOQTVectorCacheTransformIO          `json:"outputs"`
}

type AOQTResidualLowRankParameterBinding struct {
	Dim   int     `json:"dim"`
	Rank  int     `json:"rank"`
	Role  string  `json:"role"`
	Alpha float64 `json:"alpha"`
	Seed  string  `json:"seed"`
}

func AOQTResidualLowRankRanks() []int {
	return []int{4, 8}
}

func AOQTResidualLowRankRoles() []string {
	return []string{AOQTResidualLowRankRoleQueryOnly, AOQTResidualLowRankRoleDocumentOnly, AOQTResidualLowRankRoleSymmetric}
}

func AOQTResidualLowRankMagnitudes() []float64 {
	return []float64{0.00125, 0.0025, 0.005, 0.01}
}

func NewAOQTResidualLowRankParameters(rank int, role string, alpha float64, seed string) (AOQTResidualLowRankParameters, error) {
	header := AOQTResidualLowRankParameters{
		Schema:  AOQTResidualLowRankTransformSchema,
		Version: AOQTResidualLowRankTransformVersion,
		Dim:     AOQTResidualLowRankDim,
		Rank:    rank,
		Role:    role,
		Alpha:   alpha,
		Seed:    strings.TrimSpace(seed),
	}
	if header.Seed == "" {
		return AOQTResidualLowRankParameters{}, fmt.Errorf("AOQT residual low-rank seed is required")
	}
	if err := validateAOQTResidualLowRankHeader(header); err != nil {
		return AOQTResidualLowRankParameters{}, err
	}
	p := AOQTResidualLowRankParameters{
		Schema:  AOQTResidualLowRankTransformSchema,
		Version: AOQTResidualLowRankTransformVersion,
		Dim:     AOQTResidualLowRankDim,
		Rank:    rank,
		Role:    role,
		Alpha:   alpha,
		Seed:    header.Seed,
		A:       make([][]float32, rank),
		B:       make([][]float32, AOQTResidualLowRankDim),
	}
	for k := 0; k < rank; k++ {
		p.A[k] = make([]float32, AOQTResidualLowRankDim)
		for d := 0; d < AOQTResidualLowRankDim; d++ {
			p.A[k][d] = aoqtResidualDeterministicValue(p.Seed, "a", k, d)
		}
		aoqtResidualNormalizeInPlace(p.A[k])
	}
	for d := 0; d < AOQTResidualLowRankDim; d++ {
		p.B[d] = make([]float32, rank)
		for k := 0; k < rank; k++ {
			p.B[d][k] = aoqtResidualDeterministicValue(p.Seed, "b", d, k)
		}
	}
	if err := p.Validate(); err != nil {
		return AOQTResidualLowRankParameters{}, err
	}
	return p, nil
}

func (p AOQTResidualLowRankParameters) Validate() error {
	if err := validateAOQTResidualLowRankHeader(p); err != nil {
		return err
	}
	if len(p.A) != p.Rank {
		return fmt.Errorf("AOQT residual low-rank A rows = %d, want rank %d", len(p.A), p.Rank)
	}
	for i, row := range p.A {
		if len(row) != p.Dim {
			return fmt.Errorf("AOQT residual low-rank A[%d] dim = %d, want %d", i, len(row), p.Dim)
		}
		if err := validateAOQTFiniteVector(row, fmt.Sprintf("AOQT residual low-rank A[%d]", i)); err != nil {
			return err
		}
	}
	if len(p.B) != p.Dim {
		return fmt.Errorf("AOQT residual low-rank B rows = %d, want dim %d", len(p.B), p.Dim)
	}
	for i, row := range p.B {
		if len(row) != p.Rank {
			return fmt.Errorf("AOQT residual low-rank B[%d] rank dim = %d, want %d", i, len(row), p.Rank)
		}
		if err := validateAOQTFiniteVector(row, fmt.Sprintf("AOQT residual low-rank B[%d]", i)); err != nil {
			return err
		}
	}
	return nil
}

func (p AOQTResidualLowRankParameters) SHA256() (string, error) {
	return aoqtCanonicalSHA256(p)
}

func (p AOQTResidualLowRankParameters) Apply(vector []float32) ([]float32, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if len(vector) != p.Dim {
		return nil, fmt.Errorf("AOQT residual low-rank vector dim = %d, want %d", len(vector), p.Dim)
	}
	if err := validateAOQTFiniteVector(vector, "AOQT residual low-rank vector"); err != nil {
		return nil, err
	}
	norm := aoqtResidualNorm(vector)
	if math.Abs(norm-1) > 1e-3 {
		return nil, fmt.Errorf("AOQT residual low-rank input vector must be unit-normalized, norm=%g", norm)
	}
	latent := make([]float64, p.Rank)
	for k := 0; k < p.Rank; k++ {
		for d := 0; d < p.Dim; d++ {
			latent[k] += float64(p.A[k][d]) * float64(vector[d])
		}
	}
	residual := make([]float64, p.Dim)
	for d := 0; d < p.Dim; d++ {
		for k := 0; k < p.Rank; k++ {
			residual[d] += float64(p.B[d][k]) * latent[k]
		}
	}
	dot := 0.0
	for d := 0; d < p.Dim; d++ {
		dot += float64(vector[d]) * residual[d]
	}
	out := make([]float32, p.Dim)
	outNormSq := 0.0
	for d := 0; d < p.Dim; d++ {
		tangent := residual[d] - dot*float64(vector[d])
		value := float64(vector[d]) + p.Alpha*tangent
		if !isFinite64(value) {
			return nil, fmt.Errorf("AOQT residual low-rank output value %d is not finite", d)
		}
		out[d] = float32(value)
		outNormSq += value * value
	}
	outNorm := math.Sqrt(outNormSq)
	if !isFinite64(outNorm) || outNorm == 0 {
		return nil, fmt.Errorf("AOQT residual low-rank output norm is invalid")
	}
	for d := range out {
		out[d] = float32(float64(out[d]) / outNorm)
	}
	if math.Abs(aoqtResidualNorm(out)-1) > 2e-5 {
		return nil, fmt.Errorf("AOQT residual low-rank output norm policy failed")
	}
	return out, nil
}

func TransformAOQTResidualLowRankVectorCaches(cfg AOQTResidualLowRankCacheTransformConfig) (AOQTResidualLowRankCacheBinding, error) {
	if err := cfg.Parameters.Validate(); err != nil {
		return AOQTResidualLowRankCacheBinding{}, err
	}
	if strings.TrimSpace(cfg.DocVectorPath) == "" || strings.TrimSpace(cfg.QueryVectorPath) == "" {
		return AOQTResidualLowRankCacheBinding{}, fmt.Errorf("AOQT residual low-rank input doc and query vector paths are required")
	}
	if strings.TrimSpace(cfg.OutputDocPath) == "" || strings.TrimSpace(cfg.OutputQueryPath) == "" || strings.TrimSpace(cfg.SidecarPath) == "" {
		return AOQTResidualLowRankCacheBinding{}, fmt.Errorf("AOQT residual low-rank output doc/query vector paths and sidecar path are required")
	}
	docAudit, err := validateAOQTResidualLowRankCacheFile(cfg.DocVectorPath)
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, fmt.Errorf("validate residual low-rank document vectors: %w", err)
	}
	queryAudit, err := validateAOQTResidualLowRankCacheFile(cfg.QueryVectorPath)
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, fmt.Errorf("validate residual low-rank query vectors: %w", err)
	}
	cleanup := []string{}
	cleanupOnFailure := true
	defer func() {
		if cleanupOnFailure {
			for _, path := range cleanup {
				_ = os.Remove(path)
			}
		}
	}()
	docCount, err := transformAOQTResidualLowRankCacheFile(cfg.Parameters, cfg.DocVectorPath, cfg.OutputDocPath, aoqtResidualLowRankApplyDocs(cfg.Parameters.Role))
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, fmt.Errorf("transform residual low-rank document vectors: %w", err)
	}
	cleanup = append(cleanup, cfg.OutputDocPath)
	queryCount, err := transformAOQTResidualLowRankCacheFile(cfg.Parameters, cfg.QueryVectorPath, cfg.OutputQueryPath, aoqtResidualLowRankApplyQueries(cfg.Parameters.Role))
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, fmt.Errorf("transform residual low-rank query vectors: %w", err)
	}
	cleanup = append(cleanup, cfg.OutputQueryPath)
	paramSHA, err := cfg.Parameters.SHA256()
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, err
	}
	inputDocSHA, err := sha256FileHexAOQTVectorCache(cfg.DocVectorPath)
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, err
	}
	inputQuerySHA, err := sha256FileHexAOQTVectorCache(cfg.QueryVectorPath)
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, err
	}
	outputDocSHA, err := sha256FileHexAOQTVectorCache(cfg.OutputDocPath)
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, err
	}
	outputQuerySHA, err := sha256FileHexAOQTVectorCache(cfg.OutputQueryPath)
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, err
	}
	binding := AOQTResidualLowRankCacheBinding{
		Schema:          AOQTResidualLowRankTransformSchema,
		Version:         AOQTResidualLowRankTransformVersion,
		Dataset:         cfg.Dataset,
		Artifact:        cfg.Artifact,
		ParameterSHA256: paramSHA,
		Parameters: AOQTResidualLowRankParameterBinding{
			Dim: cfg.Parameters.Dim, Rank: cfg.Parameters.Rank, Role: cfg.Parameters.Role, Alpha: cfg.Parameters.Alpha, Seed: cfg.Parameters.Seed,
		},
		Inputs: AOQTVectorCacheTransformIO{
			DocVectorPath: cfg.DocVectorPath, DocVectorSHA256: inputDocSHA, DocVectorCount: docAudit.Count,
			QueryVectorPath: cfg.QueryVectorPath, QueryVectorSHA256: inputQuerySHA, QueryVectorCount: queryAudit.Count,
			VectorFieldPolicy: "read_first_present_vector_embedding_values", IDsAndRolesPolicy: "preserve_all_non_vector_fields_including_id__id_parent_child_role",
		},
		Outputs: AOQTVectorCacheTransformIO{
			DocVectorPath: cfg.OutputDocPath, DocVectorSHA256: outputDocSHA, DocVectorCount: docCount,
			QueryVectorPath: cfg.OutputQueryPath, QueryVectorSHA256: outputQuerySHA, QueryVectorCount: queryCount,
			VectorFieldPolicy: "replace_first_present_vector_embedding_values", IDsAndRolesPolicy: "preserve_all_non_vector_fields_including_id__id_parent_child_role",
		},
	}
	data, err := json.MarshalIndent(binding, "", "  ")
	if err != nil {
		return AOQTResidualLowRankCacheBinding{}, err
	}
	data = append(data, '\n')
	if err := writeExclusiveAOQTVectorCacheFile(cfg.SidecarPath, data, 0o644); err != nil {
		return AOQTResidualLowRankCacheBinding{}, fmt.Errorf("write AOQT residual low-rank sidecar: %w", err)
	}
	cleanup = append(cleanup, cfg.SidecarPath)
	cleanupOnFailure = false
	return binding, nil
}

func transformAOQTResidualLowRankCacheFile(params AOQTResidualLowRankParameters, inputPath, outputPath string, apply bool) (int, error) {
	in, err := os.Open(inputPath)
	if err != nil {
		return 0, err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return 0, err
	}
	out, err := os.OpenFile(outputPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return 0, fmt.Errorf("output %q already exists", outputPath)
		}
		return 0, err
	}
	removeOnFailure := true
	defer func() {
		_ = out.Close()
		if removeOnFailure {
			_ = os.Remove(outputPath)
		}
	}()
	writer := bufio.NewWriter(out)
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 64*1024), 64*1024*1024)
	count := 0
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		transformed, err := transformAOQTResidualLowRankCacheRecord(params, line, apply)
		if err != nil {
			return 0, fmt.Errorf("%s:%d: %w", inputPath, lineNo, err)
		}
		if _, err := writer.Write(transformed); err != nil {
			return 0, err
		}
		if err := writer.WriteByte('\n'); err != nil {
			return 0, err
		}
		count++
	}
	if err := scanner.Err(); err != nil {
		return 0, err
	}
	if count == 0 {
		return 0, fmt.Errorf("%s: no vector records found", inputPath)
	}
	if err := writer.Flush(); err != nil {
		return 0, err
	}
	if err := out.Sync(); err != nil {
		return 0, err
	}
	if err := out.Close(); err != nil {
		return 0, err
	}
	removeOnFailure = false
	return count, nil
}

func transformAOQTResidualLowRankCacheRecord(params AOQTResidualLowRankParameters, line []byte, apply bool) ([]byte, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(line, &object); err != nil {
		return nil, err
	}
	field, raw, err := firstAOQTVectorField(object)
	if err != nil {
		return nil, err
	}
	var vector []float32
	if err := json.Unmarshal(raw, &vector); err != nil {
		return nil, fmt.Errorf("%s must be a numeric vector: %w", field, err)
	}
	if err := validateAOQTFiniteVector(vector, field); err != nil {
		return nil, err
	}
	if err := validateAOQTResidualLowRankVector(vector, field); err != nil {
		return nil, err
	}
	out := append([]float32(nil), vector...)
	if apply {
		out, err = params.Apply(vector)
		if err != nil {
			return nil, err
		}
	}
	data, err := json.Marshal(out)
	if err != nil {
		return nil, err
	}
	object[field] = data
	return json.Marshal(object)
}

type aoqtResidualLowRankCacheAudit struct {
	Count  int
	SHA256 string
}

func validateAOQTResidualLowRankCacheFile(path string) (aoqtResidualLowRankCacheAudit, error) {
	file, err := os.Open(path)
	if err != nil {
		return aoqtResidualLowRankCacheAudit{}, err
	}
	defer file.Close()
	seen := map[string]struct{}{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 64*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		id, vector, err := readAOQTResidualLowRankCacheRecordIdentityAndVector(line)
		if err != nil {
			return aoqtResidualLowRankCacheAudit{}, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		if _, ok := seen[id]; ok {
			return aoqtResidualLowRankCacheAudit{}, fmt.Errorf("%s:%d: duplicate vector id %q", path, lineNo, id)
		}
		seen[id] = struct{}{}
		if err := validateAOQTResidualLowRankVector(vector, "vector id "+id); err != nil {
			return aoqtResidualLowRankCacheAudit{}, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
	}
	if err := scanner.Err(); err != nil {
		return aoqtResidualLowRankCacheAudit{}, err
	}
	if len(seen) == 0 {
		return aoqtResidualLowRankCacheAudit{}, fmt.Errorf("%s: no vector records found", path)
	}
	sha, err := sha256FileHexAOQTVectorCache(path)
	if err != nil {
		return aoqtResidualLowRankCacheAudit{}, err
	}
	return aoqtResidualLowRankCacheAudit{Count: len(seen), SHA256: sha}, nil
}

func readAOQTResidualLowRankCacheRecordIdentityAndVector(line []byte) (string, []float32, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(line, &object); err != nil {
		return "", nil, err
	}
	id, err := aoqtResidualLowRankRecordID(object)
	if err != nil {
		return "", nil, err
	}
	field, raw, err := firstAOQTVectorField(object)
	if err != nil {
		return "", nil, err
	}
	var vector []float32
	if err := json.Unmarshal(raw, &vector); err != nil {
		return "", nil, fmt.Errorf("%s must be a numeric vector: %w", field, err)
	}
	return id, vector, nil
}

func aoqtResidualLowRankRecordID(object map[string]json.RawMessage) (string, error) {
	for _, field := range []string{"_id", "id"} {
		raw, ok := object[field]
		if !ok || len(raw) == 0 || string(raw) == "null" {
			continue
		}
		var id string
		if err := json.Unmarshal(raw, &id); err != nil {
			return "", fmt.Errorf("%s must be a string id: %w", field, err)
		}
		id = strings.TrimSpace(id)
		if id == "" {
			return "", fmt.Errorf("%s must be non-empty", field)
		}
		return id, nil
	}
	return "", fmt.Errorf("record missing _id or id field")
}

func validateAOQTResidualLowRankVector(vector []float32, label string) error {
	if len(vector) != AOQTResidualLowRankDim {
		return fmt.Errorf("AOQT residual low-rank %s dim = %d, want %d", label, len(vector), AOQTResidualLowRankDim)
	}
	if err := validateAOQTFiniteVector(vector, "AOQT residual low-rank "+label); err != nil {
		return err
	}
	norm := aoqtResidualNorm(vector)
	if math.Abs(norm-1) > 1e-3 {
		return fmt.Errorf("AOQT residual low-rank %s must be unit-normalized, norm=%g", label, norm)
	}
	return nil
}

func validateAOQTResidualLowRankHeader(p AOQTResidualLowRankParameters) error {
	if p.Schema != AOQTResidualLowRankTransformSchema || p.Version != AOQTResidualLowRankTransformVersion {
		return fmt.Errorf("AOQT residual low-rank schema/version is unsupported")
	}
	if p.Dim != AOQTResidualLowRankDim {
		return fmt.Errorf("AOQT residual low-rank dim = %d, want %d", p.Dim, AOQTResidualLowRankDim)
	}
	if !containsIntAOQTResidual(p.Rank, AOQTResidualLowRankRanks()) {
		return fmt.Errorf("AOQT residual low-rank rank %d is not canonical", p.Rank)
	}
	if !containsStringAOQTResidual(p.Role, AOQTResidualLowRankRoles()) {
		return fmt.Errorf("AOQT residual low-rank role %q is not canonical", p.Role)
	}
	if !containsFloat64AOQTResidual(p.Alpha, AOQTResidualLowRankMagnitudes()) {
		return fmt.Errorf("AOQT residual low-rank alpha %g is not canonical", p.Alpha)
	}
	if strings.TrimSpace(p.Seed) == "" {
		return fmt.Errorf("AOQT residual low-rank seed is required")
	}
	return nil
}

func aoqtResidualLowRankApplyDocs(role string) bool {
	return role == AOQTResidualLowRankRoleDocumentOnly || role == AOQTResidualLowRankRoleSymmetric
}

func aoqtResidualLowRankApplyQueries(role string) bool {
	return role == AOQTResidualLowRankRoleQueryOnly || role == AOQTResidualLowRankRoleSymmetric
}

func aoqtResidualDeterministicValue(seed, matrix string, i, j int) float32 {
	h := sha256.New()
	h.Write([]byte(seed))
	h.Write([]byte{0})
	h.Write([]byte(matrix))
	var buf [16]byte
	binary.LittleEndian.PutUint64(buf[:8], uint64(i))
	binary.LittleEndian.PutUint64(buf[8:], uint64(j))
	h.Write(buf[:])
	sum := h.Sum(nil)
	u := binary.LittleEndian.Uint64(sum[:8]) >> 11
	value := (float64(u)/float64(uint64(1)<<53))*2 - 1
	return float32(value)
}

func aoqtResidualNormalizeInPlace(vector []float32) {
	norm := aoqtResidualNorm(vector)
	if norm == 0 || !isFinite64(norm) {
		return
	}
	for i := range vector {
		vector[i] = float32(float64(vector[i]) / norm)
	}
}

func aoqtResidualNorm(vector []float32) float64 {
	sum := 0.0
	for _, value := range vector {
		sum += float64(value) * float64(value)
	}
	return math.Sqrt(sum)
}

func containsIntAOQTResidual(value int, values []int) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func containsStringAOQTResidual(value string, values []string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func containsFloat64AOQTResidual(value float64, values []float64) bool {
	for _, candidate := range values {
		if value == candidate {
			return true
		}
	}
	return false
}

func sha256HexAOQTResidual(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
