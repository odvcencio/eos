package eosruntime

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAOQTV10bGradientSVDParametersAreDeterministicSourceDerivedAndUnitNorm(t *testing.T) {
	source := aoqtV10bTestSource()
	a, err := DeriveAOQTV10bGradientSVDParameters(source, 4, AOQTResidualLowRankRoleSymmetric, 0.0025)
	if err != nil {
		t.Fatalf("derive params: %v", err)
	}
	b, err := DeriveAOQTV10bGradientSVDParameters(source, 4, AOQTResidualLowRankRoleSymmetric, 0.0025)
	if err != nil {
		t.Fatalf("derive params b: %v", err)
	}
	shaA, err := a.SHA256()
	if err != nil {
		t.Fatalf("sha a: %v", err)
	}
	shaB, err := b.SHA256()
	if err != nil {
		t.Fatalf("sha b: %v", err)
	}
	if shaA != shaB {
		t.Fatalf("V10b derivation is not deterministic: %s != %s", shaA, shaB)
	}
	if a.Schema != AOQTV10bGradientSVDParameterSchema || a.SourceSHA256 == "" || a.ProjectedOperatorSHA256 == "" {
		t.Fatalf("parameter provenance incomplete: %+v", a)
	}
	if len(a.A) != 4 || len(a.A[0]) != AOQTV10bGradientSVDDim || len(a.B) != AOQTV10bGradientSVDDim || len(a.B[0]) != 4 {
		t.Fatalf("unexpected parameter shape")
	}
	vector := make([]float32, AOQTV10bGradientSVDDim)
	vector[1] = float32(1 / math.Sqrt2)
	vector[2] = float32(1 / math.Sqrt2)
	out, err := a.Apply(vector)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if len(out) != AOQTV10bGradientSVDDim || math.Abs(aoqtResidualNorm(out)-1) > 2e-5 {
		t.Fatalf("output dim/norm invalid: len=%d norm=%g", len(out), aoqtResidualNorm(out))
	}
	if math.Abs(float64(out[1]-vector[1])) < 1e-8 && math.Abs(float64(out[2]-vector[2])) < 1e-8 {
		t.Fatalf("expected V10b residual movement")
	}
}

func TestAOQTV10bGradientSVDRejectsNonfiniteDimensionAndRankTamper(t *testing.T) {
	source := aoqtV10bTestSource()
	source.Q3Gradient[0] = math.NaN()
	_, err := DeriveAOQTV10bGradientSVDParameters(source, 4, AOQTResidualLowRankRoleSymmetric, 0.0025)
	if err == nil || !strings.Contains(err.Error(), "finite") {
		t.Fatalf("nonfinite source error = %v, want finite rejection", err)
	}
	source = aoqtV10bTestSource()
	source.Q3TargetCovariance = source.Q3TargetCovariance[:AOQTV10bGradientSVDDim-1]
	_, err = DeriveAOQTV10bGradientSVDParameters(source, 4, AOQTResidualLowRankRoleSymmetric, 0.0025)
	if err == nil || !strings.Contains(err.Error(), "covariance rows") {
		t.Fatalf("dimension error = %v, want covariance dimension rejection", err)
	}
	source = aoqtV10bTestSource()
	_, err = DeriveAOQTV10bGradientSVDParameters(source, 7, AOQTResidualLowRankRoleSymmetric, 0.0025)
	if err == nil || !strings.Contains(err.Error(), "noncanonical") {
		t.Fatalf("rank error = %v, want noncanonical rejection", err)
	}
}

func TestAOQTV10bGradientSVDProjectsProtectedDirection(t *testing.T) {
	source := aoqtV10bTestSource()
	params, err := DeriveAOQTV10bGradientSVDParameters(source, 4, AOQTResidualLowRankRoleSymmetric, 0.0025)
	if err != nil {
		t.Fatalf("derive params: %v", err)
	}
	protected := source.ProtectedGradients[0].Grad
	for k, row := range params.A {
		dot := 0.0
		for i, value := range row {
			dot += float64(value) * protected[i]
		}
		if math.Abs(dot) > 1e-8 {
			t.Fatalf("A[%d] was not projected away from protected gradient: dot=%g", k, dot)
		}
	}
}

func TestAOQTV10bGradientSVDReceiptReplayAndTamperRejection(t *testing.T) {
	source := aoqtV10bTestSource()
	sourceSHA, err := aoqtCanonicalSHA256(source)
	if err != nil {
		t.Fatalf("source sha: %v", err)
	}
	binding := AOQTV10bGradientSVDBinding{
		SourceFileSHA256:               sourceSHA,
		SourceSHA256:                   sourceSHA,
		FoldID:                         source.FoldID,
		TrainRowsSHA256:                source.TrainRowsSHA256,
		SplitManifestSHA256:            source.SplitManifestSHA256,
		CanonicalV7R6ReceiptFileSHA256: AOQTV10bCanonicalV7R6ReceiptFileSHA256,
		CanonicalV8ReceiptFileSHA256:   AOQTV10bCanonicalV8ReceiptFileSHA256,
		CanonicalV9ReceiptFileSHA256:   AOQTV10bCanonicalV9ReceiptFileSHA256,
	}
	receipt, err := BuildAOQTV10bGradientSVDReceipt(binding, source)
	if err != nil {
		t.Fatalf("build receipt: %v", err)
	}
	if err := receipt.ValidateWithSource(source); err != nil {
		t.Fatalf("validate with source: %v", err)
	}
	if receipt.Passed || receipt.FullVerificationCompleted || len(receipt.Endpoints) != AOQTV10bGradientSVDCandidateCap {
		t.Fatalf("fail-closed accounting invalid: passed=%v full=%v endpoints=%d", receipt.Passed, receipt.FullVerificationCompleted, len(receipt.Endpoints))
	}
	tampered := receipt
	tampered.Endpoints = append([]AOQTV10bGradientSVDEndpoint(nil), receipt.Endpoints...)
	tampered.Endpoints[0].ParameterSHA256 = strings.Repeat("a", 64)
	tampered.ParameterScheduleSHA256 = aoqtV10bTestScheduleSHA(t, tampered.Endpoints)
	tampered.EndpointHashChain, tampered.EndpointHashChainTailSHA256 = aoqtV10bTestChain(t, tampered.Endpoints)
	if err := tampered.ValidateWithSource(source); err == nil || !strings.Contains(err.Error(), "parameter replay hash mismatch") {
		t.Fatalf("parameter tamper error = %v, want replay mismatch", err)
	}
	tamperedSource := source
	tamperedSource.Q3Gradient = append([]float64(nil), source.Q3Gradient...)
	tamperedSource.Q3Gradient[3] += 0.25
	if err := receipt.ValidateWithSource(tamperedSource); err == nil || !strings.Contains(err.Error(), "source replay mismatch") {
		t.Fatalf("source tamper error = %v, want source replay mismatch", err)
	}
}

func TestRunAOQTV10bGradientSVDScreenRequiresCanonicalTerminalReceipts(t *testing.T) {
	dir := t.TempDir()
	source := aoqtV10bTestSource()
	sourcePath := filepath.Join(dir, "source.json")
	aoqtV10bWriteJSON(t, sourcePath, source)
	v7 := filepath.Join(dir, "v7.json")
	v8 := filepath.Join(dir, "v8.json")
	v9 := filepath.Join(dir, "v9.json")
	if err := os.WriteFile(v7, []byte("v7\n"), 0o644); err != nil {
		t.Fatalf("write v7: %v", err)
	}
	if err := os.WriteFile(v8, []byte("v8\n"), 0o644); err != nil {
		t.Fatalf("write v8: %v", err)
	}
	if err := os.WriteFile(v9, []byte("v9\n"), 0o644); err != nil {
		t.Fatalf("write v9: %v", err)
	}
	receiptPath := filepath.Join(dir, "receipt.json")
	_, err := RunAOQTV10bGradientSVDScreen(AOQTV10bGradientSVDRunConfig{
		SourcePath: sourcePath, OutputReceiptJSONPath: receiptPath,
		CanonicalV7R6ReceiptPath: v7, CanonicalV7R6ReceiptFileSHA256: shaFileForTest(t, v7),
		CanonicalV8ReceiptPath: v8, CanonicalV8ReceiptFileSHA256: shaFileForTest(t, v8),
		CanonicalV9ReceiptPath: v9, CanonicalV9ReceiptFileSHA256: shaFileForTest(t, v9),
	})
	if err == nil || !strings.Contains(err.Error(), "exact terminal V7-r6/V8/V9") {
		t.Fatalf("placeholder terminal receipt error = %v, want canonical hash rejection", err)
	}
	if _, statErr := os.Stat(receiptPath); !os.IsNotExist(statErr) {
		t.Fatalf("receipt output should not exist after canonical guard failure: %v", statErr)
	}
	if _, err := RunAOQTV10bGradientSVDScreen(AOQTV10bGradientSVDRunConfig{
		SourcePath:               sourcePath,
		CanonicalV7R6ReceiptPath: v7, CanonicalV7R6ReceiptFileSHA256: strings.Repeat("0", 64),
		CanonicalV8ReceiptPath: v8, CanonicalV8ReceiptFileSHA256: shaFileForTest(t, v8),
		CanonicalV9ReceiptPath: v9, CanonicalV9ReceiptFileSHA256: shaFileForTest(t, v9),
	}); err == nil || !strings.Contains(err.Error(), "hash mismatch") {
		t.Fatalf("receipt hash tamper error = %v, want hash mismatch", err)
	}
}

func TestBuildAOQTV10bGradientSVDSourceFromCalibrationSet(t *testing.T) {
	set, objective := aoqtV10bTinyPreparedIPSet(t)
	source, err := BuildAOQTV10bGradientSVDSourceFromCalibrationSet(set, objective, "fold-0", strings.Repeat("a", 64), strings.Repeat("b", 64))
	if err != nil {
		t.Fatalf("build source: %v", err)
	}
	if err := source.Validate(); err != nil {
		t.Fatalf("validate source: %v", err)
	}
	if source.FoldID != "fold-0" || len(source.Q3Gradient) != AOQTV10bGradientSVDDim || len(source.Q3TargetCovariance) != AOQTV10bGradientSVDDim {
		t.Fatalf("source shape/provenance invalid: fold=%q q3=%d cov=%d", source.FoldID, len(source.Q3Gradient), len(source.Q3TargetCovariance))
	}
	if len(source.ProtectedGradients) != 1 || source.ProtectedGradients[0].Name != "q3_order_guard" {
		t.Fatalf("protected gradients = %+v, want q3_order_guard only", source.ProtectedGradients)
	}
	badObjective := aoqtV10bWrappingObjective{inner: objective}
	if _, err := BuildAOQTV10bGradientSVDSourceFromCalibrationSet(set, badObjective, "fold-0", strings.Repeat("a", 64), strings.Repeat("b", 64)); err == nil || !strings.Contains(err.Error(), "concrete prepared-IP") {
		t.Fatalf("wrapped objective error = %v, want concrete prepared-IP rejection", err)
	}
}

func aoqtV10bTestSource() AOQTV10bGradientSVDSource {
	q3 := make([]float64, AOQTV10bGradientSVDDim)
	q3[1], q3[5], q3[8] = 1.0, -0.5, 0.25
	protected := make([]float64, AOQTV10bGradientSVDDim)
	protected[0] = 1
	cov := make([][]float64, AOQTV10bGradientSVDDim)
	for i := range cov {
		cov[i] = make([]float64, AOQTV10bGradientSVDDim)
	}
	for i := 1; i <= 12; i++ {
		cov[i][i] = float64(40 - i)
	}
	cov[5][8], cov[8][5] = 0.125, 0.125
	return AOQTV10bGradientSVDSource{
		Schema: AOQTV10bGradientSVDSchema, Version: AOQTV10bGradientSVDVersion, Dim: AOQTV10bGradientSVDDim,
		FoldID: "fold-0", TrainRowsSHA256: strings.Repeat("1", 64), SplitManifestSHA256: strings.Repeat("2", 64), OperatorPolicy: AOQTV10bGradientSVDOperatorPolicy,
		Q3Gradient: q3, ProtectedGradients: []AOQTV10bNamedGradient{{Name: "q3_order_guard", Grad: protected}}, Q3TargetCovariance: cov,
	}
}

type aoqtV10bWrappingObjective struct {
	inner AOQTSidecarPreparedIPObjective
}

func (o aoqtV10bWrappingObjective) EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
	return o.inner.EvaluateAOQT(input)
}

func (o aoqtV10bWrappingObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return o.inner.AOQTPreparedIPObjectiveConfig()
}

func aoqtV10bTinyPreparedIPSet(t *testing.T) (AOQTSidecarCalibrationSet, AOQTSidecarPreparedIPObjective) {
	t.Helper()
	cfg := normalizedAOQTPreparedIPObjectiveConfig(AOQTSidecarPreparedIPObjectiveConfig{TurboQuantSeed: AOQTResidualLowRankTurboQuantSeed})
	objective, err := NewAOQTSidecarPreparedIPObjective(cfg)
	if err != nil {
		t.Fatalf("new prepared objective: %v", err)
	}
	weights := AOQTSidecarRowWeights{Q3Gain: 1, Q3OrderGuard: 1}
	contract := cfg.ObjectiveContract(weights)
	split := AOQTSidecarTrainOnlySplitProof{Split: "train", TrainOnly: true, ProofSHA256: strings.Repeat("c", 64), ExclusionIdentities: append([]string(nil), requiredAOQTSplitExclusionIdentities...)}
	query := make([]float32, AOQTSidecarDim)
	c0 := make([]float32, AOQTSidecarDim)
	c1 := make([]float32, AOQTSidecarDim)
	query[0], c0[0], c1[1] = 1, 1, 1
	candidates := [][]float32{c0, c1}
	dense := []float32{dotAOQT(query, c0), dotAOQT(query, c1)}
	q3 := newAOQTPreparedIPSurface(query, candidates, AOQTSidecarDim, contract.Q3GuardBit, cfg.TurboQuantSeed).scores
	q5 := newAOQTPreparedIPSurface(query, candidates, AOQTSidecarDim, contract.Q5GuardBit, cfg.TurboQuantSeed).scores
	sourceHash := strings.Repeat("d", 64)
	qrelsSHA := strings.Repeat("e", 64)
	compat := strings.Repeat("f", 64)
	docIDs := []string{"d0", "d1"}
	row := AOQTSidecarCalibrationRow{
		Schema: AOQTSidecarRowSchema, RowID: "row-001", Dataset: "toy", QueryID: "q1", QueryVectorID: "qv1", QueryVectorSHA256: aoqtVectorSHA256(query), QueryVector: query,
		CandidateDocIDs: docIDs, CandidateVectorIDs: []string{"cv0", "cv1"}, CandidateVectorSHA256: []string{aoqtVectorSHA256(c0), aoqtVectorSHA256(c1)}, CandidateVectors: candidates,
		QrelGains: []float32{2, 0}, CandidateSources: []string{"toy", "toy"}, EligiblePairMask: [][]bool{{false, true}, {false, false}}, GuardClass: "unit-test",
		AnchorScores: AOQTSidecarAnchorScores{Dense: dense, Q3: q3, Q5: q5}, AnchorRanks: AOQTSidecarAnchorRanks{Dense: ranksAOQT(docIDs, dense), Q3: ranksAOQT(docIDs, q3), Q5: ranksAOQT(docIDs, q5)},
		Weights: weights, SourceArtifactHash: sourceHash, QrelsSHA256: qrelsSHA, SplitProof: split, CompatibilityDigest: compat, LegalGates: AOQTSidecarLegalGates{ResearchTrainAllowed: true},
	}
	manifest := AOQTSidecarCalibrationManifest{
		Schema: AOQTSidecarManifestSchema, AnchorArtifactSHA256: strings.Repeat("1", 64), AnchorPackageManifestSHA256: strings.Repeat("2", 64), AnchorEmbeddingSpaceID: "eos-d384-test",
		Dim: AOQTSidecarDim, Topology: AOQTSidecarTopologyBinding{Kind: AOQTTopologyKindGivensV1, Dim: AOQTSidecarDim, Stages: AOQTSidecarStages, PairsPerStage: AOQTSidecarPairsPerStage, AngleCount: AOQTSidecarAngleCount, Seed: 11, PairingsSHA256: strings.Repeat("3", 64)},
		TurboQuantSeed: cfg.TurboQuantSeed, QuantSurfaces: []AOQTSidecarQuantSurface{{BitWidth: 3, Seed: cfg.TurboQuantSeed, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true}, {BitWidth: 5, Seed: cfg.TurboQuantSeed, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true}},
		ObjectiveContract: contract, QrelsSHA256ByDataset: map[string]string{"toy": qrelsSHA}, SplitProof: split, SourceArtifactHashes: []string{sourceHash}, VectorCacheHashes: []string{strings.Repeat("4", 64)},
		RowCount: 1, RowIDSHA256: aoqtRowIDSHA256([]string{row.RowID}), CompatibilityDigest: compat, LegalGates: AOQTSidecarLegalGates{ResearchTrainAllowed: true},
	}
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: []AOQTSidecarCalibrationRow{row}}
	if err := set.Validate(); err != nil {
		t.Fatalf("tiny set validate: %v", err)
	}
	return set, objective
}

func aoqtV10bWriteJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatalf("marshal json: %v", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write json: %v", err)
	}
}

func aoqtV10bTestScheduleSHA(t *testing.T, endpoints []AOQTV10bGradientSVDEndpoint) string {
	t.Helper()
	sha, err := aoqtV10bScheduleSHA256(endpoints)
	if err != nil {
		t.Fatalf("schedule sha: %v", err)
	}
	return sha
}

func aoqtV10bTestChain(t *testing.T, endpoints []AOQTV10bGradientSVDEndpoint) (string, string) {
	t.Helper()
	chain, tail, err := aoqtV10bEndpointHashChain(endpoints)
	if err != nil {
		t.Fatalf("chain: %v", err)
	}
	return chain, tail
}
