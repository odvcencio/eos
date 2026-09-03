package eosruntime

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAOQTSidecarTrainRunnerAcceptsCanonicalMaterializerManifestDigest(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	preflight, err := MaterializeAOQTSidecarCalibration(materializerCfg)
	if err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	rawManifestSHA := mustSHA256FileAOQTTest(t, materializerCfg.ManifestJSONPath)
	if rawManifestSHA == preflight.CalibrationManifestSHA256 {
		t.Fatalf("fixture raw manifest SHA unexpectedly equals canonical SHA %s", rawManifestSHA)
	}

	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	result, err := RunAOQTSidecarTraining(trainCfg)
	if err != nil {
		t.Fatalf("run AOQT sidecar train preflight: %v", err)
	}
	if result.IOReport.ManifestSHA256 != preflight.CalibrationManifestSHA256 {
		t.Fatalf("IO manifest SHA = %s, want preflight canonical SHA %s", result.IOReport.ManifestSHA256, preflight.CalibrationManifestSHA256)
	}
	if result.Metrics.Inputs.DatasetManifestSHA256 != preflight.CalibrationManifestSHA256 {
		t.Fatalf("metrics dataset manifest SHA = %s, want canonical SHA %s", result.Metrics.Inputs.DatasetManifestSHA256, preflight.CalibrationManifestSHA256)
	}
}

func TestAOQTSidecarTrainRunnerRejectsSemanticManifestTamper(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	manifest := readAOQTManifestForTrainRunnerTest(t, materializerCfg.ManifestJSONPath)
	manifest.CreatedAtUTC = "2030-01-01T00:00:00Z"
	if err := writeJSONFileAOQT(materializerCfg.ManifestJSONPath, manifest); err != nil {
		t.Fatal(err)
	}

	if _, err := RunAOQTSidecarTraining(trainCfg); err == nil || !strings.Contains(err.Error(), "AOQT calibration manifest sha256") {
		t.Fatalf("semantic manifest tamper error = %v, want manifest digest rejection", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsPreflightManifestDigestDrift(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	preflight, err := MaterializeAOQTSidecarCalibration(materializerCfg)
	if err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	rawManifestSHA := mustSHA256FileAOQTTest(t, materializerCfg.ManifestJSONPath)
	if rawManifestSHA == preflight.CalibrationManifestSHA256 {
		t.Fatalf("fixture raw manifest SHA unexpectedly equals canonical SHA %s", rawManifestSHA)
	}
	preflight.CalibrationManifestSHA256 = rawManifestSHA
	if err := writeJSONFileAOQT(materializerCfg.PreflightJSONPath, preflight); err != nil {
		t.Fatal(err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)

	if _, err := RunAOQTSidecarTraining(trainCfg); err == nil || !strings.Contains(err.Error(), "preflight calibration manifest sha256 mismatch") {
		t.Fatalf("preflight manifest digest drift error = %v, want preflight mismatch rejection", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsZeroAcceptedWithoutPackage(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OutputArtifactPath = filepath.Join(t.TempDir(), "candidate.mll")
	objectiveFactory := func(contract AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		return &statefulRejectingAOQTObjective{config: AOQTSidecarPreparedIPObjectiveConfig{
			Dim:              contract.Dim,
			TurboQuantSeed:   contract.TurboQuantSeed,
			GainBit:          contract.GainBit,
			Q3GuardBit:       contract.Q3GuardBit,
			Q5GuardBit:       contract.Q5GuardBit,
			GainCutoff:       contract.GainCutoff,
			GainTau:          contract.GainTau,
			GainMargin:       contract.GainMargin,
			GuardTau:         contract.GuardTau,
			GuardMargin:      contract.GuardMargin,
			ScoreDistillTau:  contract.ScoreDistillTau,
			NFBoundarySource: contract.NFBoundarySource,
		}}, nil
	}

	if _, err := runAOQTSidecarTraining(trainCfg, objectiveFactory); err == nil || !strings.Contains(err.Error(), "accepted zero safe steps") {
		t.Fatalf("run AOQT sidecar training error = %v, want zero-accepted failure", err)
	}
	for role, candidate := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(trainCfg.OutputArtifactPath, true)) {
		if _, err := os.Lstat(candidate); err == nil {
			t.Fatalf("runner wrote %s output %q despite zero accepted safe steps", role, candidate)
		} else if !os.IsNotExist(err) {
			t.Fatalf("stat %s output: %v", role, err)
		}
	}
}

func aoqtTrainRunnerConfigFromMaterializerFixture(t *testing.T, materializerCfg AOQTSidecarMaterializeConfig) AOQTSidecarTrainRunnerConfig {
	t.Helper()
	manifest := readAOQTManifestForTrainRunnerTest(t, materializerCfg.ManifestJSONPath)
	manifestSHA := mustAOQTManifestSHA256Test(t, manifest)
	return AOQTSidecarTrainRunnerConfig{
		ManifestPath:                        materializerCfg.ManifestJSONPath,
		RowsJSONLPath:                       materializerCfg.RowJSONLPath,
		PreflightJSONPath:                   materializerCfg.PreflightJSONPath,
		MetricsJSONPath:                     filepath.Join(t.TempDir(), "metrics.json"),
		ExpectedManifestSHA256:              manifestSHA,
		ExpectedRowsSHA256:                  mustSHA256FileAOQTTest(t, materializerCfg.RowJSONLPath),
		ExpectedPreflightSHA256:             mustSHA256FileAOQTTest(t, materializerCfg.PreflightJSONPath),
		ExpectedAnchorArtifactSHA256:        manifest.AnchorArtifactSHA256,
		ExpectedAnchorPackageManifestSHA256: manifest.AnchorPackageManifestSHA256,
		ExpectedAnchorEmbeddingSpaceID:      manifest.AnchorEmbeddingSpaceID,
		ExpectedCompatibilityDigest:         manifest.CompatibilityDigest,
		ExpectedSourceArtifactHashes:        append([]string(nil), manifest.SourceArtifactHashes...),
		ExpectedVectorCacheHashes:           append([]string(nil), manifest.VectorCacheHashes...),
		PlanOnly:                            true,
		AllowResearchOnly:                   true,
	}
}

func readAOQTManifestForTrainRunnerTest(t *testing.T, path string) AOQTSidecarCalibrationManifest {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(data, &manifest); err != nil {
		t.Fatal(err)
	}
	return manifest
}
