package main

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

func runTrainAOQTSidecar(args []string) error {
	fs := flag.NewFlagSet("train-aoqt-sidecar", flag.ContinueOnError)
	var cfg eosruntime.AOQTSidecarTrainRunnerConfig
	var sourceHashes string
	var sourceHashesFile string
	var sourceHashesFileSHA256 string
	var vectorHashes string
	var learningRate float64
	fs.BoolVar(&cfg.PlanOnly, "plan-only", false, "validate AOQT training inputs and write metrics without optimizer or package output")
	fs.BoolVar(&cfg.AllowResearchOnly, "allow-research-only-aoqt", false, "explicitly allow research-only AOQT train rows; release/commercial/free-open gates must remain false")
	fs.StringVar(&cfg.ManifestPath, "manifest", "", "strict AOQT calibration manifest JSON")
	fs.StringVar(&cfg.RowsJSONLPath, "rows", "", "strict AOQT calibration rows JSONL")
	fs.StringVar(&cfg.PreflightJSONPath, "preflight", "", "strict AOQT materializer preflight JSON")
	fs.StringVar(&cfg.MetricsJSONPath, "metrics-json", "", "write AOQT sidecar training metrics JSON")
	fs.StringVar(&cfg.OutputArtifactPath, "output", "", "write guarded AOQT candidate package artifact for non-plan training")
	fs.StringVar(&cfg.ExpectedManifestSHA256, "expected-manifest-sha256", "", "expected calibration manifest sha256")
	fs.StringVar(&cfg.ExpectedRowsSHA256, "expected-rows-sha256", "", "expected calibration rows JSONL sha256")
	fs.StringVar(&cfg.ExpectedPreflightSHA256, "expected-preflight-sha256", "", "expected materializer preflight sha256")
	fs.StringVar(&cfg.ExpectedAnchorArtifactSHA256, "expected-anchor-artifact-sha256", "", "expected immutable anchor artifact sha256")
	fs.StringVar(&cfg.ExpectedAnchorPackageManifestSHA256, "expected-anchor-package-manifest-sha256", "", "expected immutable anchor package manifest sha256")
	fs.StringVar(&cfg.ExpectedAnchorEmbeddingSpaceID, "anchor-embedding-space-id", "", "expected anchor embedding space id")
	fs.StringVar(&cfg.ExpectedCompatibilityDigest, "compatibility-digest", "", "expected AOQT compatibility digest")
	fs.StringVar(&sourceHashes, "source-artifact-sha256", "", "comma-separated expected source artifact sha256 values; for large lists use --source-artifact-sha256-file")
	fs.StringVar(&sourceHashesFile, "source-artifact-sha256-file", "", "newline-delimited expected source artifact sha256 values")
	fs.StringVar(&sourceHashesFileSHA256, "source-artifact-sha256-file-sha256", "", "expected sha256 of --source-artifact-sha256-file")
	fs.StringVar(&vectorHashes, "vector-cache-sha256", "", "comma-separated expected vector cache sha256 values")
	fs.IntVar(&cfg.MaxSteps, "max-steps", 0, "positive optimizer steps for non-plan training")
	fs.Float64Var(&learningRate, "lr", 0, "AOQT optimizer learning rate")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos train-aoqt-sidecar [flags]")
	}
	var err error
	cfg.ExpectedSourceArtifactHashes, err = parseRequiredHashListBinding(sourceHashes, sourceHashesFile, sourceHashesFileSHA256, "source-artifact-sha256")
	if err != nil {
		return err
	}
	cfg.ExpectedVectorCacheHashes, err = parseRequiredHashList(vectorHashes, "vector-cache-sha256")
	if err != nil {
		return err
	}
	cfg.LearningRate = float32(learningRate)
	result, err := eosruntime.RunAOQTSidecarTraining(cfg)
	if err != nil {
		return err
	}
	fmt.Printf("AOQT sidecar metrics: %s\n", cfg.MetricsJSONPath)
	fmt.Printf("plan: rows=%d candidates=%d pairs=%d steps=%d plan_only=%t\n",
		result.IOReport.RowCount,
		result.IOReport.CandidateCount,
		result.IOReport.PairCount,
		result.Metrics.Plan.StepCount,
		result.Metrics.Plan.PlanOnly,
	)
	fmt.Printf("seeds: turboquant=%d topology=%d\n", result.IOReport.TurboQuantSeed, result.IOReport.Topology.Seed)
	if result.PackageResult != nil {
		fmt.Printf("candidate package: %s\n", result.PackageResult.Paths.ArtifactPath)
	}
	return nil
}

func parseRequiredHashListBinding(raw, filePath, fileSHA256, name string) ([]string, error) {
	if filePath != "" {
		if strings.TrimSpace(filePath) == "" {
			return nil, fmt.Errorf("--%s-file path is empty", name)
		}
		if strings.TrimSpace(raw) != "" {
			return nil, fmt.Errorf("--%s and --%s-file are mutually exclusive", name, name)
		}
		return parseRequiredHashListFile(filePath, fileSHA256, name+"-file")
	}
	if strings.TrimSpace(fileSHA256) != "" {
		return nil, fmt.Errorf("--%s-file-sha256 requires --%s-file", name, name)
	}
	return parseRequiredHashList(raw, name)
}

func parseRequiredHashList(raw, name string) ([]string, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("AOQT sidecar training requires --%s", name)
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	seen := map[string]bool{}
	for _, part := range parts {
		value := strings.TrimSpace(part)
		if value == "" {
			return nil, fmt.Errorf("--%s contains an empty sha256", name)
		}
		if err := validateCLIHashListSHA256(value, "--"+name); err != nil {
			return nil, err
		}
		if seen[value] {
			return nil, fmt.Errorf("--%s contains duplicate sha256 %q", name, value)
		}
		seen[value] = true
		out = append(out, value)
	}
	return out, nil
}

func parseRequiredHashListFile(path, expectedSHA256, name string) ([]string, error) {
	if strings.TrimSpace(expectedSHA256) == "" {
		return nil, fmt.Errorf("--%s-sha256 is required when --%s is used", name, name)
	}
	if err := validateCLIHashListSHA256(expectedSHA256, "--"+name+"-sha256"); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read --%s %q: %w", name, path, err)
	}
	sum := sha256.Sum256(data)
	gotSHA256 := hex.EncodeToString(sum[:])
	if gotSHA256 != expectedSHA256 {
		return nil, fmt.Errorf("--%s sha256 = %s, want %s", name, gotSHA256, expectedSHA256)
	}
	lines := strings.Split(string(data), "\n")
	out := make([]string, 0, len(lines))
	seen := map[string]bool{}
	for i, line := range lines {
		if line == "" && i == len(lines)-1 {
			continue
		}
		if strings.TrimSpace(line) == "" {
			return nil, fmt.Errorf("--%s contains an empty sha256 at line %d", name, i+1)
		}
		if line != strings.TrimSpace(line) {
			return nil, fmt.Errorf("--%s line %d must contain only a sha256 value", name, i+1)
		}
		if err := validateCLIHashListSHA256(line, fmt.Sprintf("--%s line %d", name, i+1)); err != nil {
			return nil, err
		}
		if len(out) > 0 && line < out[len(out)-1] {
			return nil, fmt.Errorf("--%s must be sorted in ascending sha256 order; line %d is out of order", name, i+1)
		}
		if seen[line] {
			return nil, fmt.Errorf("--%s contains duplicate sha256 %q at line %d", name, line, i+1)
		}
		seen[line] = true
		out = append(out, line)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("--%s must contain at least one sha256", name)
	}
	return out, nil
}

func validateCLIHashListSHA256(value, label string) error {
	if len(value) != sha256.Size*2 {
		return fmt.Errorf("%s must be %d hex chars", label, sha256.Size*2)
	}
	if _, err := hex.DecodeString(value); err != nil {
		return fmt.Errorf("%s is invalid: %w", label, err)
	}
	return nil
}

func rejectAOQTCandidateExportMLL(artifactPath string) error {
	manifest, err := eosruntime.ReadPackageManifestFile(eosruntime.DefaultPackageManifestPath(artifactPath))
	if err != nil {
		return nil
	}
	if manifest.AOQTTransform.Enabled || manifest.HasFileRole(eosruntime.EmbeddingPostPoolTransformRole) {
		return fmt.Errorf("export-mll rejects AOQT candidate packages; keep the sidecar package layout so AOQT policy/provenance gates remain enforceable")
	}
	return nil
}
