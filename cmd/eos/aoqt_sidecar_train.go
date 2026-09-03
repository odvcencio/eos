package main

import (
	"flag"
	"fmt"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

func runTrainAOQTSidecar(args []string) error {
	fs := flag.NewFlagSet("train-aoqt-sidecar", flag.ContinueOnError)
	var cfg eosruntime.AOQTSidecarTrainRunnerConfig
	var sourceHashes string
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
	fs.StringVar(&sourceHashes, "source-artifact-sha256", "", "comma-separated expected source artifact sha256 values")
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
	cfg.ExpectedSourceArtifactHashes, err = parseRequiredHashList(sourceHashes, "source-artifact-sha256")
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

func parseRequiredHashList(raw, name string) ([]string, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("AOQT sidecar training requires --%s", name)
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		value := strings.TrimSpace(part)
		if value == "" {
			return nil, fmt.Errorf("--%s contains an empty sha256", name)
		}
		out = append(out, value)
	}
	return out, nil
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
