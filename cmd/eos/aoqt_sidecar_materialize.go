package main

import (
	"flag"
	"fmt"
	"strconv"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

func runMaterializeAOQTSidecar(args []string) error {
	fs := flag.NewFlagSet("materialize-aoqt-sidecar", flag.ContinueOnError)
	planPath := fs.String("plan", "", "Stage3 AOQT calibration plan JSON")
	vectorPaths := stringListFlag{}
	scoreEvidencePaths := stringListFlag{}
	qrelsPaths := stringListFlag{}
	exclusionQIDPaths := commaStringListFlag{}
	fs.Var(&vectorPaths, "vectors", "AOQT vector JSONL input; repeat for query/doc files")
	fs.Var(&scoreEvidencePaths, "score-evidence", "AOQT q3/q5 top120 score evidence JSON; repeat per dataset/bit")
	fs.Var(&qrelsPaths, "qrels", "train qrels JSONL or TREC file; repeat per dataset")
	fs.Var(&exclusionQIDPaths, "exclusion-qids", "dataset-scoped AOQT qid-only exclusion manifest JSON; repeat or comma-separate exactly dev4,reserve4,official-test")
	rowsPath := fs.String("rows-jsonl", "", "output AOQT calibration rows JSONL")
	manifestPath := fs.String("manifest-json", "", "output AOQT calibration manifest JSON")
	preflightPath := fs.String("preflight-json", "", "output AOQT materializer preflight JSON")
	anchorEmbeddingSpaceID := fs.String("anchor-embedding-space-id", "", "expected frozen anchor embedding-space id")
	anchorArtifactPath := fs.String("anchor-artifact", "", "optional frozen anchor artifact path to record")
	anchorArtifactSHA256 := fs.String("anchor-artifact-sha256", "", "optional frozen anchor artifact sha256 override")
	packageManifestSHA256 := fs.String("anchor-package-manifest-sha256", "", "optional frozen anchor package manifest sha256 override")
	createdAtUTC := fs.String("created-at-utc", "", "optional deterministic created_at_utc")
	allowResearchOnly := fs.Bool("allow-research-only-aoqt", false, "acknowledge research-only AOQT materialization")
	turboQuantSeed := fs.Int64("turboquant-seed", eosruntime.AOQTSidecarMaterializerQuantSeed, "required AOQT TurboQuant prepared-IP seed")
	topologySeed := fs.Int64("topology-seed", eosruntime.AOQTSidecarMaterializerTopologySeed, "required AOQT topology seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos materialize-aoqt-sidecar [flags]")
	}
	if err := validateMaterializeAOQTExclusionQIDPaths(exclusionQIDPaths); err != nil {
		return err
	}
	preflight, err := eosruntime.MaterializeAOQTSidecarCalibration(eosruntime.AOQTSidecarMaterializeConfig{
		PlanPath:               *planPath,
		ExclusionQIDPaths:      exclusionQIDPaths,
		VectorPaths:            vectorPaths,
		ScoreEvidencePaths:     scoreEvidencePaths,
		QrelsPaths:             qrelsPaths,
		RowJSONLPath:           *rowsPath,
		ManifestJSONPath:       *manifestPath,
		PreflightJSONPath:      *preflightPath,
		AnchorEmbeddingSpaceID: *anchorEmbeddingSpaceID,
		CreatedAtUTC:           *createdAtUTC,
		AllowResearchOnly:      *allowResearchOnly,
		ExpectedTurboQuantSeed: *turboQuantSeed,
		ExpectedTopologySeed:   *topologySeed,
		AnchorArtifactPath:     *anchorArtifactPath,
		AnchorArtifactSHA256:   *anchorArtifactSHA256,
		PackageManifestSHA256:  *packageManifestSHA256,
	})
	if err != nil {
		return err
	}
	fmt.Printf("AOQT materialized rows=%d candidates=%d pairs=%d seed=%s\n", preflight.RowCount, preflight.CandidateCount, preflight.PairCount, strconv.FormatInt(preflight.TurboQuantSeed, 10))
	fmt.Printf("rows: %s\nmanifest: %s\npreflight: %s\n", *rowsPath, *manifestPath, *preflightPath)
	return nil
}

type stringListFlag []string

func (f *stringListFlag) String() string {
	return strings.Join(*f, ",")
}

func (f *stringListFlag) Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return fmt.Errorf("empty path")
	}
	*f = append(*f, value)
	return nil
}

type commaStringListFlag []string

func (f *commaStringListFlag) String() string {
	return strings.Join(*f, ",")
}

func (f *commaStringListFlag) Set(value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("empty path")
	}
	for _, part := range strings.Split(value, ",") {
		path := strings.TrimSpace(part)
		if path == "" {
			return fmt.Errorf("empty path")
		}
		*f = append(*f, path)
	}
	return nil
}

func validateMaterializeAOQTExclusionQIDPaths(paths []string) error {
	if len(paths) == 0 {
		return fmt.Errorf("AOQT materializer requires --exclusion-qids exactly 3 times or comma-separated with dev4,reserve4,official-test manifests")
	}
	if len(paths) != 3 {
		return fmt.Errorf("AOQT materializer requires exactly 3 --exclusion-qids manifest paths, got %d", len(paths))
	}
	seen := map[string]bool{}
	for _, path := range paths {
		if strings.TrimSpace(path) == "" {
			return fmt.Errorf("AOQT materializer --exclusion-qids contains an empty path")
		}
		if seen[path] {
			return fmt.Errorf("AOQT materializer duplicate --exclusion-qids path %q", path)
		}
		seen[path] = true
	}
	return nil
}
