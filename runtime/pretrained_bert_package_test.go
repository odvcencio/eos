package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"strings"
	"testing"
)

func TestInferPretrainedBERTRetrievalRoleContractKnownModels(t *testing.T) {
	tests := []struct {
		modelName      string
		pooling        string
		maxLength      int
		queryPrefix    string
		documentPrefix string
	}{
		{
			modelName:      "BAAI/bge-small-en-v1.5",
			pooling:        "cls",
			maxLength:      512,
			queryPrefix:    "Represent this sentence for searching relevant passages: ",
			documentPrefix: "",
		},
		{
			modelName:      "intfloat/e5-small-v2",
			pooling:        "masked_mean",
			maxLength:      512,
			queryPrefix:    "query: ",
			documentPrefix: "passage: ",
		},
	}
	for _, tt := range tests {
		t.Run(tt.modelName, func(t *testing.T) {
			got := InferPretrainedBERTRetrievalRoleContract(tt.modelName, tt.pooling, tt.maxLength)
			if got == nil {
				t.Fatal("contract = nil")
			}
			if got.Schema != PretrainedBERTRetrievalRoleContractSchema {
				t.Fatalf("schema = %q", got.Schema)
			}
			if got.QueryRole != "query" || got.DocumentRole != "document" {
				t.Fatalf("roles = %q/%q", got.QueryRole, got.DocumentRole)
			}
			if got.QueryPrefix != tt.queryPrefix || got.DocumentPrefix != tt.documentPrefix {
				t.Fatalf("prefixes = %q/%q", got.QueryPrefix, got.DocumentPrefix)
			}
			if got.Pooling != tt.pooling || got.MaxLength != tt.maxLength {
				t.Fatalf("pooling/max_length = %q/%d", got.Pooling, got.MaxLength)
			}
		})
	}
	if got := InferPretrainedBERTRetrievalRoleContract("sentence-transformers/all-MiniLM-L6-v2", "masked_mean", 256); got != nil {
		t.Fatalf("unknown model contract = %+v, want nil", got)
	}
}

func TestPretrainedBERTPackageReadAllowsLegacyPackageWithoutRetrievalRoleContract(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	pkg, err := ReadPretrainedBERTPackageFile(packagePath)
	if err != nil {
		t.Fatalf("read package: %v", err)
	}
	if pkg.RetrievalRoleContract != nil {
		t.Fatalf("legacy package retrieval contract = %+v, want nil", pkg.RetrievalRoleContract)
	}
}

func TestPretrainedBERTPackageValidateRetrievalRoleContract(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	pkg, err := ReadPretrainedBERTPackageFile(packagePath)
	if err != nil {
		t.Fatalf("read package: %v", err)
	}

	tests := []struct {
		name      string
		contract  *PretrainedBERTRetrievalRoleContract
		wantError string
	}{
		{
			name: "bad schema",
			contract: &PretrainedBERTRetrievalRoleContract{
				Schema:         "wrong.schema",
				QueryRole:      "query",
				DocumentRole:   "document",
				QueryPrefix:    "query: ",
				DocumentPrefix: "passage: ",
			},
			wantError: "schema",
		},
		{
			name: "missing roles",
			contract: &PretrainedBERTRetrievalRoleContract{
				Schema:         PretrainedBERTRetrievalRoleContractSchema,
				QueryPrefix:    "query: ",
				DocumentPrefix: "passage: ",
			},
			wantError: "query_role",
		},
		{
			name: "missing prefixes",
			contract: &PretrainedBERTRetrievalRoleContract{
				Schema:       PretrainedBERTRetrievalRoleContractSchema,
				QueryRole:    "query",
				DocumentRole: "document",
			},
			wantError: "at least one role prefix",
		},
		{
			name: "pooling mismatch",
			contract: &PretrainedBERTRetrievalRoleContract{
				Schema:         PretrainedBERTRetrievalRoleContractSchema,
				QueryRole:      "query",
				DocumentRole:   "document",
				QueryPrefix:    "query: ",
				DocumentPrefix: "passage: ",
				Pooling:        "cls",
			},
			wantError: "pooling",
		},
		{
			name: "max_length mismatch",
			contract: &PretrainedBERTRetrievalRoleContract{
				Schema:         PretrainedBERTRetrievalRoleContractSchema,
				QueryRole:      "query",
				DocumentRole:   "document",
				QueryPrefix:    "query: ",
				DocumentPrefix: "passage: ",
				MaxLength:      pkg.MaxLength + 1,
			},
			wantError: "max_length",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidate := pkg
			candidate.RetrievalRoleContract = tt.contract
			if err := candidate.Validate(); err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Validate() err = %v, want %q", err, tt.wantError)
			}
		})
	}
}

func TestPretrainedBERTPackageIdentityIncludesRetrievalRoleContract(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	pkg, err := ReadPretrainedBERTPackageFile(packagePath)
	if err != nil {
		t.Fatalf("read package: %v", err)
	}
	pkg.RetrievalRoleContract = &PretrainedBERTRetrievalRoleContract{
		Schema:         PretrainedBERTRetrievalRoleContractSchema,
		QueryRole:      "query",
		DocumentRole:   "document",
		QueryPrefix:    "query: ",
		DocumentPrefix: "passage: ",
		Pooling:        pkg.Pooling,
		MaxLength:      pkg.MaxLength,
	}
	withContract := pkg.IdentityHash()
	pkg.RetrievalRoleContract = nil
	withoutContract := pkg.IdentityHash()
	if withContract == withoutContract {
		t.Fatalf("identity did not change when retrieval contract was removed: %q", withContract)
	}
}

func TestLoadPretrainedBERTPackageRuntimeArtifactsAttachesVerifiedProvenance(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	artifacts, err := LoadPretrainedBERTPackageRuntimeArtifacts(packagePath)
	if err != nil {
		t.Fatalf("load runtime artifacts: %v", err)
	}
	data, err := os.ReadFile(packagePath)
	if err != nil {
		t.Fatalf("read package: %v", err)
	}
	sum := sha256.Sum256(data)
	wantPackageSHA := hex.EncodeToString(sum[:])
	if artifacts.PackageSHA256 != wantPackageSHA {
		t.Fatalf("package sha = %s, want %s", artifacts.PackageSHA256, wantPackageSHA)
	}
	if artifacts.WeightSetGeneration == "" {
		t.Fatal("weight generation is empty")
	}
	meta := artifacts.Module.Metadata
	if meta[PretrainedBERTPackageMetadataVerified] != true {
		t.Fatalf("verified metadata = %v, want true", meta[PretrainedBERTPackageMetadataVerified])
	}
	for _, item := range []struct {
		key  string
		want string
	}{
		{PretrainedBERTPackageMetadataProvenanceVersion, PretrainedBERTPackageRuntimeProvenanceVersion},
		{PretrainedBERTPackageMetadataSHA256, artifacts.PackageSHA256},
		{PretrainedBERTPackageMetadataIdentitySHA256, artifacts.Package.IdentitySHA256},
		{PretrainedBERTPackageMetadataModuleSHA256, artifacts.Package.ModuleSHA256},
		{PretrainedBERTPackageMetadataWeightsSHA256, artifacts.Package.WeightsSHA256},
		{PretrainedBERTPackageMetadataWeightGeneration, artifacts.WeightSetGeneration},
	} {
		if got, _ := meta[item.key].(string); got != item.want {
			t.Fatalf("%s metadata = %q, want %q", item.key, got, item.want)
		}
	}
	if _, _, err := artifacts.Package.moduleWithRuntimeProvenance("not-a-sha"); err == nil || !strings.Contains(err.Error(), "package sha256") {
		t.Fatalf("moduleWithRuntimeProvenance malformed sha err = %v, want package sha256 rejection", err)
	}
}
