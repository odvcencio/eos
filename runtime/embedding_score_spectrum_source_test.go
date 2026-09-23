package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestEmbeddingTextScoreSpectrumSourceIndexesRowsAndMatchesReader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	rows := []string{
		`{"row_id":"r1","source":"source-a","query":"q1","candidate_doc_ids":["p","n1","n2"],"candidate_texts":["Positive text","negative text"," Negative   Text "],"positive_indexes":[0],"hard_negative_eligible":[false,true,true],"target_probabilities":[0.5,0.2,0.3],"train_policy":"policy-a","source_artifact_hash":"hash-b","release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true,"extra":"kept"}`,
		`{"row_id":"r2","source":"source-b","query":"q2","candidate_doc_ids":["p2","n2"],"candidate_texts":["another positive","another negative"],"positive_indexes":[0],"hard_negative_eligible":[false,true],"target_probabilities":[0.8,0.2],"source_artifact_hash":"hash-a","release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(rows, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	opts := EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}
	tokenizer := newEmbeddingTextDatasetTestTokenizer(t)
	wantText, err := ReadEmbeddingTextScoreSpectrumExamplesFile(path, opts)
	if err != nil {
		t.Fatalf("read baseline text rows: %v", err)
	}
	want, err := TokenizeEmbeddingTextScoreSpectrumExamples(wantText, tokenizer, opts)
	if err != nil {
		t.Fatalf("tokenize baseline text rows: %v", err)
	}
	source, err := OpenEmbeddingTextScoreSpectrumSource(path, tokenizer, opts)
	if err != nil {
		t.Fatalf("open text source: %v", err)
	}
	defer source.Close()
	if source.Len() != len(want) {
		t.Fatalf("source length = %d, want %d", source.Len(), len(want))
	}
	if source.CandidateCount(0) != 2 || source.CandidateCount(1) != 2 || source.CandidateCount(-1) != 0 || source.CandidateCount(2) != 0 {
		t.Fatalf("candidate counts = %d/%d/%d/%d, want 2/2/0/0", source.CandidateCount(0), source.CandidateCount(1), source.CandidateCount(-1), source.CandidateCount(2))
	}
	if source.MaxCandidateCount() != 2 {
		t.Fatalf("max candidate count = %d, want 2", source.MaxCandidateCount())
	}
	for i := range want {
		got, err := source.Example(i)
		if err != nil {
			t.Fatalf("source example %d: %v", i, err)
		}
		if !reflect.DeepEqual(got, want[i]) {
			t.Fatalf("source example %d = %+v, want %+v", i, got, want[i])
		}
	}
	policy := source.SourcePolicy()
	if !policy.ScoreSpectrumTrain || !policy.ScoreSpectrumResearchOnly || !policy.TrainAllowedForResearch || policy.ReleaseTrainAllowed || policy.CommercialUseAllowed || policy.ScoreSpectrumRowCount != 2 {
		t.Fatalf("source policy = %+v, want research-only two-row policy", policy)
	}
	if !reflect.DeepEqual(policy.SourceArtifactHashes, []string{"hash-a", "hash-b"}) {
		t.Fatalf("source hashes = %v, want sorted hashes", policy.SourceArtifactHashes)
	}
	policy.SourceArtifactHashes[0] = "mutated"
	if source.SourcePolicy().SourceArtifactHashes[0] != "hash-a" {
		t.Fatal("source policy returned aliased source hash slice")
	}
}

func TestEmbeddingScoreSpectrumSourceTopKMetadataHonorsExplicitRowWeightZero(t *testing.T) {
	dir := t.TempDir()
	textPath := filepath.Join(dir, "score-spectrum.jsonl")
	textRows := []string{
		`{"row_id":"default","query":"q1","candidate_doc_ids":["p","n"],"candidate_sources":["qrel","q3"],"qrel_gains":[1,0],"candidate_texts":["positive","negative"],"positive_indexes":[0],"hard_negative_eligible":[false,true],"target_probabilities":[1,0],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
		`{"row_id":"disabled","query":"q2","candidate_doc_ids":["p","n"],"candidate_sources":["qrel","q3"],"qrel_gains":[1,0],"candidate_texts":["positive","negative"],"positive_indexes":[0],"hard_negative_eligible":[false,true],"target_probabilities":[1,0],"turboquant_topk_loss_weight":0,"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}`,
	}
	if err := os.WriteFile(textPath, []byte(strings.Join(textRows, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write text source: %v", err)
	}
	opts := EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}
	textSource, err := OpenEmbeddingTextScoreSpectrumSource(textPath, newEmbeddingTextDatasetTestTokenizer(t), opts)
	if err != nil {
		t.Fatalf("open text source: %v", err)
	}
	defer textSource.Close()
	meta, ok := textSource.(EmbeddingScoreSpectrumTopKMetadataSource)
	if !ok {
		t.Fatal("text source missing top-k metadata interface")
	}
	if got := meta.TopKEligiblePairCount(0, TurboQuantTopKNegativeMaskQ3); got != 1 {
		t.Fatalf("default text row top-k pairs = %d, want 1", got)
	}
	if got := meta.TopKEligiblePairCount(1, TurboQuantTopKNegativeMaskQ3); got != 0 {
		t.Fatalf("zero-weight text row top-k pairs = %d, want 0", got)
	}
	objectiveMeta, ok := textSource.(EmbeddingScoreSpectrumObjectiveMetadataSource)
	if !ok {
		t.Fatal("text source missing objective metadata interface")
	}
	if objectiveMeta.BaseLossDisabled(0) || objectiveMeta.BaseLossDisabled(1) {
		t.Fatalf("text source base disabled flags = %v/%v, want false/false", objectiveMeta.BaseLossDisabled(0), objectiveMeta.BaseLossDisabled(1))
	}

	tokenizedPath := filepath.Join(dir, "score-spectrum-tokenized.jsonl")
	rows := []EmbeddingScoreSpectrumExample{
		{
			RowID:                   "default",
			QueryTokens:             []int32{1},
			CandidateIDs:            []string{"p", "n"},
			CandidateSources:        []string{"qrel", "q3"},
			QrelGains:               []float32{1, 0},
			CandidateTokens:         [][]int32{{1}, {2}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			TrainAllowedForResearch: true,
		},
		{
			RowID:                    "disabled",
			QueryTokens:              []int32{1},
			CandidateIDs:             []string{"p", "n"},
			CandidateSources:         []string{"qrel", "q3"},
			QrelGains:                []float32{1, 0},
			CandidateTokens:          [][]int32{{1}, {2}},
			PositiveIndexes:          []int{0},
			HardNegativeEligible:     []bool{false, true},
			TargetProbabilities:      []float32{1, 0},
			TurboQuantTopKLossWeight: float32Ptr(0),
			TrainAllowedForResearch:  true,
		},
		{
			RowID:                   "aux-only",
			QueryTokens:             []int32{1},
			CandidateIDs:            []string{"p", "n"},
			CandidateSources:        []string{"qrel", "q3"},
			QrelGains:               []float32{1, 0},
			CandidateTokens:         [][]int32{{1}, {2}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			BaseLossWeight:          float32Ptr(0),
			TrainAllowedForResearch: true,
		},
	}
	if err := WriteEmbeddingScoreSpectrumExamplesFile(tokenizedPath, rows, opts); err != nil {
		t.Fatalf("write tokenized source: %v", err)
	}
	tokenizedSource, err := OpenEmbeddingScoreSpectrumSource(tokenizedPath, opts)
	if err != nil {
		t.Fatalf("open tokenized source: %v", err)
	}
	defer tokenizedSource.Close()
	tokenizedMeta, ok := tokenizedSource.(EmbeddingScoreSpectrumTopKMetadataSource)
	if !ok {
		t.Fatal("tokenized source missing top-k metadata interface")
	}
	if got := tokenizedMeta.TopKEligiblePairCount(0, TurboQuantTopKNegativeMaskQ3); got != 1 {
		t.Fatalf("default tokenized row top-k pairs = %d, want 1", got)
	}
	if got := tokenizedMeta.TopKEligiblePairCount(1, TurboQuantTopKNegativeMaskQ3); got != 0 {
		t.Fatalf("zero-weight tokenized row top-k pairs = %d, want 0", got)
	}
	tokenizedObjectiveMeta, ok := tokenizedSource.(EmbeddingScoreSpectrumObjectiveMetadataSource)
	if !ok {
		t.Fatal("tokenized source missing objective metadata interface")
	}
	if !tokenizedObjectiveMeta.BaseLossDisabled(2) {
		t.Fatal("tokenized source did not preserve explicit base_loss_weight=0 metadata")
	}
}

func TestEmbeddingScoreSpectrumSourceRereadReturnsFreshRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	row := `{"row_id":"r1","source":"source","query":"query","candidate_doc_ids":["p","n"],"candidate_texts":["positive","negative"],"positive_indexes":[0],"hard_negative_eligible":[false,true],"target_probabilities":[0.8,0.2],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true,"provenance":{"v":1}}` + "\n"
	if err := os.WriteFile(path, []byte(row), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	opts := EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}
	source, err := OpenEmbeddingTextScoreSpectrumSource(path, newEmbeddingTextDatasetTestTokenizer(t), opts)
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer source.Close()
	first, err := source.Example(0)
	if err != nil {
		t.Fatalf("first example: %v", err)
	}
	first.QueryTokens[0] = 99
	first.QueryMask[0] = 0
	first.CandidateTokens[0][0] = 99
	first.CandidateMasks[0][0] = 0
	first.CandidateIDs[0] = "mutated"
	first.PositiveIndexes[0] = 99
	first.HardNegativeEligible[1] = false
	first.TargetProbabilities[0] = 0
	first.ExtraFields["provenance"][0] = '{'
	second, err := source.Example(0)
	if err != nil {
		t.Fatalf("second example: %v", err)
	}
	if second.QueryTokens[0] == 99 || second.QueryMask[0] == 0 || second.CandidateTokens[0][0] == 99 || second.CandidateMasks[0][0] == 0 || second.CandidateIDs[0] == "mutated" || second.PositiveIndexes[0] == 99 || !second.HardNegativeEligible[1] || second.TargetProbabilities[0] == 0 {
		t.Fatalf("second example shares mutable row state: %+v", second)
	}
	if string(second.ExtraFields["provenance"]) != `{"v":1}` {
		t.Fatalf("second extra fields = %s, want pristine JSON", second.ExtraFields["provenance"])
	}
}

func TestEmbeddingTextScoreSpectrumSourceRejectsResearchRowsWithoutAllow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum.jsonl")
	row := `{"query":"q","candidate_doc_ids":["p","n"],"candidate_texts":["positive","negative"],"positive_indexes":[0],"target_probabilities":[0.8,0.2],"release_train_allowed":false,"commercial_use_allowed":false,"train_allowed_for_research":true}` + "\n"
	if err := os.WriteFile(path, []byte(row), 0o644); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	if _, err := OpenEmbeddingTextScoreSpectrumSource(path, newEmbeddingTextDatasetTestTokenizer(t)); err == nil || !strings.Contains(err.Error(), "research-only") {
		t.Fatalf("open without research permission error = %v, want research-only rejection", err)
	}
	if source, err := OpenEmbeddingTextScoreSpectrumSource(path, newEmbeddingTextDatasetTestTokenizer(t), EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}); err != nil {
		t.Fatalf("open with research permission: %v", err)
	} else {
		_ = source.Close()
	}
}

func TestEmbeddingTokenizedScoreSpectrumSourceMatchesReaderAndFreshness(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum-tokenized.jsonl")
	rows := []EmbeddingScoreSpectrumExample{
		{
			RowID:                   "r1",
			Source:                  "tokenized",
			QueryTokens:             []int32{1, 2},
			QueryMask:               []int32{1, 1},
			CandidateIDs:            []string{"p", "n"},
			CandidateTokens:         [][]int32{{3}, {4, 5}},
			CandidateMasks:          [][]int32{{1}, {1, 1}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{0.75, 0.25},
			TrainPolicy:             "tokenized",
			TrainAllowedForResearch: true,
			SourceArtifactHash:      "hash",
			ExtraFields: map[string]json.RawMessage{
				"provenance": json.RawMessage(`{"version":1}`),
			},
		},
	}
	opts := EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}
	if err := WriteEmbeddingScoreSpectrumExamplesFile(path, rows, opts); err != nil {
		t.Fatalf("write tokenized dataset: %v", err)
	}
	want, err := ReadEmbeddingScoreSpectrumExamplesFile(path, opts)
	if err != nil {
		t.Fatalf("read baseline tokenized rows: %v", err)
	}
	source, err := OpenEmbeddingScoreSpectrumSource(path, opts)
	if err != nil {
		t.Fatalf("open tokenized source: %v", err)
	}
	defer source.Close()
	got, err := source.Example(0)
	if err != nil {
		t.Fatalf("source example: %v", err)
	}
	if !reflect.DeepEqual(got, want[0]) {
		t.Fatalf("source example = %+v, want %+v", got, want[0])
	}
	if source.Len() != 1 || source.CandidateCount(0) != 2 || source.MaxCandidateCount() != 2 {
		t.Fatalf("source counts = len %d candidates %d max %d, want 1/2/2", source.Len(), source.CandidateCount(0), source.MaxCandidateCount())
	}
	got.CandidateTokens[0][0] = 99
	got.ExtraFields["missing"] = json.RawMessage(`true`)
	fresh, err := source.Example(0)
	if err != nil {
		t.Fatalf("fresh source example: %v", err)
	}
	if fresh.CandidateTokens[0][0] == 99 || len(fresh.ExtraFields) != len(want[0].ExtraFields) {
		t.Fatal("tokenized source returned aliased state")
	}
}

func TestEmbeddingTokenizedScoreSpectrumSourceAcceptsLargeJSONLRecord(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum-large.jsonl")
	padding := strings.Repeat("x", 2_300_000)
	row := fmt.Sprintf(`{"row_id":"large","query_tokens":[1],"candidate_tokens":[[2]],"candidate_ids":["p"],"positive_indexes":[0],"hard_negative_eligible":[false],"target_probabilities":[1],"release_train_allowed":true,"commercial_use_allowed":true,"train_allowed_for_research":false,"padding":%q}`+"\n", padding)
	if len(row) <= 2_200_000 {
		t.Fatalf("large row length = %d, want >2.2MB", len(row))
	}
	if err := os.WriteFile(path, []byte(row), 0o644); err != nil {
		t.Fatalf("write large score-spectrum dataset: %v", err)
	}
	source, err := OpenEmbeddingScoreSpectrumSource(path)
	if err != nil {
		t.Fatalf("open large source: %v", err)
	}
	defer source.Close()
	if source.Len() != 1 || source.CandidateCount(0) != 1 {
		t.Fatalf("large source counts = %d/%d, want 1/1", source.Len(), source.CandidateCount(0))
	}
	got, err := source.Example(0)
	if err != nil {
		t.Fatalf("read large source example: %v", err)
	}
	if got.RowID != "large" || len(got.CandidateTokens) != 1 || got.CandidateTokens[0][0] != 2 {
		t.Fatalf("large source example = %+v, want compact tokenized row", got)
	}
}

func TestEmbeddingTextScoreSpectrumSourceRejectsOverLimitRecords(t *testing.T) {
	for _, terminated := range []bool{true, false} {
		name := "unterminated"
		if terminated {
			name = "newline-terminated"
		}
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "score-spectrum-over-limit.jsonl")
			data := []byte(strings.Repeat("x", embeddingJSONLMaxRecordBytes+1))
			if terminated {
				data = append(data, '\n')
			}
			if err := os.WriteFile(path, data, 0o644); err != nil {
				t.Fatalf("write over-limit text dataset: %v", err)
			}
			_, err := OpenEmbeddingTextScoreSpectrumSource(path, newEmbeddingTextDatasetTestTokenizer(t))
			if err == nil || !strings.Contains(err.Error(), "line 1: record exceeds maximum size") {
				t.Fatalf("open over-limit text source error = %v, want bounded record-size rejection", err)
			}
		})
	}
}

func TestEmbeddingTokenizedScoreSpectrumSourceRejectsOverLimitRecords(t *testing.T) {
	for _, terminated := range []bool{true, false} {
		name := "unterminated"
		if terminated {
			name = "newline-terminated"
		}
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "score-spectrum-over-limit.jsonl")
			data := []byte(strings.Repeat("x", embeddingJSONLMaxRecordBytes+1))
			if terminated {
				data = append(data, '\n')
			}
			if err := os.WriteFile(path, data, 0o644); err != nil {
				t.Fatalf("write over-limit tokenized dataset: %v", err)
			}
			_, err := OpenEmbeddingScoreSpectrumSource(path)
			if err == nil || !strings.Contains(err.Error(), "line 1: record exceeds maximum size") {
				t.Fatalf("open over-limit tokenized source error = %v, want bounded record-size rejection", err)
			}
		})
	}
}

func TestEmbeddingTextScoreSpectrumSourcePreservesCRLFOffsetsAndFinalUnterminatedRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "score-spectrum-crlf.jsonl")
	row := func(id string) string {
		return fmt.Sprintf(`{"row_id":%q,"query":"query-%s","candidate_doc_ids":["p","n"],"candidate_texts":["positive","negative"],"positive_indexes":[0],"hard_negative_eligible":[false,true],"target_probabilities":[0.8,0.2],"release_train_allowed":true,"commercial_use_allowed":true,"train_allowed_for_research":false}`, id, id)
	}
	if err := os.WriteFile(path, []byte(row("r1")+"\r\n"+row("r2")), 0o644); err != nil {
		t.Fatalf("write CRLF score-spectrum dataset: %v", err)
	}
	source, err := OpenEmbeddingTextScoreSpectrumSource(path, newEmbeddingTextDatasetTestTokenizer(t))
	if err != nil {
		t.Fatalf("open CRLF score-spectrum source: %v", err)
	}
	defer source.Close()
	if source.Len() != 2 {
		t.Fatalf("CRLF source length = %d, want 2", source.Len())
	}
	first, err := source.Example(0)
	if err != nil {
		t.Fatalf("read first CRLF row: %v", err)
	}
	second, err := source.Example(1)
	if err != nil {
		t.Fatalf("read final unterminated row: %v", err)
	}
	if first.RowID != "r1" || second.RowID != "r2" {
		t.Fatalf("CRLF row ids = %q/%q, want r1/r2", first.RowID, second.RowID)
	}
}

func TestEmbeddingScoreSpectrumSliceSourceCopiesRowsAndPolicy(t *testing.T) {
	selected := 0
	examples := []EmbeddingScoreSpectrumExample{{
		RowID:                   "r1",
		QueryTokens:             []int32{1},
		CandidateTokens:         [][]int32{{2}, {3}},
		CandidateIDs:            []string{"p", "n"},
		PositiveIndexes:         []int{0},
		SelectedPositiveIndex:   &selected,
		CandidateMasks:          [][]int32{{1}, {1}},
		HardNegativeEligible:    []bool{false, true},
		TargetProbabilities:     []float32{0.9, 0.1},
		TrainAllowedForResearch: true,
		SourceArtifactHash:      "hash",
	}}
	opts := EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}
	source, err := NewEmbeddingScoreSpectrumSliceSource(examples, opts)
	if err != nil {
		t.Fatalf("new slice source: %v", err)
	}
	defer source.Close()
	examples[0].CandidateTokens[0][0] = 99
	got, err := source.Example(0)
	if err != nil {
		t.Fatalf("slice source example: %v", err)
	}
	if got.CandidateTokens[0][0] == 99 || source.SourcePolicy().ScoreSpectrumRowCount != 1 {
		t.Fatalf("slice source did not isolate input row: %+v", got)
	}
}
