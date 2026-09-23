package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func syntheticRecord(spec sampleSpec, steps int, wall int64, residual float64) resultRecord {
	shape, err := legacyShapeForProfile(spec.Cell.Profile)
	if err != nil {
		panic(err)
	}
	legacy, err := expectedLegacyStatsWire(spec, steps)
	if err != nil {
		panic(err)
	}
	warmupSteps := 1
	if spec.Cell.Graph == 1 && spec.Events == 1 {
		warmupSteps = 2
	}
	record := resultRecord{
		Schema: resultSchema, SchemaVersion: resultSchemaVersion, Profile: spec.Cell.Profile,
		Modes: resultModes{Graph: spec.Cell.Graph, Upload: spec.Cell.Upload, Events: spec.Events, CUBLAS: 0, SyncEach: 0, ResidentTrain: 1, PackedForward: 0},
		Graph: spec.Cell.Graph, Upload: spec.Cell.Upload, Events: spec.Events, Shape: shape.Shape,
		WarmupSteps: warmupSteps, MeasuredSteps: steps, TelemetryMirrorEqual: spec.Events == 1,
	}
	record.WallNanosPerStep = make([]int64, steps)
	record.TopLevelHostNanos = make([]int64, steps)
	record.DispatchHostNanos = make([]int64, steps)
	record.ResidualFractions = make([]float64, steps)
	for i := 0; i < steps; i++ {
		record.WallNanosPerStep[i] = wall
		host := wall - int64(float64(wall)*residual)
		record.TopLevelHostNanos[i] = host
		record.DispatchHostNanos[i] = host
		record.ResidualFractions[i] = float64(absInt64(wall-host)) / float64(wall)
	}
	if spec.Events == 1 {
		record.DeviceEventNanos = make([]int64, steps)
		for i := range record.DeviceEventNanos {
			record.DeviceEventNanos[i] = 10
		}
		ownerPhases := syntheticOwnerPhases(spec.Cell.Graph)
		mirrorPhases := map[string]phaseWire{"k5_batch": ownerPhases["k5_batch"]}
		record.TelemetryOwner = mustJSON(telemetryWire{SchemaVersion: resultSchemaVersion, Enabled: true, EventTiming: true, View: "compact_train", Phases: ownerPhases})
		record.TelemetryMirror = mustJSON(telemetryWire{SchemaVersion: resultSchemaVersion, Enabled: true, EventTiming: true, View: "optimizer", Phases: mirrorPhases})
		record.TelemetryMirrorK5 = mustJSON(ownerPhases["k5_batch"])
		record.PhaseTelemetry = mustJSON(ownerPhases)
		legacy.Telemetry = append(json.RawMessage(nil), record.TelemetryOwner...)
	}
	record.LegacyStats = mustJSON(legacy)
	return record
}

func mustJSON(value any) json.RawMessage {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return data
}

func syntheticOwnerPhases(graph int) map[string]phaseWire {
	plain := func() phaseWire {
		return phaseWire{GoCalls: 1, ContextSets: 1, DriverCalls: 1, Attempted: 1, Enqueued: 1, Completed: 1}
	}
	event := func() phaseWire {
		phase := plain()
		phase.EventRecords, phase.EventQueries, phase.DeviceElapsedNanos = 1, 1, 1
		return phase
	}
	phases := map[string]phaseWire{
		"begin_zero": plain(), "input_h2d": plain(), "forward_boundary": plain(), "grad_h2d": plain(),
		"backward_boundary": plain(), "k6_readback": plain(), "backward_final": event(),
		"backward_ffn": event(), "backward_attention": event(), "k5_batch": event(),
	}
	if graph == 1 {
		phases["forward_replay"] = event()
	} else {
		phases["forward_direct"] = event()
	}
	return phases
}

func TestParseResultRecordExactlyOneAndRejectsMalformed(t *testing.T) {
	spec := sampleSpec{Cell: matrixCell{Profile: "canonical", Graph: 0, Upload: 0}, Pair: 1, Events: 0}
	record := syntheticRecord(spec, 2, 100, 0.01)
	data, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parseResultRecord("noise\nk9_result_json=" + string(data) + "\n")
	if err != nil {
		t.Fatalf("parse valid result: %v", err)
	}
	if parsed.Schema != resultSchema {
		t.Fatalf("schema = %q", parsed.Schema)
	}
	for _, log := range []string{
		"k9_result_json={}\nk9_result_json={}\n",
		"k9_result_json={not-json}\n",
		"benchmark completed\n",
	} {
		if _, err := parseResultRecord(log); err == nil {
			t.Fatalf("parse unexpectedly accepted %q", log)
		}
	}
	if err := record.validate(spec, 2); err != nil {
		t.Fatalf("valid record failed validation: %v", err)
	}
	record.SchemaVersion++
	if err := record.validate(spec, 2); err == nil {
		t.Fatal("schema-version mismatch was accepted")
	}
}

func TestResultValidationFailsClosedForEventOffTelemetryAndWrongMode(t *testing.T) {
	spec := sampleSpec{Cell: matrixCell{Profile: "next", Graph: 1, Upload: 1}, Pair: 1, Events: 0}
	record := syntheticRecord(spec, 2, 100, 0.01)
	record.TelemetryOwner = json.RawMessage(`{"enabled":false}`)
	if err := record.validate(spec, 2); err == nil {
		t.Fatal("event-off telemetry was accepted")
	}
	record = syntheticRecord(spec, 2, 100, 0.01)
	wrong := spec
	wrong.Events = 1
	if err := record.validate(wrong, 2); err == nil {
		t.Fatal("mode mismatch was accepted")
	}
	record = syntheticRecord(spec, 2, 100, 0.01)
	record.WallNanosPerStep[1] = 0
	if err := record.validate(spec, 2); err == nil {
		t.Fatal("non-positive wall sample was accepted")
	}
}

func TestStrictTelemetryValidationRejectsAdversarialRecords(t *testing.T) {
	spec := sampleSpec{Cell: matrixCell{Profile: "canonical", Graph: 0, Upload: 0}, Pair: 1, Events: 1}
	base := syntheticRecord(spec, 2, 100, 0.01)
	tests := []struct {
		name   string
		mutate func(*resultRecord)
	}{
		{name: "unknown phase", mutate: func(record *resultRecord) {
			var owner telemetryWire
			if err := decodeStrictJSON(record.TelemetryOwner, &owner); err != nil {
				t.Fatal(err)
			}
			owner.Phases["unexpected_phase"] = phaseWire{}
			record.TelemetryOwner = mustJSON(owner)
		}},
		{name: "K5 mismatch", mutate: func(record *resultRecord) {
			var mirror telemetryWire
			if err := decodeStrictJSON(record.TelemetryMirror, &mirror); err != nil {
				t.Fatal(err)
			}
			phase := mirror.Phases["k5_batch"]
			phase.DriverCalls++
			mirror.Phases["k5_batch"] = phase
			record.TelemetryMirror = mustJSON(mirror)
		}},
		{name: "whole-view equality", mutate: func(record *resultRecord) {
			var owner telemetryWire
			if err := decodeStrictJSON(record.TelemetryOwner, &owner); err != nil {
				t.Fatal(err)
			}
			record.TelemetryMirror = mustJSON(telemetryWire{SchemaVersion: resultSchemaVersion, Enabled: true, EventTiming: true, View: "optimizer", Phases: owner.Phases})
		}},
		{name: "dispatch mismatch", mutate: func(record *resultRecord) {
			record.DispatchHostNanos[0]++
		}},
		{name: "residual mismatch", mutate: func(record *resultRecord) {
			record.ResidualFractions[0] = 0.25
		}},
		{name: "phase failure", mutate: func(record *resultRecord) {
			var owner telemetryWire
			if err := decodeStrictJSON(record.TelemetryOwner, &owner); err != nil {
				t.Fatal(err)
			}
			phase := owner.Phases["backward_boundary"]
			phase.Failures = 1
			owner.Phases["backward_boundary"] = phase
			record.TelemetryOwner = mustJSON(owner)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			record := base
			record.WallNanosPerStep = append([]int64(nil), base.WallNanosPerStep...)
			record.TopLevelHostNanos = append([]int64(nil), base.TopLevelHostNanos...)
			record.DispatchHostNanos = append([]int64(nil), base.DispatchHostNanos...)
			record.ResidualFractions = append([]float64(nil), base.ResidualFractions...)
			test.mutate(&record)
			if err := record.validate(spec, 2); err == nil {
				t.Fatal("adversarial telemetry record was accepted")
			}
		})
	}
}

func TestImmutableIdentityAndEnvironmentRedaction(t *testing.T) {
	base := runIdentity{Repo: "/repo", GPU: "0", GoBinary: "go", GoVersion: "go1.26", GitHEAD: "head", StagedDiffHash: "staged", UnstagedDiffHash: "unstaged", TrackedStatusHash: "status", OwnedSourceHash: "owned", GPUIdentity: gpuIdentity{Index: "0", Name: "GPU", DriverVersion: "1", ComputeCapability: "8.6", UUID: "GPU-1"}}
	if !immutableIdentityEqual(base, base) {
		t.Fatal("identical immutable identities differ")
	}
	for _, mutate := range []func(*runIdentity){
		func(value *runIdentity) { value.GitHEAD = "changed" },
		func(value *runIdentity) { value.OwnedSourceHash = "changed" },
		func(value *runIdentity) { value.GoVersion = "changed" },
		func(value *runIdentity) { value.GPUIdentity.UUID = "GPU-2" },
	} {
		changed := base
		mutate(&changed)
		if immutableIdentityEqual(base, changed) {
			t.Fatalf("identity mutation was accepted: %+v", changed)
		}
	}
	env := redactEnvironment([]string{"HOME=/home/user", "OPENAI_API_KEY=secret", "CUDA_VISIBLE_DEVICES=0", "CUSTOM=value"})
	joined := strings.Join(env, "\n")
	if strings.Contains(joined, "secret") || strings.Contains(joined, "HOME=/home/user") || !strings.Contains(joined, "OPENAI_API_KEY=<sha256:") || !strings.Contains(joined, "HOME=<sha256:") {
		t.Fatalf("redacted environment=%q", joined)
	}
}

func TestEnvironmentEvidenceIsSafeForCredentialBearingValues(t *testing.T) {
	sentinel := "k9-unique-secret-sentinel-7f4f3f"
	rawEnv := []string{
		"CUDA_VISIBLE_DEVICES=0",
		"EOS_CUDA_WARM_BENCH_PROFILE=canonical",
		"HTTP_PROXY=http://runner:" + sentinel + "@proxy.example.test:8080",
		"HTTPS_PROXY=http://runner:" + sentinel + "@proxy.example.test:8443",
		"ALL_PROXY=socks5://runner:" + sentinel + "@proxy.example.test:1080",
		"DATABASE_URL=postgres://runner:" + sentinel + "@db.example.test/eos",
		"MONGO_URI=mongodb://runner:" + sentinel + "@db.example.test/eos",
		"PG_DSN=host=db.example.test user=runner password=" + sentinel,
		"CONNECTION_STRING=Server=db.example.test;User=runner;Password=" + sentinel,
		"ARBITRARY_UNKNOWN=" + sentinel,
	}
	redacted := redactEnvironment(rawEnv)
	joined := strings.Join(redacted, "\n")
	if strings.Contains(joined, sentinel) {
		t.Fatalf("redacted environment leaked sentinel: %q", joined)
	}
	if !strings.Contains(joined, "CUDA_VISIBLE_DEVICES=0") || !strings.Contains(joined, "EOS_CUDA_WARM_BENCH_PROFILE=canonical") {
		t.Fatalf("known K9 controls were not retained: %q", joined)
	}
	for _, key := range []string{"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "DATABASE_URL", "MONGO_URI", "PG_DSN", "CONNECTION_STRING", "ARBITRARY_UNKNOWN"} {
		if !strings.Contains(joined, key+"=<sha256:") {
			t.Fatalf("%s was not hashed in evidence: %q", key, joined)
		}
	}
	if got, want := hashEnvironment(rawEnv), hashEnvironment(rawEnv); got != want {
		t.Fatalf("environment hash changed unexpectedly: got %q want %q", got, want)
	}
	keys := make(map[string]struct{}, len(redacted))
	for _, item := range redacted {
		key, _, ok := strings.Cut(item, "=")
		if !ok {
			t.Fatalf("redacted entry lacks key inventory: %q", item)
		}
		keys[key] = struct{}{}
	}
	for _, item := range rawEnv {
		key, _, _ := strings.Cut(item, "=")
		if _, ok := keys[key]; !ok {
			t.Fatalf("key %q missing from evidence inventory", key)
		}
	}

	dir := t.TempDir()
	ledger := attemptLedger{Environment: redacted, EnvironmentHash: hashEnvironment(rawEnv), Reason: "bounded test"}
	runner := sampleRunner{samplesJSON: filepath.Join(dir, "samples.jsonl"), samplesTSV: filepath.Join(dir, "samples.tsv")}
	if err := runner.writeSample(ledger); err != nil {
		t.Fatal(err)
	}
	ledgerPath := filepath.Join(dir, "attempts.jsonl")
	if err := writeJSONLine(ledgerPath, ledger); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{runner.samplesJSON, runner.samplesTSV, ledgerPath} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(data), sentinel) {
			t.Fatalf("serialized evidence %s leaked sentinel: %q", path, data)
		}
	}
}

func TestLegacyStatsStrictWireContractRejectsAdversarialRecords(t *testing.T) {
	spec := sampleSpec{Cell: matrixCell{Profile: "canonical", Graph: 0, Upload: 0}, Pair: 1, Events: 0}
	base := syntheticRecord(spec, 2, 100, 0.01)
	tests := []struct {
		name   string
		mutate func(*resultRecord)
	}{
		{name: "null", mutate: func(record *resultRecord) { record.LegacyStats = json.RawMessage("null") }},
		{name: "empty object", mutate: func(record *resultRecord) { record.LegacyStats = json.RawMessage("{}") }},
		{name: "array", mutate: func(record *resultRecord) { record.LegacyStats = json.RawMessage("[]") }},
		{name: "unknown field", mutate: func(record *resultRecord) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(record.LegacyStats, &fields); err != nil {
				t.Fatal(err)
			}
			fields["UnknownCounter"] = json.RawMessage("1")
			record.LegacyStats = mustJSON(fields)
		}},
		{name: "missing required field", mutate: func(record *resultRecord) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(record.LegacyStats, &fields); err != nil {
				t.Fatal(err)
			}
			delete(fields, "ForwardCalls")
			record.LegacyStats = mustJSON(fields)
		}},
		{name: "graph mismatch", mutate: func(record *resultRecord) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(record.LegacyStats, &fields); err != nil {
				t.Fatal(err)
			}
			fields["GraphReplays"] = json.RawMessage("1")
			record.LegacyStats = mustJSON(fields)
		}},
		{name: "last forward cublas", mutate: func(record *resultRecord) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(record.LegacyStats, &fields); err != nil {
				t.Fatal(err)
			}
			fields["LastForwardCublasGemmCalls"] = json.RawMessage("1")
			record.LegacyStats = mustJSON(fields)
		}},
		{name: "last backward cublas", mutate: func(record *resultRecord) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(record.LegacyStats, &fields); err != nil {
				t.Fatal(err)
			}
			fields["LastBackwardCublasGemmCalls"] = json.RawMessage("1")
			record.LegacyStats = mustJSON(fields)
		}},
		{name: "event-off telemetry", mutate: func(record *resultRecord) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(record.LegacyStats, &fields); err != nil {
				t.Fatal(err)
			}
			fields["telemetry"] = json.RawMessage(`{"schema_version":1}`)
			record.LegacyStats = mustJSON(fields)
		}},
		{name: "shape mismatch", mutate: func(record *resultRecord) {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(record.LegacyStats, &fields); err != nil {
				t.Fatal(err)
			}
			var shapeFields map[string]json.RawMessage
			if err := json.Unmarshal(fields["LastShape"], &shapeFields); err != nil {
				t.Fatal(err)
			}
			shapeFields["Batch"] = json.RawMessage("99")
			fields["LastShape"] = mustJSON(shapeFields)
			record.LegacyStats = mustJSON(fields)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			record := base
			test.mutate(&record)
			if err := record.validate(spec, 2); err == nil {
				t.Fatal("adversarial legacy_stats record was accepted")
			}
		})
	}

	eventSpec := sampleSpec{Cell: matrixCell{Profile: "canonical", Graph: 0, Upload: 0}, Pair: 1, Events: 1}
	eventRecord := syntheticRecord(eventSpec, 2, 100, 0.01)
	var legacy legacyStatsWire
	if err := decodeStrictJSON(eventRecord.LegacyStats, &legacy); err != nil {
		t.Fatal(err)
	}
	legacy.Telemetry = json.RawMessage(`{"schema_version":1,"enabled":true,"event_timing":true,"view":"compact_train","phases":{}}`)
	eventRecord.LegacyStats = mustJSON(legacy)
	if err := eventRecord.validate(eventSpec, 2); err == nil {
		t.Fatal("malformed legacy telemetry was accepted")
	}
}

func TestRunSummaryAlwaysRecordsRejectedState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "summary.json")
	state := &runSummaryState{PlanProcesses: 2, AcceptedSamples: 1, RejectedAttempts: 1, Replacements: 1, StartedUTC: "now"}
	failure := errors.New("identity changed")
	if err := state.write(path, failure); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var summary map[string]any
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatal(err)
	}
	if summary["status"] != "rejected" || summary["accepted"] != false || summary["failure"] != failure.Error() || summary["replacements"].(float64) != 1 {
		t.Fatalf("rejection summary=%v", summary)
	}
}

func TestIsolatedEnvironmentRemovesConflictingControlsAndPreservesSystem(t *testing.T) {
	base := []string{
		"HOME=/keep/home", "PATH=/bin", "CUDA_HOME=/opt/cuda", "EOS_CUDA_COMPACT_PROFILE_EVENTS=1",
		"EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN=0", "MY_BENCHMARK_OVERRIDE=bad", "GOFLAGS=-bench=.", "CUSTOM=value",
	}
	env := isolatedEnvironment(base, "2", matrixCell{Profile: "next", Graph: 1, Upload: 0}, 1, 20)
	values := make(map[string]string)
	for _, item := range env {
		key, value, ok := strings.Cut(item, "=")
		if ok {
			values[key] = value
		}
	}
	if values["HOME"] != "/keep/home" || values["PATH"] != "/bin" || values["CUDA_HOME"] != "/opt/cuda" || values["CUSTOM"] != "value" {
		t.Fatalf("system environment was not preserved: %+v", values)
	}
	for _, key := range []string{"EOS_CUDA_COMPACT_PROFILE_EVENTS", "EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN", "MY_BENCHMARK_OVERRIDE"} {
		if key == "MY_BENCHMARK_OVERRIDE" {
			if _, ok := values[key]; ok {
				t.Fatalf("benchmark variable %s survived isolation", key)
			}
			continue
		}
	}
	if _, ok := values["GOFLAGS"]; ok {
		t.Fatal("GOFLAGS benchmark controls survived isolation")
	}
	for key, want := range map[string]string{
		"CUDA_VISIBLE_DEVICES": "2", "EOS_CUDA_COMPACT_PROFILE_EVENTS": "1", "EOS_CUDA_COMPACT_TRAIN_FORWARD_GRAPH": "1",
		"EOS_CUDA_COMPACT_TRAIN_FORWARD_UPLOAD_BATCH": "0", "EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN": "1", "EOS_TRAIN_ENABLE_COMPACT_PACKED_FORWARD": "0",
		"EOS_CUDA_K9_MEASURED_STEPS": "20", "CGO_ENABLED": "1", "GOMAXPROCS": "1", "GOWORK": "off",
	} {
		if values[key] != want {
			t.Fatalf("isolated %s=%q, want %q", key, values[key], want)
		}
	}
	for _, item := range env {
		if strings.HasPrefix(item, "HOME=") && strings.Contains(item, "CODEX") {
			t.Fatal("HOME was repurposed")
		}
	}
}

func TestBuildProcessSpecAndHashes(t *testing.T) {
	cell := matrixCell{Profile: "canonical", Graph: 0, Upload: 1}
	base := []string{"PATH=/bin", "HOME=/tmp/home", "EOS_CUDA_BAD=1"}
	first := buildProcessSpec("/repo", "0", cell, 1, 20, base, "go")
	second := buildProcessSpec("/repo", "0", cell, 1, 20, base, "go")
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("process spec is not deterministic:\n%+v\n%+v", first, second)
	}
	wantArgs := []string{"test", "./runtime", "-run", "^$", "-bench", "^" + benchmarkName + "$", "-benchtime=1x", "-count=1", "-v"}
	if !reflect.DeepEqual(first.Args, wantArgs) {
		t.Fatalf("args=%v, want %v", first.Args, wantArgs)
	}
	if first.CommandHash == "" || first.EnvironmentHash == "" {
		t.Fatal("command/environment hashes are empty")
	}
	changed := buildProcessSpec("/repo", "0", cell, 0, 20, base, "go")
	if changed.EnvironmentHash == first.EnvironmentHash {
		t.Fatal("event mode did not change environment hash")
	}
}

func TestBalancedPlanDefaultAndSmoke(t *testing.T) {
	plan := buildMatrixPlan(12, false)
	if len(plan) != 192 {
		t.Fatalf("default plan length=%d, want 192", len(plan))
	}
	counts := map[string][2]int{}
	for _, sample := range plan {
		key := sample.Cell.key() + ":" + strconvPair(sample.Pair)
		count := counts[key]
		count[sample.Events]++
		counts[key] = count
	}
	for key, count := range counts {
		if count != [2]int{1, 1} {
			t.Fatalf("pair %s counts=%v, want one off/on", key, count)
		}
	}
	if got := balancedEventOrder(1); !reflect.DeepEqual(got, []int{0, 1}) {
		t.Fatalf("pair 1 order=%v", got)
	}
	if got := balancedEventOrder(2); !reflect.DeepEqual(got, []int{1, 0}) {
		t.Fatalf("pair 2 order=%v", got)
	}
	smoke := buildMatrixPlan(1, true)
	if len(smoke) != 2 || smoke[0].Cell.Profile != "canonical" || smoke[0].Cell.Graph != 0 || smoke[0].Cell.Upload != 0 {
		t.Fatalf("smoke plan=%+v", smoke)
	}
}

func strconvPair(pair int) string {
	return string(rune('a' + pair))
}

func TestQuietClassification(t *testing.T) {
	thresholds := quietThresholds{GPUUtil: 5, Load1: 1, LoadRise: 0.5}
	pre := gpuSnapshot{GPUUtil: 2, Load1: 0.4}
	post := gpuSnapshot{GPUUtil: 4, Load1: 0.8}
	if decision := classifyQuiet(pre, post, 123, thresholds); !decision.Eligible {
		t.Fatalf("quiet snapshots rejected: %+v", decision)
	}
	post = gpuSnapshot{GPUUtil: 6, Load1: 0.8}
	if decision := classifyQuiet(pre, post, 123, thresholds); decision.Eligible || len(decision.Reasons) == 0 {
		t.Fatal("high GPU utilization was accepted")
	}
	post = gpuSnapshot{GPUUtil: 2, Load1: 1.1}
	if decision := classifyQuiet(pre, post, 123, thresholds); decision.Eligible {
		t.Fatal("high post load was accepted")
	}
	pre.ComputePIDs = []int{999}
	post = gpuSnapshot{GPUUtil: 2, Load1: 0.4}
	if decision := classifyQuiet(pre, post, 123, thresholds); decision.Eligible {
		t.Fatal("unrelated compute PID was accepted")
	}
	during := gpuSnapshot{GPUUtil: 90, Load1: 9, ComputePIDs: []int{123}}
	if decision := classifyDuringRunNoise(during, gpuSnapshot{Load1: 0.5}, []int{123, os.Getpid()}, thresholds); !decision.Eligible {
		t.Fatalf("owned child workload was incorrectly classified as host noise: %+v", decision)
	}
	during.ComputePIDs = []int{123, 999}
	if decision := classifyDuringRunNoise(during, pre, []int{123, os.Getpid()}, thresholds); decision.Eligible {
		t.Fatal("unrelated during-run compute PID was accepted")
	}
}

func TestLockOwnershipAndBoundedContention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lock")
	ctx := context.Background()
	first, err := acquireGPULock(ctx, path, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	secondCtx, cancel := context.WithTimeout(ctx, 80*time.Millisecond)
	defer cancel()
	if _, err := acquireGPULock(secondCtx, path, 50*time.Millisecond); err == nil {
		t.Fatal("second lock acquisition unexpectedly succeeded")
	}
	if err := first.Release(); err != nil {
		t.Fatal(err)
	}
	second, err := acquireGPULock(ctx, path, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := second.Release(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lock path remains after release: %v", err)
	}
}

func TestRatiosBootstrapAndThresholdDeterminism(t *testing.T) {
	spec := sampleSpec{Cell: matrixCell{Profile: "canonical", Graph: 0, Upload: 0}, Pair: 1, Events: 0}
	samples := make([]pairedSample, 12)
	for i := range samples {
		offSpec, onSpec := spec, spec
		onSpec.Events = 1
		samples[i] = pairedSample{Cell: spec.Cell, Pair: i + 1, Off: syntheticRecord(offSpec, 2, 1000, 0.01), On: syntheticRecord(onSpec, 2, 1010, 0.01)}
	}
	first, err := summarizeCell(spec.Cell, samples, 200, 77)
	if err != nil {
		t.Fatal(err)
	}
	second, err := summarizeCell(spec.Cell, samples, 200, 77)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("bootstrap is not deterministic:\n%+v\n%+v", first, second)
	}
	if first.GeometricMeanOverhead < 0.009 || first.GeometricMeanOverhead > 0.011 || !first.Accepted {
		t.Fatalf("unexpected low-overhead summary=%+v", first)
	}
	for i := range samples {
		samples[i].On = syntheticRecord(sampleSpec{Cell: spec.Cell, Pair: i + 1, Events: 1}, 2, 1100, 0.01)
	}
	rejected, err := summarizeCell(spec.Cell, samples, 200, 77)
	if err != nil {
		t.Fatal(err)
	}
	if rejected.Accepted || rejected.GeometricMeanOverhead < 0.09 {
		t.Fatalf("high-overhead summary accepted: %+v", rejected)
	}
}

func TestParseFlagsSmokeIsBounded(t *testing.T) {
	config, err := parseFlags([]string{"-smoke", "-repo", "/repo", "-output", "/tmp/out"}, time.Unix(0, 0), io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if !config.smoke || config.pairs != 1 || config.steps != 2 || config.quietConsecutive != 2 {
		t.Fatalf("smoke config=%+v", config)
	}
	if _, err := parseFlags([]string{"-pairs", "0"}, time.Now(), io.Discard); err == nil {
		t.Fatal("invalid pair count accepted")
	}
	if _, err := parseFlags([]string{"-quiet-consecutive", "0"}, time.Now(), io.Discard); err == nil {
		t.Fatal("invalid quiet cooldown accepted")
	}
}
