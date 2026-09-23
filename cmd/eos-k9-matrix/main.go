// Command eos-k9-matrix runs the process-isolated CUDA telemetry accounting
// matrix. It is intentionally stdlib-only: the benchmark process is the
// source of truth for CUDA/runtime identities, while this command owns sample
// isolation, host-noise rejection, and paired statistics.
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	resultSchema        = "eos-k9-cuda-telemetry/v1"
	resultSchemaVersion = 1
	benchmarkName       = "BenchmarkVectorDistillCompactResidentTrainCUDAK9TelemetryMatrix"

	defaultPairs          = 12
	defaultMeasuredSteps  = 20
	defaultBootstrapDraws = 10000

	defaultQuietWait       = 2 * time.Minute
	defaultQuietPoll       = 1 * time.Second
	defaultProcessTimeout  = 20 * time.Minute
	defaultLockWait        = 2 * time.Minute
	defaultMaxReplacements = 3
)

type runnerConfig struct {
	repo             string
	gpu              string
	pairs            int
	steps            int
	output           string
	quietWait        time.Duration
	quietPoll        time.Duration
	quietConsecutive int
	processTimeout   time.Duration
	lockWait         time.Duration
	maxReplacements  int
	quietGPUUtil     float64
	quietLoad1       float64
	quietLoadRise    float64
	bootstrapDraws   int
	seed             int64
	smoke            bool
	dryRun           bool
	goBinary         string
	stdout           io.Writer
}

type matrixCell struct {
	Profile string `json:"profile"`
	Graph   int    `json:"graph"`
	Upload  int    `json:"upload"`
}

func (c matrixCell) key() string {
	return fmt.Sprintf("%s-g%d-u%d", c.Profile, c.Graph, c.Upload)
}

type sampleSpec struct {
	Cell   matrixCell
	Pair   int
	Events int
	Order  int
}

func (s sampleSpec) key() string {
	return fmt.Sprintf("%s-pair-%02d-%s", s.Cell.key(), s.Pair, eventLabel(s.Events))
}

func eventLabel(events int) string {
	if events == 1 {
		return "on"
	}
	return "off"
}

func balancedEventOrder(pair int) []int {
	if pair < 1 {
		return nil
	}
	if pair%2 == 1 {
		return []int{0, 1}
	}
	return []int{1, 0}
}

func buildMatrixPlan(pairs int, smoke bool) []sampleSpec {
	if pairs < 1 {
		return nil
	}
	profiles := []string{"canonical", "next"}
	cells := make([]matrixCell, 0, 8)
	if smoke {
		cells = append(cells, matrixCell{Profile: "canonical", Graph: 0, Upload: 0})
	} else {
		for _, profile := range profiles {
			for graph := 0; graph <= 1; graph++ {
				for upload := 0; upload <= 1; upload++ {
					cells = append(cells, matrixCell{Profile: profile, Graph: graph, Upload: upload})
				}
			}
		}
	}
	plan := make([]sampleSpec, 0, len(cells)*pairs*2)
	for _, cell := range cells {
		for pair := 1; pair <= pairs; pair++ {
			for order, events := range balancedEventOrder(pair) {
				plan = append(plan, sampleSpec{Cell: cell, Pair: pair, Events: events, Order: order})
			}
		}
	}
	return plan
}

type resultModes struct {
	Graph         int `json:"graph"`
	Upload        int `json:"upload"`
	Events        int `json:"events"`
	CUBLAS        int `json:"cublas"`
	SyncEach      int `json:"sync_each"`
	ResidentTrain int `json:"resident_train"`
	PackedForward int `json:"packed_forward"`
}

// resultRecord mirrors only the machine-readable benchmark envelope. The
// nested telemetry value remains RawMessage so this command stays independent
// of runtime packages and can reject malformed records before any statistical
// aggregation. legacy_stats is decoded into the command-local wire contract
// below during validation.
type resultRecord struct {
	Schema               string          `json:"schema"`
	SchemaVersion        int             `json:"schema_version"`
	Profile              string          `json:"profile"`
	Shape                string          `json:"shape"`
	Modes                resultModes     `json:"modes"`
	Graph                int             `json:"graph"`
	Upload               int             `json:"upload"`
	Events               int             `json:"events"`
	WarmupSteps          int             `json:"warmup_steps"`
	MeasuredSteps        int             `json:"measured_steps"`
	WallNanosPerStep     []int64         `json:"wall_ns_per_step"`
	TopLevelHostNanos    []int64         `json:"top_level_host_ns"`
	DeviceEventNanos     []int64         `json:"device_event_ns"`
	DispatchHostNanos    []int64         `json:"dispatch_host_ns"`
	ResidualFractions    []float64       `json:"residual_fraction"`
	LegacyStats          json.RawMessage `json:"legacy_stats"`
	TelemetryOwner       json.RawMessage `json:"telemetry_owner"`
	TelemetryMirror      json.RawMessage `json:"telemetry_mirror"`
	TelemetryMirrorK5    json.RawMessage `json:"optimizer_k5_mirror"`
	TelemetryMirrorEqual bool            `json:"telemetry_mirror_equal"`
	PhaseTelemetry       json.RawMessage `json:"phase_telemetry"`

	ownerWire  *telemetryWire
	mirrorWire *telemetryWire
	phaseWire  map[string]phaseWire
}

type phaseWire struct {
	GoCalls              int64  `json:"go_calls,omitempty"`
	ContextSets          int64  `json:"context_sets,omitempty"`
	DriverCalls          int64  `json:"driver_calls,omitempty"`
	KernelLaunches       int64  `json:"kernel_launches,omitempty"`
	CublasCalls          int64  `json:"cublas_calls,omitempty"`
	GraphBegin           int64  `json:"graph_begin,omitempty"`
	GraphEnd             int64  `json:"graph_end,omitempty"`
	GraphInstantiate     int64  `json:"graph_instantiate,omitempty"`
	GraphLaunches        int64  `json:"graph_launches,omitempty"`
	GraphNodes           int64  `json:"graph_nodes,omitempty"`
	GraphKernelNodes     int64  `json:"graph_kernel_nodes,omitempty"`
	GraphCublasNodes     int64  `json:"graph_cublas_nodes,omitempty"`
	GraphUnknownNodes    int64  `json:"graph_unknown_nodes,omitempty"`
	H2DCopies            int64  `json:"h2d_copies,omitempty"`
	H2DBytes             int64  `json:"h2d_bytes,omitempty"`
	D2HCopies            int64  `json:"d2h_copies,omitempty"`
	D2HBytes             int64  `json:"d2h_bytes,omitempty"`
	MemsetCalls          int64  `json:"memset_calls,omitempty"`
	HostDescriptorAllocs int64  `json:"host_descriptor_allocs,omitempty"`
	StreamSynchronizes   int64  `json:"stream_syncs,omitempty"`
	EventRecords         int64  `json:"event_records,omitempty"`
	EventQueries         int64  `json:"event_queries,omitempty"`
	HostNanos            int64  `json:"host_nanos,omitempty"`
	DeviceElapsedNanos   int64  `json:"device_elapsed_nanos,omitempty"`
	Attempted            int64  `json:"attempted,omitempty"`
	Enqueued             int64  `json:"enqueued,omitempty"`
	Completed            int64  `json:"completed,omitempty"`
	Failures             int64  `json:"failures,omitempty"`
	FailIndex            int64  `json:"fail_index,omitempty"`
	HasFailIndex         bool   `json:"has_fail_index,omitempty"`
	FailureStage         string `json:"failure_stage,omitempty"`
	EventControlFailures int64  `json:"event_control_failures,omitempty"`
	EventControlStage    string `json:"event_control_stage,omitempty"`
}

type telemetryWire struct {
	SchemaVersion int                  `json:"schema_version"`
	Enabled       bool                 `json:"enabled,omitempty"`
	EventTiming   bool                 `json:"event_timing,omitempty"`
	View          string               `json:"view,omitempty"`
	Phases        map[string]phaseWire `json:"phases,omitempty"`
}

// legacyShapeWire mirrors backend.CompactForwardShape's JSON representation
// without importing the runtime package. The benchmark only admits the two
// prevalidated shapes below, so the runner can independently verify the
// profile identity reported by legacy_stats.
type legacyShapeWire struct {
	Batch               int  `json:"Batch"`
	Tokens              int  `json:"Tokens"`
	ModelDim            int  `json:"ModelDim"`
	FFNDim              int  `json:"FFNDim"`
	Heads               int  `json:"Heads"`
	HeadDim             int  `json:"HeadDim"`
	Layers              int  `json:"Layers"`
	OutputDim           int  `json:"OutputDim"`
	HasOutputProjection bool `json:"HasOutputProjection"`
}

// legacyStatsWire is intentionally command-local. The field names match the
// default encoding/json names of backend.CompactTrainAcceleratorStats, with
// the existing lower-case telemetry tag preserved. All non-telemetry fields
// are emitted by the benchmark even when their value is zero.
type legacyStatsWire struct {
	ForwardCalls                      int64           `json:"ForwardCalls"`
	BackwardCalls                     int64           `json:"BackwardCalls"`
	HandlesCreated                    int64           `json:"HandlesCreated"`
	HandlesReleased                   int64           `json:"HandlesReleased"`
	LiveHandles                       int64           `json:"LiveHandles"`
	ArenaReuseHits                    int64           `json:"ArenaReuseHits"`
	ArenaAllocations                  int64           `json:"ArenaAllocations"`
	GradientReuseHits                 int64           `json:"GradientReuseHits"`
	GradientAllocations               int64           `json:"GradientAllocations"`
	GradientZeroCalls                 int64           `json:"GradientZeroCalls"`
	ResidentGradBytes                 int64           `json:"ResidentGradBytes"`
	ActivationArenaBytes              int64           `json:"ActivationArenaBytes"`
	WorkspaceArenaBytes               int64           `json:"WorkspaceArenaBytes"`
	UploadedBytes                     int64           `json:"UploadedBytes"`
	DownloadedBytes                   int64           `json:"DownloadedBytes"`
	PooledDownloadedBytes             int64           `json:"PooledDownloadedBytes"`
	GradPooledUploadedBytes           int64           `json:"GradPooledUploadedBytes"`
	StatusDownloadedBytes             int64           `json:"StatusDownloadedBytes"`
	ForwardReadbackBatchEntries       int64           `json:"ForwardReadbackBatchEntries"`
	ForwardReadbackContextSets        int64           `json:"ForwardReadbackContextSets"`
	ForwardReadbackDeviceCopies       int64           `json:"ForwardReadbackDeviceCopies"`
	ForwardInputUploadBatchCalls      int64           `json:"ForwardInputUploadBatchCalls"`
	ForwardInputUploadContextSets     int64           `json:"ForwardInputUploadContextSets"`
	ForwardInputUploadDeviceCopies    int64           `json:"ForwardInputUploadDeviceCopies"`
	ForwardInputUploadFailures        int64           `json:"ForwardInputUploadFailures"`
	ForwardInputUploadScalarFallbacks int64           `json:"ForwardInputUploadScalarFallbacks"`
	PackedBytesAvoided                int64           `json:"PackedBytesAvoided"`
	HostGradUploadBytesAvoided        int64           `json:"HostGradUploadBytesAvoided"`
	KernelLaunches                    int64           `json:"KernelLaunches"`
	CublasGemmCalls                   int64           `json:"CublasGemmCalls"`
	KernelSynchronizations            int64           `json:"KernelSynchronizations"`
	GraphCaptures                     int64           `json:"GraphCaptures"`
	GraphReplays                      int64           `json:"GraphReplays"`
	GraphLaunches                     int64           `json:"GraphLaunches"`
	GraphNodes                        int64           `json:"GraphNodes"`
	GraphCaptureFailures              int64           `json:"GraphCaptureFailures"`
	GraphReplayFailures               int64           `json:"GraphReplayFailures"`
	GraphInvalidations                int64           `json:"GraphInvalidations"`
	GraphParityFailures               int64           `json:"GraphParityFailures"`
	GraphFallbacks                    int64           `json:"GraphFallbacks"`
	GraphSynchronizations             int64           `json:"GraphSynchronizations"`
	DirectForwardSubmissions          int64           `json:"DirectForwardSubmissions"`
	GraphExecutedNodes                int64           `json:"GraphExecutedNodes"`
	ForwardDeviceKernelWork           int64           `json:"ForwardDeviceKernelWork"`
	FallbackOrUnhandled               int64           `json:"FallbackOrUnhandled"`
	ForwardNanos                      int64           `json:"ForwardNanos"`
	BackwardNanos                     int64           `json:"BackwardNanos"`
	OptimizerResidentGradNanos        int64           `json:"OptimizerResidentGradNanos"`
	LastShape                         legacyShapeWire `json:"LastShape"`
	LastForwardLaunches               int64           `json:"LastForwardLaunches"`
	LastBackwardLaunches              int64           `json:"LastBackwardLaunches"`
	LastForwardCublasGemmCalls        int64           `json:"LastForwardCublasGemmCalls"`
	LastBackwardCublasGemmCalls       int64           `json:"LastBackwardCublasGemmCalls"`
	LastForwardSyncs                  int64           `json:"LastForwardSyncs"`
	LastBackwardSyncs                 int64           `json:"LastBackwardSyncs"`
	LastForwardDirectSubmissions      int64           `json:"LastForwardDirectSubmissions"`
	LastForwardDeviceKernelWork       int64           `json:"LastForwardDeviceKernelWork"`
	Telemetry                         json.RawMessage `json:"telemetry,omitempty"`
}

type legacyShapeExpectation struct {
	Shape               string
	Batch               int
	Tokens              int
	ModelDim            int
	FFNDim              int
	Heads               int
	HeadDim             int
	Layers              int
	OutputDim           int
	HasOutputProjection bool
	ForwardKernels      int64
	BackwardKernels     int64
	WholeKernels        int64
	ForwardUploadBytes  int64
	GradUploadBytes     int64
	WholeUploadBytes    int64
	DownloadedBytes     int64
	PooledDownloaded    int64
	StatusDownloaded    int64
}

func legacyShapeForProfile(profile string) (legacyShapeExpectation, error) {
	switch profile {
	case "canonical":
		return legacyShapeExpectation{
			Shape: "B=4,T=4,D=4,H=6,L=2,O=3,heads=2,head_dim=2,vocab=5,max_seq=8",
			Batch: 4, Tokens: 4, ModelDim: 4, FFNDim: 6, Heads: 2, HeadDim: 2,
			Layers: 2, OutputDim: 3, HasOutputProjection: true,
			ForwardKernels: 24, BackwardKernels: 44, WholeKernels: 68,
			ForwardUploadBytes: 148, GradUploadBytes: 48, WholeUploadBytes: 196,
			DownloadedBytes: 68, PooledDownloaded: 48, StatusDownloaded: 20,
		}, nil
	case "next":
		return legacyShapeExpectation{
			Shape: "B=1,T=256,D=128,H=256,L=2,O=128,heads=4,head_dim=32,vocab=16384,max_seq=256",
			Batch: 1, Tokens: 256, ModelDim: 128, FFNDim: 256, Heads: 4, HeadDim: 32,
			Layers: 2, OutputDim: 128, HasOutputProjection: false,
			ForwardKernels: 22, BackwardKernels: 43, WholeKernels: 65,
			ForwardUploadBytes: 2056, GradUploadBytes: 512, WholeUploadBytes: 2568,
			DownloadedBytes: 520, PooledDownloaded: 512, StatusDownloaded: 8,
		}, nil
	default:
		return legacyShapeExpectation{}, fmt.Errorf("legacy_stats profile %q is not prevalidated", profile)
	}
}

// expectedLegacyStatsWire builds the fixed counter portion used by the
// command's adversarial tests. Runtime results are still accepted only through
// decodeLegacyStats plus validateLegacyStats; this helper is not a parser or a
// source of authority for child output.
func expectedLegacyStatsWire(expected sampleSpec, steps int) (legacyStatsWire, error) {
	var wire legacyStatsWire
	shape, err := legacyShapeForProfile(expected.Cell.Profile)
	if err != nil {
		return wire, err
	}
	if steps < 1 {
		return wire, errors.New("expected legacy stats requires positive measured steps")
	}
	perStep := func(value int64) int64 { return value * int64(steps) }
	wire.ForwardCalls = perStep(1)
	wire.BackwardCalls = perStep(1)
	wire.HandlesCreated = perStep(1)
	wire.HandlesReleased = perStep(1)
	wire.ArenaReuseHits = perStep(1)
	wire.GradientReuseHits = perStep(1)
	wire.GradientZeroCalls = perStep(1)
	wire.UploadedBytes = perStep(shape.WholeUploadBytes)
	wire.GradPooledUploadedBytes = perStep(shape.GradUploadBytes)
	wire.DownloadedBytes = perStep(shape.DownloadedBytes)
	wire.PooledDownloadedBytes = perStep(shape.PooledDownloaded)
	wire.StatusDownloadedBytes = perStep(shape.StatusDownloaded)
	wire.ForwardReadbackBatchEntries = perStep(1)
	wire.ForwardReadbackContextSets = perStep(1)
	wire.ForwardReadbackDeviceCopies = perStep(3)
	wire.KernelLaunches = perStep(shape.WholeKernels)
	if expected.Cell.Graph == 1 {
		wire.KernelLaunches = perStep(shape.BackwardKernels)
	}
	wire.KernelSynchronizations = perStep(2)
	wire.GraphReplays = perStep(int64(expected.Cell.Graph))
	wire.GraphLaunches = perStep(int64(expected.Cell.Graph))
	wire.GraphExecutedNodes = perStep(int64(expected.Cell.Graph) * shape.ForwardKernels)
	wire.DirectForwardSubmissions = perStep(int64(1-expected.Cell.Graph) * shape.ForwardKernels)
	wire.ForwardDeviceKernelWork = perStep(shape.ForwardKernels)
	wire.ForwardInputUploadBatchCalls = perStep(int64(expected.Cell.Upload))
	wire.ForwardInputUploadContextSets = perStep(int64(expected.Cell.Upload))
	wire.ForwardInputUploadDeviceCopies = perStep(int64(expected.Cell.Upload) * 4)
	wire.LastForwardLaunches = int64(1-expected.Cell.Graph) * shape.ForwardKernels
	wire.LastBackwardLaunches = shape.BackwardKernels
	wire.LastForwardDirectSubmissions = wire.LastForwardLaunches
	wire.LastForwardDeviceKernelWork = shape.ForwardKernels
	wire.LastForwardSyncs = 1
	wire.LastBackwardSyncs = 1
	wire.LastShape = legacyShapeWire{
		Batch: shape.Batch, Tokens: shape.Tokens, ModelDim: shape.ModelDim, FFNDim: shape.FFNDim,
		Heads: shape.Heads, HeadDim: shape.HeadDim, Layers: shape.Layers, OutputDim: shape.OutputDim,
		HasOutputProjection: shape.HasOutputProjection,
	}
	return wire, nil
}

var legacyStatsFields = map[string]struct{}{
	"ForwardCalls": {}, "BackwardCalls": {}, "HandlesCreated": {}, "HandlesReleased": {},
	"LiveHandles": {}, "ArenaReuseHits": {}, "ArenaAllocations": {}, "GradientReuseHits": {},
	"GradientAllocations": {}, "GradientZeroCalls": {}, "ResidentGradBytes": {},
	"ActivationArenaBytes": {}, "WorkspaceArenaBytes": {}, "UploadedBytes": {},
	"DownloadedBytes": {}, "PooledDownloadedBytes": {}, "GradPooledUploadedBytes": {},
	"StatusDownloadedBytes": {}, "ForwardReadbackBatchEntries": {}, "ForwardReadbackContextSets": {},
	"ForwardReadbackDeviceCopies": {}, "ForwardInputUploadBatchCalls": {},
	"ForwardInputUploadContextSets": {}, "ForwardInputUploadDeviceCopies": {},
	"ForwardInputUploadFailures": {}, "ForwardInputUploadScalarFallbacks": {},
	"PackedBytesAvoided": {}, "HostGradUploadBytesAvoided": {}, "KernelLaunches": {},
	"CublasGemmCalls": {}, "KernelSynchronizations": {}, "GraphCaptures": {},
	"GraphReplays": {}, "GraphLaunches": {}, "GraphNodes": {}, "GraphCaptureFailures": {},
	"GraphReplayFailures": {}, "GraphInvalidations": {}, "GraphParityFailures": {},
	"GraphFallbacks": {}, "GraphSynchronizations": {}, "DirectForwardSubmissions": {},
	"GraphExecutedNodes": {}, "ForwardDeviceKernelWork": {}, "FallbackOrUnhandled": {},
	"ForwardNanos": {}, "BackwardNanos": {}, "OptimizerResidentGradNanos": {},
	"LastShape": {}, "LastForwardLaunches": {}, "LastBackwardLaunches": {},
	"LastForwardCublasGemmCalls": {}, "LastBackwardCublasGemmCalls": {},
	"LastForwardSyncs": {}, "LastBackwardSyncs": {}, "LastForwardDirectSubmissions": {},
	"LastForwardDeviceKernelWork": {}, "telemetry": {},
}

var legacyShapeFields = map[string]struct{}{
	"Batch": {}, "Tokens": {}, "ModelDim": {}, "FFNDim": {}, "Heads": {}, "HeadDim": {},
	"Layers": {}, "OutputDim": {}, "HasOutputProjection": {},
}

func rejectNullObjectFields(fields map[string]json.RawMessage) error {
	for name, raw := range fields {
		if string(bytesTrimSpace(raw)) == "null" {
			return fmt.Errorf("legacy_stats field %q is null", name)
		}
	}
	return nil
}

func decodeJSONObject(raw json.RawMessage, label string) (map[string]json.RawMessage, error) {
	trimmed := bytesTrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return nil, fmt.Errorf("%s is absent or null", label)
	}
	if trimmed[0] != '{' {
		return nil, fmt.Errorf("%s must be a JSON object", label)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &fields); err != nil {
		return nil, fmt.Errorf("decode %s object: %w", label, err)
	}
	if len(fields) == 0 {
		return nil, fmt.Errorf("%s must not be empty", label)
	}
	if err := rejectNullObjectFields(fields); err != nil {
		return nil, err
	}
	return fields, nil
}

func requireExactFields(fields map[string]json.RawMessage, expected map[string]struct{}, label string, optional map[string]struct{}) error {
	for name := range fields {
		if _, ok := expected[name]; !ok {
			return fmt.Errorf("%s contains unknown field %q", label, name)
		}
	}
	for name := range expected {
		if _, ok := optional[name]; ok {
			continue
		}
		if _, ok := fields[name]; !ok {
			return fmt.Errorf("%s is missing required field %q", label, name)
		}
	}
	return nil
}

func decodeLegacyStats(raw json.RawMessage, expected sampleSpec, steps int) (legacyStatsWire, error) {
	var wire legacyStatsWire
	fields, err := decodeJSONObject(raw, "legacy_stats")
	if err != nil {
		return wire, err
	}
	allowed := legacyStatsFields
	if expected.Events == 0 {
		allowed = make(map[string]struct{}, len(legacyStatsFields)-1)
		for name := range legacyStatsFields {
			if name != "telemetry" {
				allowed[name] = struct{}{}
			}
		}
	}
	if err := requireExactFields(fields, allowed, "legacy_stats", nil); err != nil {
		return wire, err
	}
	if err := decodeStrictJSON(raw, &wire); err != nil {
		return wire, fmt.Errorf("decode legacy_stats wire: %w", err)
	}
	shapeFields, err := decodeJSONObject(fields["LastShape"], "legacy_stats LastShape")
	if err != nil {
		return wire, err
	}
	if err := requireExactFields(shapeFields, legacyShapeFields, "legacy_stats LastShape", nil); err != nil {
		return wire, err
	}
	if err := decodeStrictJSON(fields["LastShape"], &wire.LastShape); err != nil {
		return wire, fmt.Errorf("decode legacy_stats LastShape: %w", err)
	}
	shape, err := legacyShapeForProfile(expected.Cell.Profile)
	if err != nil {
		return wire, err
	}
	if err := validateLegacyStats(wire, shape, expected, steps); err != nil {
		return wire, err
	}
	return wire, nil
}

func validateLegacyStats(wire legacyStatsWire, shape legacyShapeExpectation, expected sampleSpec, steps int) error {
	if steps < 1 {
		return errors.New("legacy_stats validation requires positive measured steps")
	}
	if expected.Cell.Graph != 0 && expected.Cell.Graph != 1 || expected.Cell.Upload != 0 && expected.Cell.Upload != 1 {
		return fmt.Errorf("legacy_stats received invalid graph/upload mode %d/%d", expected.Cell.Graph, expected.Cell.Upload)
	}
	perStep := func(value int64) int64 { return value * int64(steps) }
	expect := func(name string, got, want int64) error {
		if got != want {
			return fmt.Errorf("legacy_stats %s=%d, want %d", name, got, want)
		}
		return nil
	}
	checks := []struct {
		name string
		got  int64
		want int64
	}{
		{"ForwardCalls", wire.ForwardCalls, perStep(1)},
		{"BackwardCalls", wire.BackwardCalls, perStep(1)},
		{"HandlesCreated", wire.HandlesCreated, perStep(1)},
		{"HandlesReleased", wire.HandlesReleased, perStep(1)},
		{"ArenaReuseHits", wire.ArenaReuseHits, perStep(1)},
		{"ArenaAllocations", wire.ArenaAllocations, 0},
		{"GradientReuseHits", wire.GradientReuseHits, perStep(1)},
		{"GradientAllocations", wire.GradientAllocations, 0},
		{"GradientZeroCalls", wire.GradientZeroCalls, perStep(1)},
		{"UploadedBytes", wire.UploadedBytes, perStep(shape.WholeUploadBytes)},
		{"GradPooledUploadedBytes", wire.GradPooledUploadedBytes, perStep(shape.GradUploadBytes)},
		{"DownloadedBytes", wire.DownloadedBytes, perStep(shape.DownloadedBytes)},
		{"PooledDownloadedBytes", wire.PooledDownloadedBytes, perStep(shape.PooledDownloaded)},
		{"StatusDownloadedBytes", wire.StatusDownloadedBytes, perStep(shape.StatusDownloaded)},
		{"ForwardReadbackBatchEntries", wire.ForwardReadbackBatchEntries, perStep(1)},
		{"ForwardReadbackContextSets", wire.ForwardReadbackContextSets, perStep(1)},
		{"ForwardReadbackDeviceCopies", wire.ForwardReadbackDeviceCopies, perStep(3)},
		{"KernelLaunches", wire.KernelLaunches, perStep(func() int64 {
			if expected.Cell.Graph == 1 {
				return shape.BackwardKernels
			}
			return shape.WholeKernels
		}())},
		{"KernelSynchronizations", wire.KernelSynchronizations, perStep(2)},
		{"CublasGemmCalls", wire.CublasGemmCalls, 0},
		{"GraphCaptures", wire.GraphCaptures, 0},
		{"GraphReplays", wire.GraphReplays, perStep(int64(expected.Cell.Graph))},
		{"GraphLaunches", wire.GraphLaunches, perStep(int64(expected.Cell.Graph))},
		{"GraphNodes", wire.GraphNodes, 0},
		{"GraphCaptureFailures", wire.GraphCaptureFailures, 0},
		{"GraphReplayFailures", wire.GraphReplayFailures, 0},
		{"GraphInvalidations", wire.GraphInvalidations, 0},
		{"GraphParityFailures", wire.GraphParityFailures, 0},
		{"GraphFallbacks", wire.GraphFallbacks, 0},
		{"GraphSynchronizations", wire.GraphSynchronizations, 0},
		{"GraphExecutedNodes", wire.GraphExecutedNodes, perStep(int64(expected.Cell.Graph) * shape.ForwardKernels)},
		{"DirectForwardSubmissions", wire.DirectForwardSubmissions, perStep(int64(1-expected.Cell.Graph) * shape.ForwardKernels)},
		{"ForwardDeviceKernelWork", wire.ForwardDeviceKernelWork, perStep(shape.ForwardKernels)},
		{"FallbackOrUnhandled", wire.FallbackOrUnhandled, 0},
		{"ForwardInputUploadBatchCalls", wire.ForwardInputUploadBatchCalls, perStep(int64(expected.Cell.Upload))},
		{"ForwardInputUploadContextSets", wire.ForwardInputUploadContextSets, perStep(int64(expected.Cell.Upload))},
		{"ForwardInputUploadDeviceCopies", wire.ForwardInputUploadDeviceCopies, perStep(int64(expected.Cell.Upload) * 4)},
		{"ForwardInputUploadFailures", wire.ForwardInputUploadFailures, 0},
		{"ForwardInputUploadScalarFallbacks", wire.ForwardInputUploadScalarFallbacks, 0},
	}
	for _, check := range checks {
		if err := expect(check.name, check.got, check.want); err != nil {
			return err
		}
	}
	for name, value := range map[string]int64{
		"LiveHandles": wire.LiveHandles, "ResidentGradBytes": wire.ResidentGradBytes,
		"ActivationArenaBytes": wire.ActivationArenaBytes, "WorkspaceArenaBytes": wire.WorkspaceArenaBytes,
		"PackedBytesAvoided": wire.PackedBytesAvoided, "HostGradUploadBytesAvoided": wire.HostGradUploadBytesAvoided,
		"ForwardNanos": wire.ForwardNanos, "BackwardNanos": wire.BackwardNanos,
		"OptimizerResidentGradNanos": wire.OptimizerResidentGradNanos,
	} {
		if value < 0 {
			return fmt.Errorf("legacy_stats %s=%d is negative", name, value)
		}
	}
	if wire.LiveHandles != 0 {
		return fmt.Errorf("legacy_stats LiveHandles=%d, want 0", wire.LiveHandles)
	}
	lastForward := int64(1 - expected.Cell.Graph)
	lastChecks := []struct {
		name string
		got  int64
		want int64
	}{
		{"LastForwardLaunches", wire.LastForwardLaunches, lastForward * shape.ForwardKernels},
		{"LastBackwardLaunches", wire.LastBackwardLaunches, shape.BackwardKernels},
		{"LastForwardCublasGemmCalls", wire.LastForwardCublasGemmCalls, 0},
		{"LastBackwardCublasGemmCalls", wire.LastBackwardCublasGemmCalls, 0},
		{"LastForwardSyncs", wire.LastForwardSyncs, 1},
		{"LastBackwardSyncs", wire.LastBackwardSyncs, 1},
		{"LastForwardDirectSubmissions", wire.LastForwardDirectSubmissions, lastForward * shape.ForwardKernels},
		{"LastForwardDeviceKernelWork", wire.LastForwardDeviceKernelWork, shape.ForwardKernels},
	}
	for _, check := range lastChecks {
		if err := expect(check.name, check.got, check.want); err != nil {
			return err
		}
	}
	wantLastShape := legacyShapeWire{
		Batch: shape.Batch, Tokens: shape.Tokens, ModelDim: shape.ModelDim, FFNDim: shape.FFNDim,
		Heads: shape.Heads, HeadDim: shape.HeadDim, Layers: shape.Layers, OutputDim: shape.OutputDim,
		HasOutputProjection: shape.HasOutputProjection,
	}
	if wire.LastShape != wantLastShape {
		return fmt.Errorf("legacy_stats LastShape=%+v, want %+v", wire.LastShape, wantLastShape)
	}
	return nil
}

var knownTelemetryPhases = map[string]struct{}{
	"begin_zero": {}, "input_h2d": {}, "forward_direct": {}, "forward_capture": {},
	"forward_replay": {}, "forward_boundary": {}, "grad_h2d": {}, "backward_final": {},
	"backward_ffn": {}, "backward_attention": {}, "backward_boundary": {}, "k6_readback": {},
	"k5_batch": {}, "outer_optimizer": {},
}

var requiredOwnerPhases = map[string]struct{}{
	"begin_zero": {}, "input_h2d": {}, "forward_boundary": {}, "grad_h2d": {},
	"backward_final": {}, "backward_ffn": {}, "backward_attention": {},
	"backward_boundary": {}, "k6_readback": {}, "k5_batch": {},
}

var eventBearingPhases = map[string]struct{}{
	"forward_direct": {}, "forward_replay": {}, "backward_final": {},
	"backward_ffn": {}, "backward_attention": {}, "k5_batch": {},
}

func parseResultRecord(log string) (resultRecord, error) {
	var record resultRecord
	scanner := bufio.NewScanner(strings.NewReader(log))
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	count := 0
	var payload string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "k9_result_json=") {
			count++
			payload = strings.TrimPrefix(line, "k9_result_json=")
		}
	}
	if err := scanner.Err(); err != nil {
		return record, fmt.Errorf("scan benchmark output: %w", err)
	}
	if count != 1 {
		return record, fmt.Errorf("benchmark output has %d k9_result_json records, want exactly one", count)
	}
	if strings.TrimSpace(payload) == "" {
		return record, errors.New("k9_result_json record is empty")
	}
	if err := json.Unmarshal([]byte(payload), &record); err != nil {
		return record, fmt.Errorf("decode k9_result_json: %w", err)
	}
	return record, nil
}

func decodeStrictJSON(raw json.RawMessage, destination any) error {
	if len(bytesTrimSpace(raw)) == 0 || string(bytesTrimSpace(raw)) == "null" {
		return errors.New("JSON value is absent or null")
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("JSON value has trailing data")
		}
		return err
	}
	return nil
}

func bytesTrimSpace(raw []byte) []byte {
	return []byte(strings.TrimSpace(string(raw)))
}

func phaseWireValid(phase phaseWire) error {
	values := []int64{
		phase.GoCalls, phase.ContextSets, phase.DriverCalls, phase.KernelLaunches,
		phase.CublasCalls, phase.GraphBegin, phase.GraphEnd, phase.GraphInstantiate,
		phase.GraphLaunches, phase.GraphNodes, phase.GraphKernelNodes, phase.GraphCublasNodes,
		phase.GraphUnknownNodes, phase.H2DCopies, phase.H2DBytes, phase.D2HCopies,
		phase.D2HBytes, phase.MemsetCalls, phase.HostDescriptorAllocs, phase.StreamSynchronizes,
		phase.EventRecords, phase.EventQueries, phase.HostNanos, phase.DeviceElapsedNanos,
		phase.Attempted, phase.Enqueued, phase.Completed, phase.Failures, phase.FailIndex,
		phase.EventControlFailures,
	}
	for _, value := range values {
		if value < 0 {
			return fmt.Errorf("telemetry phase contains negative counter %d", value)
		}
	}
	if phase.Completed > phase.Attempted || phase.Completed > phase.Enqueued || phase.Enqueued > phase.Attempted {
		return fmt.Errorf("telemetry phase progress is inconsistent: attempted=%d enqueued=%d completed=%d", phase.Attempted, phase.Enqueued, phase.Completed)
	}
	if phase.Failures == 0 && (phase.HasFailIndex || phase.FailureStage != "") {
		return errors.New("telemetry phase has failure metadata without workload failures")
	}
	if phase.EventControlFailures == 0 && phase.EventControlStage != "" {
		return errors.New("telemetry phase has event-control metadata without failures")
	}
	return nil
}

func decodeTelemetry(raw json.RawMessage, expectedView string, owner bool, graph int) (*telemetryWire, error) {
	var telemetry telemetryWire
	if err := decodeStrictJSON(raw, &telemetry); err != nil {
		return nil, fmt.Errorf("decode %s telemetry: %w", expectedView, err)
	}
	if telemetry.SchemaVersion != resultSchemaVersion || !telemetry.Enabled || !telemetry.EventTiming || telemetry.View != expectedView {
		return nil, fmt.Errorf("telemetry metadata schema=%d enabled=%t events=%t view=%q, want v%d enabled/event_timing and view %q", telemetry.SchemaVersion, telemetry.Enabled, telemetry.EventTiming, telemetry.View, resultSchemaVersion, expectedView)
	}
	if len(telemetry.Phases) == 0 {
		return nil, fmt.Errorf("%s telemetry has no phases", expectedView)
	}
	for name, phase := range telemetry.Phases {
		if _, ok := knownTelemetryPhases[name]; !ok {
			return nil, fmt.Errorf("%s telemetry contains unknown phase %q", expectedView, name)
		}
		if err := phaseWireValid(phase); err != nil {
			return nil, fmt.Errorf("%s telemetry phase %s: %w", expectedView, name, err)
		}
		if phase.Failures != 0 || phase.EventControlFailures != 0 {
			return nil, fmt.Errorf("%s telemetry phase %s reports workload/event-control failures", expectedView, name)
		}
	}
	if !owner {
		if _, ok := telemetry.Phases["k5_batch"]; !ok {
			return nil, errors.New("optimizer telemetry is missing named k5_batch phase")
		}
		return &telemetry, nil
	}
	for name := range requiredOwnerPhases {
		if _, ok := telemetry.Phases[name]; !ok {
			return nil, fmt.Errorf("compact_train telemetry is missing required phase %q", name)
		}
	}
	forwardName := "forward_direct"
	inactiveForward := "forward_replay"
	if graph == 1 {
		forwardName, inactiveForward = "forward_replay", "forward_direct"
	}
	if _, ok := telemetry.Phases[forwardName]; !ok {
		return nil, fmt.Errorf("compact_train telemetry is missing selected %s phase", forwardName)
	}
	for _, inactive := range []string{"forward_capture", inactiveForward, "outer_optimizer"} {
		if _, present := telemetry.Phases[inactive]; present {
			return nil, fmt.Errorf("compact_train telemetry contains inactive phase %q", inactive)
		}
	}
	for name := range eventBearingPhases {
		phase, ok := telemetry.Phases[name]
		if !ok || (name != forwardName && name != "backward_final" && name != "backward_ffn" && name != "backward_attention" && name != "k5_batch") {
			continue
		}
		if phase.DeviceElapsedNanos <= 0 || phase.EventRecords <= 0 || phase.EventQueries <= 0 || phase.Failures != 0 || phase.EventControlFailures != 0 {
			return nil, fmt.Errorf("compact_train event-bearing phase %s lacks successful positive timing: %+v", name, phase)
		}
	}
	return &telemetry, nil
}

func (r resultRecord) validate(expected sampleSpec, steps int) error {
	if r.Schema != resultSchema || r.SchemaVersion != resultSchemaVersion {
		return fmt.Errorf("result schema=%q/%d, want %q/%d", r.Schema, r.SchemaVersion, resultSchema, resultSchemaVersion)
	}
	shape, shapeErr := legacyShapeForProfile(expected.Cell.Profile)
	if shapeErr != nil {
		return shapeErr
	}
	if r.Profile != expected.Cell.Profile || r.Graph != expected.Cell.Graph || r.Upload != expected.Cell.Upload || r.Events != expected.Events {
		return fmt.Errorf("result mode profile=%s graph=%d upload=%d events=%d, want %s/%d/%d/%d", r.Profile, r.Graph, r.Upload, r.Events, expected.Cell.Profile, expected.Cell.Graph, expected.Cell.Upload, expected.Events)
	}
	if r.Shape != shape.Shape {
		return fmt.Errorf("result shape=%q, want %q", r.Shape, shape.Shape)
	}
	if r.Modes.Graph != r.Graph || r.Modes.Upload != r.Upload || r.Modes.Events != r.Events || r.Modes.CUBLAS != 0 || r.Modes.SyncEach != 0 || r.Modes.ResidentTrain != 1 || r.Modes.PackedForward != 0 {
		return fmt.Errorf("result modes are not the fixed K9 controls: %+v", r.Modes)
	}
	wantWarmups := 1
	if r.Graph == 1 && r.Events == 1 {
		wantWarmups = 2
	}
	if r.WarmupSteps != wantWarmups {
		return fmt.Errorf("result warmup_steps=%d, want %d", r.WarmupSteps, wantWarmups)
	}
	if r.MeasuredSteps != steps || len(r.WallNanosPerStep) != steps || len(r.TopLevelHostNanos) != steps || len(r.DispatchHostNanos) != steps || len(r.ResidualFractions) != steps {
		return fmt.Errorf("result measured steps/arrays=%d/%d/%d/%d/%d, want %d", r.MeasuredSteps, len(r.WallNanosPerStep), len(r.TopLevelHostNanos), len(r.DispatchHostNanos), len(r.ResidualFractions), steps)
	}
	if r.Events == 1 && len(r.DeviceEventNanos) != steps {
		return fmt.Errorf("event-on device array length=%d, want %d", len(r.DeviceEventNanos), steps)
	}
	if r.Events == 0 && len(r.DeviceEventNanos) != 0 {
		return fmt.Errorf("event-off result invents %d device event observations", len(r.DeviceEventNanos))
	}
	legacy, err := decodeLegacyStats(r.LegacyStats, expected, steps)
	if err != nil {
		return err
	}
	for i, value := range r.WallNanosPerStep {
		if value <= 0 || r.TopLevelHostNanos[i] < 0 || r.DispatchHostNanos[i] < 0 || r.ResidualFractions[i] < 0 || !isFinite(r.ResidualFractions[i]) || r.ResidualFractions[i] > 0.05 {
			return fmt.Errorf("result step %d has invalid wall/host/dispatch/residual values", i)
		}
		if r.Events == 1 && r.DeviceEventNanos[i] <= 0 {
			return fmt.Errorf("event-on result step %d has non-positive device time", i)
		}
	}
	if r.Events == 0 {
		if len(r.TelemetryOwner) != 0 || len(r.TelemetryMirror) != 0 || len(r.TelemetryMirrorK5) != 0 || len(r.PhaseTelemetry) != 0 {
			return errors.New("event-off result contains telemetry JSON")
		}
		if r.TelemetryMirrorEqual {
			return errors.New("event-off result claims optimizer telemetry mirror equality")
		}
	} else if len(r.TelemetryOwner) == 0 || len(r.TelemetryMirror) == 0 || len(r.TelemetryMirrorK5) == 0 || len(r.PhaseTelemetry) == 0 {
		return errors.New("event-on result is missing owner/mirror/phase telemetry")
	} else {
		owner, err := decodeTelemetry(r.TelemetryOwner, "compact_train", true, r.Graph)
		if err != nil {
			return err
		}
		mirror, err := decodeTelemetry(r.TelemetryMirror, "optimizer", false, r.Graph)
		if err != nil {
			return err
		}
		var mirrorK5 phaseWire
		if err := decodeStrictJSON(r.TelemetryMirrorK5, &mirrorK5); err != nil {
			return fmt.Errorf("decode optimizer K5 mirror: %w", err)
		}
		if err := phaseWireValid(mirrorK5); err != nil {
			return fmt.Errorf("optimizer K5 mirror: %w", err)
		}
		ownerK5, ownerOK := owner.Phases["k5_batch"]
		mirrorK5Named, mirrorOK := mirror.Phases["k5_batch"]
		if !ownerOK || !mirrorOK || !reflect.DeepEqual(ownerK5, mirrorK5Named) || !reflect.DeepEqual(mirrorK5, mirrorK5Named) {
			return errors.New("optimizer K5 mirror does not equal owner and named mirror phase")
		}
		if reflect.DeepEqual(owner.Phases, mirror.Phases) {
			return errors.New("optimizer and compact_train telemetry whole views are unexpectedly equal")
		}
		var phaseMap map[string]phaseWire
		if err := decodeStrictJSON(r.PhaseTelemetry, &phaseMap); err != nil {
			return fmt.Errorf("decode phase telemetry: %w", err)
		}
		if !reflect.DeepEqual(phaseMap, owner.Phases) {
			return errors.New("phase_telemetry does not equal the typed compact_train owner view")
		}
		legacyOwner, err := decodeTelemetry(legacy.Telemetry, "compact_train", true, r.Graph)
		if err != nil {
			return fmt.Errorf("decode legacy_stats telemetry: %w", err)
		}
		if !reflect.DeepEqual(legacyOwner, owner) {
			return errors.New("legacy_stats telemetry does not equal telemetry_owner")
		}
		r.ownerWire, r.mirrorWire, r.phaseWire = owner, mirror, phaseMap
		if !r.TelemetryMirrorEqual {
			return errors.New("event-on result does not prove optimizer K5 mirror equality")
		}
	}
	for i, wall := range r.WallNanosPerStep {
		device := int64(0)
		if r.Events == 1 {
			device = r.DeviceEventNanos[i]
		}
		expectedDispatch := r.TopLevelHostNanos[i] - device
		if r.DispatchHostNanos[i] != expectedDispatch || expectedDispatch < 0 {
			return fmt.Errorf("step %d dispatch=%d, recomputed=%d", i, r.DispatchHostNanos[i], expectedDispatch)
		}
		expectedResidual := float64(absInt64(wall-r.TopLevelHostNanos[i])) / float64(wall)
		if math.Abs(r.ResidualFractions[i]-expectedResidual) > 1e-12 {
			return fmt.Errorf("step %d residual=%g, recomputed=%g", i, r.ResidualFractions[i], expectedResidual)
		}
	}
	return nil
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func absInt64(value int64) int64 {
	if value < 0 {
		return -value
	}
	return value
}

func conflictingEnvKey(key string) bool {
	upper := strings.ToUpper(key)
	return strings.HasPrefix(upper, "EOS_CUDA") ||
		strings.HasPrefix(upper, "EOS_TRAIN") ||
		strings.HasPrefix(upper, "EOS_RUN_COMPACT_RESIDENT_CUDA") ||
		strings.Contains(upper, "BENCH") ||
		upper == "GOFLAGS"
}

func isolatedEnvironment(base []string, gpu string, cell matrixCell, events, steps int) []string {
	values := make(map[string]string, len(base)+16)
	for _, item := range base {
		key, value, ok := strings.Cut(item, "=")
		if !ok || key == "" || conflictingEnvKey(key) {
			continue
		}
		values[key] = value
	}
	values["GOWORK"] = "off"
	values["CGO_ENABLED"] = "1"
	values["GOMAXPROCS"] = "1"
	values["CUDA_VISIBLE_DEVICES"] = gpu
	values["EOS_RUN_COMPACT_RESIDENT_CUDA_K9_MATRIX"] = "1"
	values["EOS_CUDA_WARM_BENCH_PROFILE"] = cell.Profile
	values["EOS_CUDA_COMPACT_PROFILE_EVENTS"] = strconv.Itoa(events)
	values["EOS_CUDA_COMPACT_TRAIN_FORWARD_GRAPH"] = strconv.Itoa(cell.Graph)
	values["EOS_CUDA_COMPACT_TRAIN_FORWARD_UPLOAD_BATCH"] = strconv.Itoa(cell.Upload)
	values["EOS_CUDA_COMPACT_TRAIN_CUBLAS"] = "0"
	values["EOS_CUDA_COMPACT_SYNC_EACH_LAUNCH"] = "0"
	values["EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN"] = "1"
	values["EOS_TRAIN_ENABLE_COMPACT_PACKED_FORWARD"] = "0"
	values["EOS_CUDA_K9_MEASURED_STEPS"] = strconv.Itoa(steps)
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		out = append(out, key+"="+values[key])
	}
	return out
}

// plaintextEnvironmentKeys is deliberately a positive allowlist. These are
// fixed K9 controls and process/reproducibility knobs whose values are not
// credentials. Every inherited value outside this set is represented by a
// one-way digest in evidence, including proxy, URL, URI, DSN, and arbitrary
// user-defined variables. The child still receives the original environment.
var plaintextEnvironmentKeys = map[string]struct{}{
	"GOWORK": {}, "CGO_ENABLED": {}, "GOMAXPROCS": {}, "CUDA_VISIBLE_DEVICES": {},
	"GOARCH": {}, "GOOS": {}, "GOAMD64": {}, "GOTOOLCHAIN": {},
	"EOS_RUN_COMPACT_RESIDENT_CUDA_K9_MATRIX": {},
	"EOS_CUDA_WARM_BENCH_PROFILE":             {}, "EOS_CUDA_COMPACT_PROFILE_EVENTS": {},
	"EOS_CUDA_COMPACT_TRAIN_FORWARD_GRAPH": {}, "EOS_CUDA_COMPACT_TRAIN_FORWARD_UPLOAD_BATCH": {},
	"EOS_CUDA_COMPACT_TRAIN_CUBLAS": {}, "EOS_CUDA_COMPACT_SYNC_EACH_LAUNCH": {},
	"EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN": {}, "EOS_TRAIN_ENABLE_COMPACT_PACKED_FORWARD": {},
	"EOS_CUDA_K9_MEASURED_STEPS": {},
}

func hashEnvironmentValue(value string) string {
	hash := sha256.Sum256([]byte(value))
	return fmt.Sprintf("<sha256:%x>", hash[:])
}

func redactEnvironment(env []string) []string {
	redacted := make([]string, 0, len(env))
	for _, item := range env {
		key, value, ok := strings.Cut(item, "=")
		if !ok || key == "" {
			// os.Environ always returns key=value, but do not preserve malformed
			// entries verbatim if a caller supplies one in a test or wrapper.
			redacted = append(redacted, "<malformed-environment-entry>="+hashEnvironmentValue(item))
			continue
		}
		if _, plaintext := plaintextEnvironmentKeys[key]; !plaintext {
			value = hashEnvironmentValue(value)
		}
		redacted = append(redacted, key+"="+value)
	}
	sort.Strings(redacted)
	return redacted
}

type processSpec struct {
	Binary          string
	Args            []string
	Env             []string
	CommandHash     string
	EnvironmentHash string
}

func buildProcessSpec(repo, gpu string, cell matrixCell, events, steps int, baseEnv []string, goBinary string) processSpec {
	if goBinary == "" {
		goBinary = "go"
	}
	args := []string{"test", "./runtime", "-run", "^$", "-bench", "^" + benchmarkName + "$", "-benchtime=1x", "-count=1", "-v"}
	env := isolatedEnvironment(baseEnv, gpu, cell, events, steps)
	return processSpec{
		Binary: goBinary, Args: args, Env: env,
		CommandHash: hashCommandInDir(repo, goBinary, args), EnvironmentHash: hashEnvironment(env),
	}
}

func hashCommand(binary string, args []string) string {
	hash := sha256.New()
	_, _ = io.WriteString(hash, binary)
	_, _ = hash.Write([]byte{0})
	for _, arg := range args {
		_, _ = io.WriteString(hash, arg)
		_, _ = hash.Write([]byte{0})
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func hashCommandInDir(repo, binary string, args []string) string {
	hash := sha256.New()
	_, _ = io.WriteString(hash, repo)
	_, _ = hash.Write([]byte{0})
	_, _ = io.WriteString(hash, hashCommand(binary, args))
	return hex.EncodeToString(hash.Sum(nil))
}

func hashEnvironment(env []string) string {
	sorted := append([]string(nil), env...)
	sort.Strings(sorted)
	hash := sha256.New()
	for _, item := range sorted {
		_, _ = io.WriteString(hash, item)
		_, _ = hash.Write([]byte{0})
	}
	return hex.EncodeToString(hash.Sum(nil))
}

type gpuSnapshot struct {
	Timestamp    time.Time `json:"timestamp"`
	Phase        string    `json:"phase"`
	GPUOutput    string    `json:"gpu_output,omitempty"`
	AppsOutput   string    `json:"apps_output,omitempty"`
	Loadavg      string    `json:"loadavg,omitempty"`
	GPUUtil      float64   `json:"gpu_util"`
	Load1        float64   `json:"load1"`
	ComputePIDs  []int     `json:"compute_pids,omitempty"`
	CaptureError string    `json:"capture_error,omitempty"`
}

func parseGPUUtil(output string) (float64, error) {
	reader := csv.NewReader(strings.NewReader(output))
	reader.TrimLeadingSpace = true
	for {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return 0, err
		}
		if len(record) < 4 {
			continue
		}
		value := strings.TrimSpace(record[3])
		parsed, err := strconv.ParseFloat(value, 64)
		if err == nil {
			return parsed, nil
		}
	}
	return 0, errors.New("nvidia-smi GPU output has no numeric utilization field")
}

func parseComputePIDs(output string) []int {
	var pids []int
	reader := csv.NewReader(strings.NewReader(output))
	reader.TrimLeadingSpace = true
	for {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil || len(record) == 0 {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(record[0]))
		if err == nil && pid > 0 {
			pids = append(pids, pid)
		}
	}
	return pids
}

func parseLoad1(loadavg string) (float64, error) {
	fields := strings.Fields(loadavg)
	if len(fields) == 0 {
		return 0, errors.New("/proc/loadavg is empty")
	}
	return strconv.ParseFloat(fields[0], 64)
}

func runExternal(ctx context.Context, name string, args ...string) (string, error) {
	command := exec.CommandContext(ctx, name, args...)
	output, err := command.Output()
	if err != nil {
		return string(output), err
	}
	return string(output), nil
}

func captureGPUSnapshot(ctx context.Context, gpu, phase string) (gpuSnapshot, error) {
	snapshot := gpuSnapshot{Timestamp: time.Now().UTC(), Phase: phase}
	queryCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	gpuOutput, err := runExternal(queryCtx, "nvidia-smi", "-i", gpu, "--query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total,clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits")
	if err != nil {
		snapshot.CaptureError = "gpu: " + err.Error()
		return snapshot, fmt.Errorf("capture nvidia-smi GPU state: %w", err)
	}
	appsOutput, err := runExternal(queryCtx, "nvidia-smi", "-i", gpu, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
	if err != nil {
		snapshot.CaptureError = "apps: " + err.Error()
		return snapshot, fmt.Errorf("capture nvidia-smi compute apps: %w", err)
	}
	loadBytes, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		snapshot.CaptureError = "loadavg: " + err.Error()
		return snapshot, fmt.Errorf("read /proc/loadavg: %w", err)
	}
	snapshot.GPUOutput, snapshot.AppsOutput, snapshot.Loadavg = strings.TrimSpace(gpuOutput), strings.TrimSpace(appsOutput), strings.TrimSpace(string(loadBytes))
	snapshot.GPUUtil, err = parseGPUUtil(gpuOutput)
	if err != nil {
		return snapshot, err
	}
	snapshot.Load1, err = parseLoad1(snapshot.Loadavg)
	if err != nil {
		return snapshot, err
	}
	snapshot.ComputePIDs = parseComputePIDs(appsOutput)
	return snapshot, nil
}

type quietThresholds struct {
	GPUUtil  float64
	Load1    float64
	LoadRise float64
}

type quietDecision struct {
	Eligible bool     `json:"eligible"`
	Reasons  []string `json:"reasons,omitempty"`
}

func classifyQuiet(pre, post gpuSnapshot, ownPID int, thresholds quietThresholds) quietDecision {
	return classifyQuietPIDs(pre, post, []int{ownPID}, thresholds)
}

func classifyQuietPIDs(pre, post gpuSnapshot, ownPIDs []int, thresholds quietThresholds) quietDecision {
	decision := quietDecision{Eligible: true}
	add := func(reason string) {
		decision.Eligible = false
		decision.Reasons = append(decision.Reasons, reason)
	}
	isOwn := func(pid int) bool {
		for _, ownPID := range ownPIDs {
			if pid == ownPID {
				return true
			}
		}
		return false
	}
	for _, pid := range pre.ComputePIDs {
		if !isOwn(pid) {
			add(fmt.Sprintf("pre-existing compute PID %d", pid))
		}
	}
	for _, pid := range post.ComputePIDs {
		if !isOwn(pid) {
			add(fmt.Sprintf("post-run compute PID %d", pid))
		}
	}
	if !isFinite(pre.GPUUtil) || pre.GPUUtil > thresholds.GPUUtil {
		add(fmt.Sprintf("pre GPU utilization %.3f > %.3f", pre.GPUUtil, thresholds.GPUUtil))
	}
	if !isFinite(post.GPUUtil) || post.GPUUtil > thresholds.GPUUtil {
		add(fmt.Sprintf("post GPU utilization %.3f > %.3f", post.GPUUtil, thresholds.GPUUtil))
	}
	if !isFinite(pre.Load1) || pre.Load1 > thresholds.Load1 {
		add(fmt.Sprintf("pre load1 %.3f > %.3f", pre.Load1, thresholds.Load1))
	}
	if !isFinite(post.Load1) || post.Load1 > thresholds.Load1 {
		add(fmt.Sprintf("post load1 %.3f > %.3f", post.Load1, thresholds.Load1))
	}
	if !isFinite(post.Load1-pre.Load1) || post.Load1-pre.Load1 > thresholds.LoadRise {
		add(fmt.Sprintf("load1 rise %.3f > %.3f", post.Load1-pre.Load1, thresholds.LoadRise))
	}
	return decision
}

func classifyDuringRunNoise(snapshot, pre gpuSnapshot, ownPIDs []int, thresholds quietThresholds) quietDecision {
	decision := quietDecision{Eligible: true}
	isOwn := func(pid int) bool {
		for _, ownPID := range ownPIDs {
			if pid == ownPID {
				return true
			}
		}
		return false
	}
	for _, pid := range snapshot.ComputePIDs {
		if !isOwn(pid) {
			decision.Eligible = false
			decision.Reasons = append(decision.Reasons, fmt.Sprintf("during-run unrelated compute PID %d", pid))
		}
	}
	// The child workload itself may legitimately drive utilization/load above
	// quiet thresholds. If nvidia-smi does not show either own PID, use the
	// normal threshold policy so a transient unrelated workload cannot hide.
	hasOwn := false
	for _, pid := range snapshot.ComputePIDs {
		if isOwn(pid) {
			hasOwn = true
			break
		}
	}
	if !hasOwn {
		decision = mergeQuietDecision(decision, classifyQuietPIDs(snapshot, snapshot, ownPIDs, thresholds))
	}
	return decision
}

func mergeQuietDecision(left, right quietDecision) quietDecision {
	if !right.Eligible {
		left.Eligible = false
		left.Reasons = append(left.Reasons, right.Reasons...)
	}
	return left
}

func processTreePIDs(root int) []int {
	if root <= 0 {
		return nil
	}
	pids := map[int]struct{}{root: {}}
	changed := true
	for changed {
		changed = false
		entries, err := os.ReadDir("/proc")
		if err != nil {
			break
		}
		for _, entry := range entries {
			pid, err := strconv.Atoi(entry.Name())
			if err != nil || pid <= 0 {
				continue
			}
			data, err := os.ReadFile(filepath.Join("/proc", entry.Name(), "stat"))
			if err != nil {
				continue
			}
			closeParen := strings.LastIndex(string(data), ")")
			if closeParen < 0 {
				continue
			}
			fields := strings.Fields(string(data[closeParen+1:]))
			if len(fields) < 2 {
				continue
			}
			parent, err := strconv.Atoi(fields[1])
			if err != nil {
				continue
			}
			if _, ok := pids[parent]; ok {
				if _, seen := pids[pid]; !seen {
					pids[pid] = struct{}{}
					changed = true
				}
			}
		}
	}
	result := make([]int, 0, len(pids))
	for pid := range pids {
		result = append(result, pid)
	}
	sort.Ints(result)
	return result
}

func waitForQuiet(ctx context.Context, gpu string, wait, poll time.Duration, thresholds quietThresholds, write func(gpuSnapshot)) (gpuSnapshot, error) {
	return waitForQuietWithCooldown(ctx, gpu, wait, poll, 2, thresholds, func(snapshot gpuSnapshot) error {
		if write != nil {
			write(snapshot)
		}
		return nil
	})
}

func waitForQuietWithCooldown(ctx context.Context, gpu string, wait, poll time.Duration, consecutive int, thresholds quietThresholds, write func(gpuSnapshot) error) (gpuSnapshot, error) {
	if wait <= 0 || poll <= 0 || consecutive < 1 {
		return gpuSnapshot{}, errors.New("quiet wait/poll durations and consecutive samples must be positive")
	}
	deadline := time.NewTimer(wait)
	defer deadline.Stop()
	ticker := time.NewTicker(poll)
	defer ticker.Stop()
	quietCount := 0
	for {
		snapshot, err := captureGPUSnapshot(ctx, gpu, "quiet_wait")
		if err != nil {
			return gpuSnapshot{}, fmt.Errorf("quiet identity capture failed: %w", err)
		}
		if write != nil {
			if err := write(snapshot); err != nil {
				return gpuSnapshot{}, fmt.Errorf("write quiet snapshot: %w", err)
			}
		}
		decision := classifyQuiet(snapshot, snapshot, os.Getpid(), thresholds)
		if decision.Eligible {
			quietCount++
			if quietCount >= consecutive {
				return snapshot, nil
			}
		} else {
			quietCount = 0
		}
		select {
		case <-ctx.Done():
			return gpuSnapshot{}, ctx.Err()
		case <-deadline.C:
			return gpuSnapshot{}, errors.New("quiet host wait expired; matrix blocked by host noise")
		case <-ticker.C:
		}
	}
}

type gpuLock struct {
	path  string
	token string
}

func gpuLockPath(gpu string) string {
	safe := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			return r
		}
		return '_'
	}, gpu)
	if safe == "" {
		safe = "default"
	}
	return filepath.Join(os.TempDir(), "eos-k9-cuda-gpu-"+safe+".lock")
}

func acquireGPULock(ctx context.Context, path string, wait time.Duration) (*gpuLock, error) {
	if wait <= 0 {
		return nil, errors.New("GPU lock wait must be positive")
	}
	token := fmt.Sprintf("pid=%d token=%d\n", os.Getpid(), time.Now().UnixNano())
	deadline := time.NewTimer(wait)
	defer deadline.Stop()
	for {
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err == nil {
			if _, writeErr := io.WriteString(file, token); writeErr != nil {
				_ = file.Close()
				_ = os.Remove(path)
				return nil, writeErr
			}
			if closeErr := file.Close(); closeErr != nil {
				_ = os.Remove(path)
				return nil, closeErr
			}
			return &gpuLock{path: path, token: token}, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, fmt.Errorf("create GPU lock %q: %w", path, err)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-deadline.C:
			return nil, fmt.Errorf("GPU lock %q is held (wait expired)", path)
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func (lock *gpuLock) Release() error {
	if lock == nil || lock.path == "" {
		return nil
	}
	content, err := os.ReadFile(lock.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if string(content) != lock.token {
		return errors.New("GPU lock ownership changed; refusing removal")
	}
	if err := os.Remove(lock.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

type sampleOutcome string

const (
	outcomeAccepted  sampleOutcome = "accepted"
	outcomeHostNoise sampleOutcome = "host_noise"
	outcomeProcess   sampleOutcome = "process_failure"
)

type attemptLedger struct {
	Timestamp        string        `json:"timestamp"`
	Spec             sampleSpec    `json:"spec"`
	Attempt          int           `json:"attempt"`
	Outcome          sampleOutcome `json:"outcome"`
	Reason           string        `json:"reason,omitempty"`
	PID              int           `json:"pid,omitempty"`
	ExitCode         *int          `json:"exit_code,omitempty"`
	Command          []string      `json:"command,omitempty"`
	Environment      []string      `json:"environment,omitempty"`
	CommandHash      string        `json:"command_hash"`
	EnvironmentHash  string        `json:"environment_hash"`
	BaselineIdentity runIdentity   `json:"baseline_identity"`
	PreIdentity      runIdentity   `json:"pre_identity"`
	PostIdentity     runIdentity   `json:"post_identity"`
	ResultPath       string        `json:"result_path,omitempty"`
	RawLogPath       string        `json:"raw_log_path,omitempty"`
	PreSnapshotPath  string        `json:"pre_snapshot_path,omitempty"`
	PostSnapshotPath string        `json:"post_snapshot_path,omitempty"`
}

type gpuIdentity struct {
	Index             string `json:"index"`
	Name              string `json:"name"`
	DriverVersion     string `json:"driver_version"`
	ComputeCapability string `json:"compute_capability"`
	UUID              string `json:"uuid"`
}

type runIdentity struct {
	UTC               string      `json:"utc"`
	PID               int         `json:"pid"`
	Repo              string      `json:"repo"`
	GPU               string      `json:"gpu"`
	GoBinary          string      `json:"go_binary"`
	GoVersion         string      `json:"go_version"`
	GitHEAD           string      `json:"git_head"`
	StagedDiffHash    string      `json:"staged_diff_hash"`
	UnstagedDiffHash  string      `json:"unstaged_diff_hash"`
	TrackedStatusHash string      `json:"tracked_status_hash"`
	OwnedSourceHash   string      `json:"owned_source_hash"`
	GPUIdentity       gpuIdentity `json:"gpu_identity"`
	CPUCount          int         `json:"cpu_count"`
}

func gitHEAD(repo string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	output, err := runExternalInDir(ctx, repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(output), nil
}

func runExternalInDir(ctx context.Context, dir, name string, args ...string) (string, error) {
	command := exec.CommandContext(ctx, name, args...)
	command.Dir = dir
	output, err := command.Output()
	return string(output), err
}

func hashString(value string) string {
	hash := sha256.Sum256([]byte(value))
	return hex.EncodeToString(hash[:])
}

func fileHash(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return hashString(string(data)), nil
}

func collectSourceIdentity(repo string) (string, string, string, string, string, error) {
	gitHead, err := gitHEAD(repo)
	if err != nil {
		return "", "", "", "", "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	staged, err := runExternalInDir(ctx, repo, "git", "diff", "--binary", "--cached", "--")
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("staged diff: %w", err)
	}
	unstaged, err := runExternalInDir(ctx, repo, "git", "diff", "--binary", "--")
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("unstaged diff: %w", err)
	}
	status, err := runExternalInDir(ctx, repo, "git", "status", "--porcelain=v1", "--untracked-files=no")
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("tracked status: %w", err)
	}
	owned := make([]string, 0, 3)
	for _, relative := range []string{
		"runtime/embedding_vector_distill_resident_cuda_benchmark_linux_test.go",
		"cmd/eos-k9-matrix/main.go", "cmd/eos-k9-matrix/main_test.go",
	} {
		path := filepath.Join(repo, relative)
		value, readErr := fileHash(path)
		if readErr != nil {
			return "", "", "", "", "", fmt.Errorf("owned source %s: %w", relative, readErr)
		}
		owned = append(owned, relative+"="+value)
	}
	return gitHead, hashString(staged), hashString(unstaged), hashString(status), hashString(strings.Join(owned, "\n")), nil
}

func parseGPUIdentity(output string) (gpuIdentity, error) {
	reader := csv.NewReader(strings.NewReader(output))
	reader.TrimLeadingSpace = true
	var identity gpuIdentity
	rows := 0
	for {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return identity, err
		}
		if len(record) == 0 || strings.TrimSpace(strings.Join(record, "")) == "" {
			continue
		}
		rows++
		if rows > 1 || len(record) < 5 {
			return gpuIdentity{}, errors.New("nvidia-smi GPU identity returned unexpected rows")
		}
		identity = gpuIdentity{Index: strings.TrimSpace(record[0]), Name: strings.TrimSpace(record[1]), DriverVersion: strings.TrimSpace(record[2]), ComputeCapability: strings.TrimSpace(record[3]), UUID: strings.TrimSpace(record[4])}
	}
	if rows != 1 || identity.UUID == "" || identity.Name == "" || identity.DriverVersion == "" || identity.ComputeCapability == "" {
		return gpuIdentity{}, errors.New("nvidia-smi GPU identity is incomplete")
	}
	return identity, nil
}

func collectRunIdentity(repo, gpu, goBinary string) (runIdentity, error) {
	if strings.TrimSpace(goBinary) == "" {
		goBinary = "go"
	}
	identity := runIdentity{UTC: time.Now().UTC().Format(time.RFC3339Nano), PID: os.Getpid(), Repo: repo, GPU: gpu, GoBinary: goBinary, CPUCount: runtime.NumCPU()}
	goCtx, goCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer goCancel()
	goOutput, err := runExternal(goCtx, goBinary, "version")
	if err != nil {
		return identity, fmt.Errorf("%s version: %w", goBinary, err)
	}
	identity.GoVersion = strings.TrimSpace(goOutput)
	identity.GitHEAD, identity.StagedDiffHash, identity.UnstagedDiffHash, identity.TrackedStatusHash, identity.OwnedSourceHash, err = collectSourceIdentity(repo)
	if err != nil {
		return identity, fmt.Errorf("source identity: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	gpuOutput, err := runExternal(ctx, "nvidia-smi", "-i", gpu, "--query-gpu=index,name,driver_version,compute_cap,uuid", "--format=csv,noheader,nounits")
	if err != nil {
		return identity, fmt.Errorf("GPU identity: %w", err)
	}
	identity.GPUIdentity, err = parseGPUIdentity(gpuOutput)
	if err != nil {
		return identity, fmt.Errorf("GPU identity parse: %w", err)
	}
	return identity, nil
}

func immutableIdentityEqual(left, right runIdentity) bool {
	return left.Repo == right.Repo && left.GPU == right.GPU && left.GoBinary == right.GoBinary && left.GoVersion == right.GoVersion && left.GitHEAD == right.GitHEAD && left.StagedDiffHash == right.StagedDiffHash && left.UnstagedDiffHash == right.UnstagedDiffHash && left.TrackedStatusHash == right.TrackedStatusHash && left.OwnedSourceHash == right.OwnedSourceHash && reflect.DeepEqual(left.GPUIdentity, right.GPUIdentity)
}

func writeJSONLine(path string, value any) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	data, err := json.Marshal(value)
	if err != nil {
		_ = file.Close()
		return err
	}
	if _, err = file.Write(append(data, '\n')); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}

type sampleRunner struct {
	config      runnerConfig
	thresholds  quietThresholds
	runDir      string
	ledgerPath  string
	samplesJSON string
	samplesTSV  string
	baseline    runIdentity
	stdout      io.Writer
}

func (runner *sampleRunner) writeSnapshot(snapshot gpuSnapshot) error {
	return writeJSONLine(filepath.Join(runner.runDir, "gpu_snapshots.jsonl"), snapshot)
}

func writeSnapshotFile(path string, snapshot gpuSnapshot) error {
	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func writeRawEvidence(path, content string) error {
	return os.WriteFile(path, []byte(content), 0o644)
}

func (runner *sampleRunner) writeSample(ledger attemptLedger) error {
	if err := writeJSONLine(runner.samplesJSON, ledger); err != nil {
		return fmt.Errorf("write sample JSONL: %w", err)
	}
	file, err := os.OpenFile(runner.samplesTSV, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("open samples TSV: %w", err)
	}
	writer := csv.NewWriter(file)
	writer.Comma = '\t'
	exitCode := ""
	if ledger.ExitCode != nil {
		exitCode = strconv.Itoa(*ledger.ExitCode)
	}
	row := []string{
		ledger.Timestamp, ledger.Spec.Cell.key(), strconv.Itoa(ledger.Spec.Pair), eventLabel(ledger.Spec.Events),
		strconv.Itoa(ledger.Spec.Order), strconv.Itoa(ledger.Attempt), string(ledger.Outcome), ledger.Reason,
		strconv.Itoa(ledger.PID), exitCode, ledger.CommandHash, ledger.EnvironmentHash, ledger.RawLogPath,
		ledger.PreSnapshotPath, ledger.PostSnapshotPath,
	}
	writer.Write(row)
	writer.Flush()
	writeErr := writer.Error()
	closeErr := file.Close()
	if writeErr != nil {
		return fmt.Errorf("write samples TSV: %w", writeErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close samples TSV: %w", closeErr)
	}
	return nil
}

func ensureRawEvidence(path, content string) error {
	_, err := os.Stat(path)
	if err == nil {
		return nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return writeRawEvidence(path, content)
}

func (runner *sampleRunner) runOne(ctx context.Context, spec sampleSpec, attempt int) (resultRecord, sampleOutcome, attemptLedger, error) {
	cellDir := filepath.Join(runner.runDir, spec.Cell.key(), fmt.Sprintf("pair-%02d", spec.Pair))
	if err := os.MkdirAll(cellDir, 0o755); err != nil {
		return resultRecord{}, outcomeProcess, attemptLedger{}, err
	}
	process := buildProcessSpec(runner.config.repo, runner.config.gpu, spec.Cell, spec.Events, runner.config.steps, os.Environ(), runner.config.goBinary)
	prefix := fmt.Sprintf("%s-attempt-%02d-%s", eventLabel(spec.Events), attempt, time.Now().UTC().Format("150405.000000000"))
	rawPath := filepath.Join(cellDir, prefix+".log")
	prePath := filepath.Join(cellDir, prefix+".pre.json")
	postPath := filepath.Join(cellDir, prefix+".post.json")
	ledger := attemptLedger{
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano), Spec: spec, Attempt: attempt,
		Command: append([]string{process.Binary}, process.Args...), Environment: redactEnvironment(process.Env),
		CommandHash: process.CommandHash, EnvironmentHash: process.EnvironmentHash,
		BaselineIdentity: runner.baseline, RawLogPath: rawPath, PreSnapshotPath: prePath, PostSnapshotPath: postPath,
	}
	fail := func(outcome sampleOutcome, reason string, runErr error) (resultRecord, sampleOutcome, attemptLedger, error) {
		ledger.Outcome, ledger.Reason = outcome, reason
		if err := ensureRawEvidence(rawPath, fmt.Sprintf("attempt outcome=%s reason=%s\n", outcome, reason)); err != nil {
			return resultRecord{}, outcomeProcess, ledger, fmt.Errorf("write rejected attempt evidence: %w", err)
		}
		return resultRecord{}, outcome, ledger, runErr
	}
	pre, err := waitForQuietWithCooldown(ctx, runner.config.gpu, runner.config.quietWait, runner.config.quietPoll, runner.config.quietConsecutive, runner.thresholds, runner.writeSnapshot)
	if err != nil {
		if strings.Contains(err.Error(), "matrix blocked by host noise") {
			return fail(outcomeHostNoise, err.Error(), nil)
		}
		return fail(outcomeProcess, err.Error(), err)
	}
	if err := writeSnapshotFile(prePath, pre); err != nil {
		return fail(outcomeProcess, fmt.Sprintf("write pre snapshot: %v", err), err)
	}
	preIdentity, err := collectRunIdentity(runner.config.repo, runner.config.gpu, process.Binary)
	ledger.PreIdentity = preIdentity
	if err != nil {
		return fail(outcomeProcess, fmt.Sprintf("pre-launch identity capture: %v", err), err)
	}
	if !immutableIdentityEqual(runner.baseline, preIdentity) {
		err := errors.New("pre-launch immutable identity changed from run baseline")
		return fail(outcomeProcess, err.Error(), err)
	}
	if decision := classifyQuietPIDs(pre, pre, []int{os.Getpid()}, runner.thresholds); !decision.Eligible {
		return fail(outcomeHostNoise, strings.Join(decision.Reasons, "; "), nil)
	}

	fmt.Fprintf(runner.stdout, "  launch command=%s %s command_hash=%s env_hash=%s\n", process.Binary, strings.Join(process.Args, " "), process.CommandHash, process.EnvironmentHash)
	commandCtx, cancel := context.WithTimeout(ctx, runner.config.processTimeout)
	defer cancel()
	command := exec.CommandContext(commandCtx, process.Binary, process.Args...)
	command.Dir = runner.config.repo
	command.Env = process.Env
	var output strings.Builder
	command.Stdout = &output
	command.Stderr = &output
	if err := command.Start(); err != nil {
		return fail(outcomeProcess, err.Error(), err)
	}
	ledger.PID = command.Process.Pid
	monitorCtx, monitorCancel := context.WithCancel(commandCtx)
	var monitorWG sync.WaitGroup
	monitorErrs := make(chan error, 1)
	monitorNoise := make(chan string, 1)
	childAndRunnerPIDs := []int{os.Getpid(), command.Process.Pid}
	monitorWG.Add(1)
	go func() {
		defer monitorWG.Done()
		captureDuring := func() {
			snapshot, captureErr := captureGPUSnapshot(monitorCtx, runner.config.gpu, "during")
			if captureErr != nil {
				snapshot.CaptureError = captureErr.Error()
				if monitorCtx.Err() == nil {
					select {
					case monitorErrs <- captureErr:
					default:
					}
				}
			} else {
				ownPIDs := append([]int{os.Getpid()}, processTreePIDs(command.Process.Pid)...)
				decision := classifyDuringRunNoise(snapshot, pre, ownPIDs, runner.thresholds)
				if !decision.Eligible && monitorCtx.Err() == nil {
					select {
					case monitorNoise <- strings.Join(decision.Reasons, "; "):
					default:
					}
				}
			}
			if writeErr := runner.writeSnapshot(snapshot); writeErr != nil {
				select {
				case monitorErrs <- writeErr:
				default:
				}
			}
		}
		captureDuring()
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-monitorCtx.Done():
				return
			case <-ticker.C:
				captureDuring()
			}
		}
	}()
	waitErr := command.Wait()
	monitorCancel()
	monitorWG.Wait()
	var monitorErr error
	select {
	case monitorErr = <-monitorErrs:
	default:
	}
	var monitorNoiseReason string
	select {
	case monitorNoiseReason = <-monitorNoise:
	default:
	}
	if command.ProcessState != nil {
		exitCode := command.ProcessState.ExitCode()
		ledger.ExitCode = &exitCode
	}
	if commandCtx.Err() != nil {
		waitErr = fmt.Errorf("benchmark process timeout: %w", commandCtx.Err())
	}
	if err := os.WriteFile(rawPath, []byte(output.String()), 0o644); err != nil {
		return fail(outcomeProcess, fmt.Sprintf("write raw process log: %v", err), err)
	}
	post, postErr := captureGPUSnapshot(ctx, runner.config.gpu, "post")
	if postErr == nil {
		if err := writeSnapshotFile(postPath, post); err != nil {
			return fail(outcomeProcess, fmt.Sprintf("write post snapshot: %v", err), err)
		}
	} else {
		return fail(outcomeProcess, postErr.Error(), postErr)
	}
	postIdentity, identityErr := collectRunIdentity(runner.config.repo, runner.config.gpu, process.Binary)
	ledger.PostIdentity = postIdentity
	if identityErr != nil {
		return fail(outcomeProcess, fmt.Sprintf("post-run identity capture: %v", identityErr), identityErr)
	}
	if !immutableIdentityEqual(runner.baseline, postIdentity) {
		err := errors.New("post-run immutable identity changed from run baseline")
		return fail(outcomeProcess, err.Error(), err)
	}
	if monitorErr != nil {
		return fail(outcomeProcess, fmt.Sprintf("during-process GPU snapshot/evidence failed: %v", monitorErr), monitorErr)
	}
	if waitErr != nil {
		return fail(outcomeProcess, waitErr.Error(), fmt.Errorf("benchmark process failed: %w", waitErr))
	}
	record, parseErr := parseResultRecord(output.String())
	if parseErr != nil {
		return fail(outcomeProcess, parseErr.Error(), parseErr)
	}
	if err := record.validate(spec, runner.config.steps); err != nil {
		return fail(outcomeProcess, err.Error(), err)
	}
	if decision := classifyQuietPIDs(pre, post, childAndRunnerPIDs, runner.thresholds); !decision.Eligible {
		return record, outcomeHostNoise, func() attemptLedger {
			ledger.Outcome, ledger.Reason = outcomeHostNoise, strings.Join(decision.Reasons, "; ")
			return ledger
		}(), nil
	}
	if monitorNoiseReason != "" {
		ledger.Outcome, ledger.Reason = outcomeHostNoise, monitorNoiseReason
		return record, outcomeHostNoise, ledger, nil
	}
	ledger.Outcome = outcomeAccepted
	ledger.ResultPath = rawPath
	return record, outcomeAccepted, ledger, nil
}

type pairedSample struct {
	Cell matrixCell
	Pair int
	Off  resultRecord
	On   resultRecord
}

type cellSummary struct {
	Cell                   matrixCell `json:"cell"`
	Pairs                  int        `json:"pairs"`
	LogMedianRatios        []float64  `json:"log_median_ratios"`
	GeometricMeanOverhead  float64    `json:"geometric_mean_overhead"`
	BootstrapUpperOverhead float64    `json:"bootstrap_upper_overhead"`
	ResidualP95            float64    `json:"residual_p95"`
	ResidualBootstrapUpper float64    `json:"residual_bootstrap_upper"`
	Accepted               bool       `json:"accepted"`
}

func median(values []float64) float64 {
	if len(values) == 0 {
		return math.NaN()
	}
	copyValues := append([]float64(nil), values...)
	sort.Float64s(copyValues)
	middle := len(copyValues) / 2
	if len(copyValues)%2 == 1 {
		return copyValues[middle]
	}
	return (copyValues[middle-1] + copyValues[middle]) / 2
}

func medianInt64(values []int64) float64 {
	converted := make([]float64, len(values))
	for i, value := range values {
		converted[i] = float64(value)
	}
	return median(converted)
}

func percentile(values []float64, q float64) float64 {
	if len(values) == 0 {
		return math.NaN()
	}
	if q <= 0 {
		q = 0
	}
	if q >= 1 {
		q = 1
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	index := int(math.Ceil(q*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func pairedLogMedianRatio(off, on resultRecord) (float64, error) {
	offMedian, onMedian := medianInt64(off.WallNanosPerStep), medianInt64(on.WallNanosPerStep)
	if !isFinite(offMedian) || !isFinite(onMedian) || offMedian <= 0 || onMedian <= 0 {
		return 0, errors.New("median wall time must be positive")
	}
	return math.Log(onMedian / offMedian), nil
}

func bootstrapMeanUpper(values []float64, draws int, seed int64, quantile float64) (float64, error) {
	if len(values) == 0 || draws < 1 || !isFinite(quantile) || quantile <= 0 || quantile >= 1 {
		return 0, errors.New("bootstrap requires values, positive draws, and quantile in (0,1)")
	}
	for _, value := range values {
		if !isFinite(value) {
			return 0, errors.New("bootstrap values must be finite")
		}
	}
	random := rand.New(rand.NewSource(seed))
	means := make([]float64, draws)
	for draw := range means {
		var sum float64
		for i := 0; i < len(values); i++ {
			sum += values[random.Intn(len(values))]
		}
		means[draw] = sum / float64(len(values))
	}
	return percentile(means, quantile), nil
}

func summarizeCell(cell matrixCell, samples []pairedSample, draws int, seed int64) (cellSummary, error) {
	if len(samples) == 0 {
		return cellSummary{}, errors.New("cannot summarize an empty cell")
	}
	logRatios := make([]float64, len(samples))
	residualMedians := make([]float64, 0, len(samples)*2)
	for i, sample := range samples {
		ratio, err := pairedLogMedianRatio(sample.Off, sample.On)
		if err != nil {
			return cellSummary{}, err
		}
		logRatios[i] = ratio
		residualMedians = append(residualMedians, median(sample.Off.ResidualFractions), median(sample.On.ResidualFractions))
	}
	upperLog, err := bootstrapMeanUpper(logRatios, draws, seed, 0.95)
	if err != nil {
		return cellSummary{}, err
	}
	upperResidual, err := bootstrapMeanUpper(residualMedians, draws, seed+1, 0.95)
	if err != nil {
		return cellSummary{}, err
	}
	estimate := math.Exp(mean(logRatios)) - 1
	upperOverhead := math.Exp(upperLog) - 1
	residualP95 := percentile(residualMedians, 0.95)
	return cellSummary{
		Cell: cell, Pairs: len(samples), LogMedianRatios: logRatios,
		GeometricMeanOverhead: estimate, BootstrapUpperOverhead: upperOverhead,
		ResidualP95: residualP95, ResidualBootstrapUpper: upperResidual,
		Accepted: estimate <= 0.02 && upperOverhead <= 0.02 && residualP95 <= 0.05 && upperResidual <= 0.05,
	}, nil
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return math.NaN()
	}
	var total float64
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func defaultOutputPath(now time.Time) string {
	return filepath.Join("runs", "eos-k9-cuda-telemetry-"+now.UTC().Format("20060102T150405Z"))
}

func parseFlags(args []string, now time.Time, stdout io.Writer) (runnerConfig, error) {
	flags := flag.NewFlagSet("eos-k9-matrix", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	config := runnerConfig{pairs: defaultPairs, steps: defaultMeasuredSteps, output: defaultOutputPath(now), quietWait: defaultQuietWait, quietPoll: defaultQuietPoll, quietConsecutive: 2, processTimeout: defaultProcessTimeout, lockWait: defaultLockWait, maxReplacements: defaultMaxReplacements, quietGPUUtil: 5, quietLoad1: 1, quietLoadRise: 0.5, bootstrapDraws: defaultBootstrapDraws, seed: 1, goBinary: "go", stdout: stdout}
	flags.StringVar(&config.repo, "repo", ".", "EOS repository root")
	flags.StringVar(&config.gpu, "gpu", "0", "CUDA GPU index or UUID")
	flags.IntVar(&config.pairs, "pairs", config.pairs, "accepted event-off/on pairs per structural cell")
	flags.IntVar(&config.steps, "steps", config.steps, "internal measured steps per fresh benchmark process")
	flags.StringVar(&config.output, "output", config.output, "ignored run evidence directory")
	flags.DurationVar(&config.quietWait, "quiet-wait", config.quietWait, "bounded wait for quiet host")
	flags.DurationVar(&config.quietPoll, "quiet-poll", config.quietPoll, "quiet-host polling interval")
	flags.IntVar(&config.quietConsecutive, "quiet-consecutive", config.quietConsecutive, "eligible quiet snapshots required before launch")
	flags.DurationVar(&config.processTimeout, "process-timeout", config.processTimeout, "fresh go test timeout")
	flags.DurationVar(&config.lockWait, "lock-wait", config.lockWait, "exclusive GPU lock wait")
	flags.IntVar(&config.maxReplacements, "max-replacements", config.maxReplacements, "maximum host-noise replacements per sample")
	flags.Float64Var(&config.quietGPUUtil, "quiet-gpu-util", config.quietGPUUtil, "maximum pre/post GPU utilization")
	flags.Float64Var(&config.quietLoad1, "quiet-load1", config.quietLoad1, "maximum pre/post load1")
	flags.Float64Var(&config.quietLoadRise, "quiet-load-rise", config.quietLoadRise, "maximum load1 rise")
	flags.IntVar(&config.bootstrapDraws, "bootstrap-draws", config.bootstrapDraws, "paired bootstrap draws")
	flags.Int64Var(&config.seed, "seed", config.seed, "deterministic bootstrap seed")
	flags.BoolVar(&config.smoke, "smoke", false, "bounded canonical graph0/upload0 off/on smoke (steps=2)")
	flags.BoolVar(&config.dryRun, "dry-run", false, "print fresh-process plan without CUDA execution")
	flags.StringVar(&config.goBinary, "go-bin", config.goBinary, "Go executable")
	if err := flags.Parse(args); err != nil {
		return config, err
	}
	if strings.TrimSpace(config.repo) == "" || strings.TrimSpace(config.gpu) == "" {
		return config, errors.New("repo and gpu are required")
	}
	if config.pairs < 1 || config.steps < 1 || config.steps > 1000 || config.bootstrapDraws < 1 || config.quietWait <= 0 || config.quietPoll <= 0 || config.quietConsecutive < 1 || config.processTimeout <= 0 || config.lockWait <= 0 || config.maxReplacements < 0 || config.quietGPUUtil < 0 || config.quietLoad1 < 0 || config.quietLoadRise < 0 {
		return config, errors.New("invalid non-positive matrix/quiet/statistic flag")
	}
	if config.smoke {
		config.steps = 2
		config.pairs = 1
	}
	if config.stdout == nil {
		config.stdout = stdout
	}
	return config, nil
}

type runSummaryState struct {
	PlanProcesses    int
	AcceptedSamples  int
	RejectedAttempts int
	Replacements     int
	Summaries        []cellSummary
	AcceptedGate     bool
	StartedUTC       string
}

func (state *runSummaryState) write(path string, runErr error) error {
	status := "rejected"
	if runErr == nil && state.AcceptedGate {
		status = "accepted"
	}
	failure := ""
	if runErr != nil {
		failure = runErr.Error()
	}
	return writeJSONFile(path, map[string]any{
		"schema": resultSchema, "schema_version": resultSchemaVersion, "status": status,
		"failure": failure, "plan_processes": state.PlanProcesses,
		"accepted_samples": state.AcceptedSamples, "rejected_attempts": state.RejectedAttempts,
		"replacements": state.Replacements, "cells": state.Summaries,
		"accepted": status == "accepted", "no_speedup_claim": true,
		"started_utc": state.StartedUTC, "completed_utc": time.Now().UTC().Format(time.RFC3339Nano),
	})
}

func runMatrix(config runnerConfig) (runErr error) {
	repo, err := filepath.Abs(config.repo)
	if err != nil {
		return err
	}
	config.repo = repo
	plan := buildMatrixPlan(config.pairs, config.smoke)
	if len(plan) == 0 {
		return errors.New("matrix plan is empty")
	}
	if config.dryRun {
		for _, spec := range plan {
			process := buildProcessSpec(repo, config.gpu, spec.Cell, spec.Events, config.steps, os.Environ(), config.goBinary)
			fmt.Fprintf(config.stdout, "dry-run sample=%s events=%d order=%d command=%s %s command_hash=%s env_hash=%s\n", spec.key(), spec.Events, spec.Order, process.Binary, strings.Join(process.Args, " "), process.CommandHash, process.EnvironmentHash)
		}
		return nil
	}
	if err := os.MkdirAll(config.output, 0o755); err != nil {
		return err
	}
	runDir, err := filepath.Abs(config.output)
	if err != nil {
		return err
	}
	state := &runSummaryState{PlanProcesses: len(plan), Summaries: make([]cellSummary, 0), StartedUTC: time.Now().UTC().Format(time.RFC3339Nano)}
	defer func() {
		if summaryErr := state.write(filepath.Join(runDir, "summary.json"), runErr); summaryErr != nil {
			if runErr == nil {
				runErr = summaryErr
			} else {
				runErr = fmt.Errorf("%w; final rejection summary: %v", runErr, summaryErr)
			}
		}
	}()
	samplesTSV := filepath.Join(runDir, "samples.tsv")
	if err := os.WriteFile(samplesTSV, []byte("timestamp\tcell\tpair\tevents\torder\tattempt\toutcome\treason\tpid\texit_code\tcommand_hash\tenvironment_hash\traw_log_path\tpre_snapshot_path\tpost_snapshot_path\n"), 0o644); err != nil {
		return err
	}
	ledgerPath := filepath.Join(runDir, "attempts.jsonl")
	samplesJSON := filepath.Join(runDir, "samples.jsonl")
	for _, path := range []string{ledgerPath, samplesJSON} {
		if err := os.WriteFile(path, nil, 0o644); err != nil {
			return err
		}
	}
	if err := os.WriteFile(filepath.Join(runDir, "gpu_snapshots.jsonl"), nil, 0o644); err != nil {
		return err
	}
	runner := &sampleRunner{config: config, thresholds: quietThresholds{GPUUtil: config.quietGPUUtil, Load1: config.quietLoad1, LoadRise: config.quietLoadRise}, runDir: runDir, ledgerPath: ledgerPath, samplesJSON: samplesJSON, samplesTSV: samplesTSV, stdout: config.stdout}
	lockCtx, cancel := context.WithTimeout(context.Background(), config.lockWait)
	defer cancel()
	lock, err := acquireGPULock(lockCtx, gpuLockPath(config.gpu), config.lockWait)
	if err != nil {
		return err
	}
	defer func() {
		if releaseErr := lock.Release(); releaseErr != nil && runErr == nil {
			runErr = releaseErr
		}
	}()
	identity, err := collectRunIdentity(repo, config.gpu, config.goBinary)
	if err != nil {
		return err
	}
	runner.baseline = identity
	manifestConfig := map[string]any{
		"repo": config.repo, "gpu": config.gpu, "pairs": config.pairs, "steps": config.steps,
		"output": config.output, "quiet_wait": config.quietWait.String(), "quiet_poll": config.quietPoll.String(), "quiet_consecutive": config.quietConsecutive,
		"process_timeout": config.processTimeout.String(), "lock_wait": config.lockWait.String(),
		"max_replacements": config.maxReplacements, "quiet_gpu_util": config.quietGPUUtil,
		"quiet_load1": config.quietLoad1, "quiet_load_rise": config.quietLoadRise,
		"bootstrap_draws": config.bootstrapDraws, "seed": config.seed, "smoke": config.smoke,
	}
	manifest := map[string]any{"schema": resultSchema, "schema_version": resultSchemaVersion, "identity": identity, "config": manifestConfig, "plan_processes": len(plan), "started_utc": time.Now().UTC().Format(time.RFC3339Nano), "no_speedup_claim": true}
	if err := writeJSONFile(filepath.Join(runDir, "manifest.json"), manifest); err != nil {
		return err
	}
	acceptedPairs := make(map[string]map[int]*pairedSample)
	for index, spec := range plan {
		fmt.Fprintf(config.stdout, "[%d/%d] %s\n", index+1, len(plan), spec.key())
		var record resultRecord
		accepted := false
		for attempt := 1; attempt <= config.maxReplacements+1; attempt++ {
			processCtx, processCancel := context.WithCancel(context.Background())
			var outcome sampleOutcome
			var ledger attemptLedger
			var runErr error
			record, outcome, ledger, runErr = runner.runOne(processCtx, spec, attempt)
			processCancel()
			if err := writeJSONLine(runner.ledgerPath, ledger); err != nil {
				return err
			}
			if err := runner.writeSample(ledger); err != nil {
				return err
			}
			if outcome == outcomeAccepted {
				state.AcceptedSamples++
			} else {
				state.RejectedAttempts++
				if outcome == outcomeHostNoise && attempt <= config.maxReplacements {
					state.Replacements++
				}
			}
			if outcome == outcomeAccepted {
				accepted = true
				break
			}
			if outcome == outcomeProcess {
				return fmt.Errorf("%s process/identity failure: %w", spec.key(), runErr)
			}
			if attempt > config.maxReplacements {
				return fmt.Errorf("%s exceeded bounded host-noise replacements: %s", spec.key(), ledger.Reason)
			}
			fmt.Fprintf(config.stdout, "  host-noise replacement %d/%d: %s\n", attempt, config.maxReplacements, ledger.Reason)
		}
		if !accepted {
			return fmt.Errorf("%s did not produce an accepted sample", spec.key())
		}
		cellPairs := acceptedPairs[spec.Cell.key()]
		if cellPairs == nil {
			cellPairs = make(map[int]*pairedSample)
			acceptedPairs[spec.Cell.key()] = cellPairs
		}
		pair := cellPairs[spec.Pair]
		if pair == nil {
			pair = &pairedSample{Cell: spec.Cell, Pair: spec.Pair}
			cellPairs[spec.Pair] = pair
		}
		if spec.Events == 0 {
			pair.Off = record
		} else {
			pair.On = record
		}
	}
	summaries := make([]cellSummary, 0)
	allAccepted := true
	for _, cell := range matrixCells(config.smoke) {
		pairMap := acceptedPairs[cell.key()]
		pairs := make([]pairedSample, 0, len(pairMap))
		for pair := 1; pair <= config.pairs; pair++ {
			sample := pairMap[pair]
			if sample == nil || len(sample.Off.WallNanosPerStep) == 0 || len(sample.On.WallNanosPerStep) == 0 {
				return fmt.Errorf("cell %s pair %d is missing an accepted off/on sample", cell.key(), pair)
			}
			pairs = append(pairs, *sample)
		}
		summary, err := summarizeCell(cell, pairs, config.bootstrapDraws, config.seed+int64(len(summaries))*1009)
		if err != nil {
			return err
		}
		summaries = append(summaries, summary)
		allAccepted = allAccepted && summary.Accepted
	}
	state.Summaries = summaries
	state.AcceptedGate = allAccepted
	if !allAccepted {
		return errors.New("K9 telemetry/accounting matrix completed but one or more overhead/residual gates were rejected")
	}
	return nil
}

func matrixCells(smoke bool) []matrixCell {
	if smoke {
		return []matrixCell{{Profile: "canonical", Graph: 0, Upload: 0}}
	}
	cells := make([]matrixCell, 0, 8)
	for _, profile := range []string{"canonical", "next"} {
		for graph := 0; graph <= 1; graph++ {
			for upload := 0; upload <= 1; upload++ {
				cells = append(cells, matrixCell{Profile: profile, Graph: graph, Upload: upload})
			}
		}
	}
	return cells
}

func writeJSONFile(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func main() {
	config, err := parseFlags(os.Args[1:], time.Now(), os.Stdout)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := runMatrix(config); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
