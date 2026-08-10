package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
	mll "m31labs.dev/mll"
)

// EmbeddingTrainProfileVersion identifies the serialized Eos training-profile schema.
const EmbeddingTrainProfileVersion = "manta/embed-train-profile/v0alpha1"

var tagXTPR = [4]byte{'X', 'T', 'P', 'R'}

// EmbeddingTrainProfile captures trainer-side residency and backend prep activity at a point in time.
type EmbeddingTrainProfile struct {
	Version               string                                  `json:"version"`
	Step                  int                                     `json:"step"`
	ForwardBackend        eosartifact.BackendKind                 `json:"forward_backend,omitempty"`
	OptimizerBackend      eosartifact.BackendKind                 `json:"optimizer_backend,omitempty"`
	ActivationBackend     eosartifact.BackendKind                 `json:"activation_backend,omitempty"`
	ContrastiveBackend    eosartifact.BackendKind                 `json:"contrastive_backend,omitempty"`
	CompactForwardBackend eosartifact.BackendKind                 `json:"compact_forward_backend,omitempty"`
	CompactTrainBackend   eosartifact.BackendKind                 `json:"compact_train_backend,omitempty"`
	ForwardResidency      EmbeddingForwardResidencyStats          `json:"forward_residency"`
	CompactForwardTrainer embeddingCompactForwardTrainerStats     `json:"compact_forward_trainer"`
	CompactForward        *backend.CompactForwardAcceleratorStats `json:"compact_forward,omitempty"`
	CompactTrain          *backend.CompactTrainAcceleratorStats   `json:"compact_train,omitempty"`
	VectorDistillPhases   EmbeddingVectorDistillPhaseTimers       `json:"vector_distill_phases,omitempty"`
	Optimizer             backend.OptimizerAcceleratorStats       `json:"optimizer"`
	Activation            backend.ActivationAcceleratorStats      `json:"activation"`
	Contrastive           backend.ContrastiveAcceleratorStats     `json:"contrastive"`
}

// EmbeddingVectorDistillPhaseTimers captures host-observed compact vector-distill phase time.
type EmbeddingVectorDistillPhaseTimers struct {
	EncodeNanos         int64 `json:"encode_nanos,omitempty"`
	ProjectionLossNanos int64 `json:"projection_loss_nanos,omitempty"`
	BackwardNanos       int64 `json:"backward_nanos,omitempty"`
	OptimizerNanos      int64 `json:"optimizer_nanos,omitempty"`
}

// DefaultEmbeddingTrainProfilePath returns the conventional sibling training-profile path for a .mll artifact.
func DefaultEmbeddingTrainProfilePath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".train-profile.mll")
}

// Validate checks the serialized training-profile contract.
func (p EmbeddingTrainProfile) Validate() error {
	if p.Version == "" {
		return fmt.Errorf("training profile version is required")
	}
	if p.Version != EmbeddingTrainProfileVersion {
		return fmt.Errorf("training profile version %q is not supported, want %q", p.Version, EmbeddingTrainProfileVersion)
	}
	if p.Step < 0 {
		return fmt.Errorf("training profile step must be non-negative")
	}
	return nil
}

// WriteFile writes the training profile as an MLL sealed container.
func (p EmbeddingTrainProfile) WriteFile(path string) error {
	if err := p.Validate(); err != nil {
		return err
	}
	data, err := encodeEmbeddingTrainProfileMLL(p)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// ReadEmbeddingTrainProfileFile reads an MLL training profile.
func ReadEmbeddingTrainProfileFile(path string) (EmbeddingTrainProfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return EmbeddingTrainProfile{}, err
	}
	if !eosartifact.IsMLLBytes(data) {
		return EmbeddingTrainProfile{}, fmt.Errorf("training profile %q is not an MLL file", path)
	}
	return decodeEmbeddingTrainProfileMLL(data)
}

func encodeEmbeddingTrainProfileMLL(profile EmbeddingTrainProfile) ([]byte, error) {
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	head := mll.HeadSection{
		Name:        strg.Intern("manta-train-profile"),
		Description: strg.Intern("Eos training profile"),
		Generation:  uint64(profile.Step),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(strg, "profile_version", profile.Version),
			headStringMeta(strg, "forward_backend", string(profile.ForwardBackend)),
			headStringMeta(strg, "optimizer_backend", string(profile.OptimizerBackend)),
			headStringMeta(strg, "activation_backend", string(profile.ActivationBackend)),
			headStringMeta(strg, "contrastive_backend", string(profile.ContrastiveBackend)),
			headStringMeta(strg, "compact_forward_backend", string(profile.CompactForwardBackend)),
			headStringMeta(strg, "compact_train_backend", string(profile.CompactTrainBackend)),
			headIntMeta(strg, "step", int64(profile.Step)),
		},
	}
	body, err := json.Marshal(profile.normalized())
	if err != nil {
		return nil, err
	}

	var (
		dimsBuilder mll.DimsBuilder
		typeBuilder mll.TypeBuilder
		parmBuilder mll.ParmBuilder
		entrBuilder mll.EntrBuilder
		tnsrBuilder mll.TnsrBuilder
	)

	sections := make([]mll.SectionInput, 0, 9)
	if body, digestBody, err := encodeHeadSection(head, mll.ProfileSealed); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagHEAD,
			Body:          body,
			DigestBody:    digestBody,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(strg.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagSTRG,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(dimsBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagDIMS,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(typeBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTYPE, Body: body, SchemaVersion: 1})
	}
	if body, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagPARM,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(entrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagENTR,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagTNSR,
			Body:          body,
			Flags:         mll.SectionFlagRequired | mll.SectionFlagAligned,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection((mll.SchmSection{}).Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSCHM, Body: body, SchemaVersion: 1})
	}
	sections = append(sections, mll.SectionInput{
		Tag:           tagXTPR,
		Body:          body,
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})
	return mll.WriteToBytes(mll.ProfileSealed, mll.V1_0, sections)
}

func decodeEmbeddingTrainProfileMLL(data []byte) (EmbeddingTrainProfile, error) {
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return EmbeddingTrainProfile{}, err
	}
	if reader.Profile() != mll.ProfileSealed {
		return EmbeddingTrainProfile{}, fmt.Errorf("training profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}
	body, ok := reader.Section(tagXTPR)
	if !ok {
		return EmbeddingTrainProfile{}, fmt.Errorf("training profile missing XTPR section")
	}
	var profile EmbeddingTrainProfile
	if err := json.Unmarshal(body, &profile); err != nil {
		return EmbeddingTrainProfile{}, err
	}
	if err := profile.Validate(); err != nil {
		return EmbeddingTrainProfile{}, err
	}
	return profile, nil
}

func (p EmbeddingTrainProfile) normalized() EmbeddingTrainProfile {
	if p.Version == "" {
		p.Version = EmbeddingTrainProfileVersion
	}
	return p
}

func diffForwardResidencyStats(start, end EmbeddingForwardResidencyStats) EmbeddingForwardResidencyStats {
	return EmbeddingForwardResidencyStats{
		BindSkips: end.BindSkips - start.BindSkips,
		MatMul: backend.MatMulAcceleratorStats{
			BindCalls:          end.MatMul.BindCalls - start.MatMul.BindCalls,
			UploadedBytes:      end.MatMul.UploadedBytes - start.MatMul.UploadedBytes,
			QuantizePasses:     end.MatMul.QuantizePasses - start.MatMul.QuantizePasses,
			QuantizedBytes:     end.MatMul.QuantizedBytes - start.MatMul.QuantizedBytes,
			BindNanos:          end.MatMul.BindNanos - start.MatMul.BindNanos,
			QuantizeNanos:      end.MatMul.QuantizeNanos - start.MatMul.QuantizeNanos,
			BoundMatrices:      end.MatMul.BoundMatrices,
			RunCalls:           end.MatMul.RunCalls - start.MatMul.RunCalls,
			BoundLeftCalls:     end.MatMul.BoundLeftCalls - start.MatMul.BoundLeftCalls,
			BoundRightCalls:    end.MatMul.BoundRightCalls - start.MatMul.BoundRightCalls,
			RunUploadedBytes:   end.MatMul.RunUploadedBytes - start.MatMul.RunUploadedBytes,
			RunDownloadedBytes: end.MatMul.RunDownloadedBytes - start.MatMul.RunDownloadedBytes,
			RunNanos:           end.MatMul.RunNanos - start.MatMul.RunNanos,
		},
	}
}

func diffCompactForwardTrainerStats(start, end embeddingCompactForwardTrainerStats) embeddingCompactForwardTrainerStats {
	return embeddingCompactForwardTrainerStats{
		AttemptedCalls:      end.AttemptedCalls - start.AttemptedCalls,
		BucketCount:         end.BucketCount - start.BucketCount,
		FallbackOrUnhandled: end.FallbackOrUnhandled - start.FallbackOrUnhandled,
		PreflightFailures:   end.PreflightFailures - start.PreflightFailures,
		ResidentRefCount:    end.ResidentRefCount,
	}
}

func diffCompactForwardStats(start, end *backend.CompactForwardAcceleratorStats) *backend.CompactForwardAcceleratorStats {
	if end == nil {
		return nil
	}
	out := *end
	if start != nil {
		out.RunCalls -= start.RunCalls
		out.UnhandledCalls -= start.UnhandledCalls
		out.UploadedBytes -= start.UploadedBytes
		out.DownloadedBytes -= start.DownloadedBytes
		out.StatusDownloadedBytes -= start.StatusDownloadedBytes
		out.PackedDownloads -= start.PackedDownloads
		out.PackedBytes -= start.PackedBytes
		out.IntermediateD2H -= start.IntermediateD2H
		out.IntermediateDownloadedBytes -= start.IntermediateDownloadedBytes
		out.KernelLaunches -= start.KernelLaunches
		out.KernelSynchronizations -= start.KernelSynchronizations
		out.RunNanos -= start.RunNanos
	}
	return &out
}

func addCompactForwardStats(left, right *backend.CompactForwardAcceleratorStats) *backend.CompactForwardAcceleratorStats {
	if left == nil && right == nil {
		return nil
	}
	out := backend.CompactForwardAcceleratorStats{}
	if left != nil {
		out = *left
	}
	if right != nil {
		out.RunCalls += right.RunCalls
		out.UnhandledCalls += right.UnhandledCalls
		out.UploadedBytes += right.UploadedBytes
		out.DownloadedBytes += right.DownloadedBytes
		out.StatusDownloadedBytes += right.StatusDownloadedBytes
		out.PackedDownloads += right.PackedDownloads
		out.PackedBytes += right.PackedBytes
		out.IntermediateD2H += right.IntermediateD2H
		out.IntermediateDownloadedBytes += right.IntermediateDownloadedBytes
		out.KernelLaunches += right.KernelLaunches
		out.KernelSynchronizations += right.KernelSynchronizations
		out.RunNanos += right.RunNanos
		out.LastPackedFloats = right.LastPackedFloats
		out.LastPackedBytes = right.LastPackedBytes
		out.LastUploadBytes = right.LastUploadBytes
		out.LastDownloadBytes = right.LastDownloadBytes
		out.LastStatusDownloadedBytes = right.LastStatusDownloadedBytes
		out.LastKernelLaunches = right.LastKernelLaunches
		out.LastKernelSynchronizations = right.LastKernelSynchronizations
	}
	return &out
}

func diffCompactTrainStats(start, end *backend.CompactTrainAcceleratorStats) *backend.CompactTrainAcceleratorStats {
	if end == nil {
		return nil
	}
	out := *end
	if start != nil {
		out.ForwardCalls -= start.ForwardCalls
		out.BackwardCalls -= start.BackwardCalls
		out.HandlesCreated -= start.HandlesCreated
		out.HandlesReleased -= start.HandlesReleased
		out.GradientZeroCalls -= start.GradientZeroCalls
		out.ResidentGradBytes -= start.ResidentGradBytes
		out.ActivationArenaBytes -= start.ActivationArenaBytes
		out.WorkspaceArenaBytes -= start.WorkspaceArenaBytes
		out.UploadedBytes -= start.UploadedBytes
		out.DownloadedBytes -= start.DownloadedBytes
		out.PooledDownloadedBytes -= start.PooledDownloadedBytes
		out.GradPooledUploadedBytes -= start.GradPooledUploadedBytes
		out.StatusDownloadedBytes -= start.StatusDownloadedBytes
		out.PackedBytesAvoided -= start.PackedBytesAvoided
		out.HostGradUploadBytesAvoided -= start.HostGradUploadBytesAvoided
		out.KernelLaunches -= start.KernelLaunches
		out.CublasGemmCalls -= start.CublasGemmCalls
		out.KernelSynchronizations -= start.KernelSynchronizations
		out.GraphCaptures -= start.GraphCaptures
		out.GraphReplays -= start.GraphReplays
		out.FallbackOrUnhandled -= start.FallbackOrUnhandled
		out.ForwardNanos -= start.ForwardNanos
		out.BackwardNanos -= start.BackwardNanos
		out.OptimizerResidentGradNanos -= start.OptimizerResidentGradNanos
	}
	return &out
}

func addCompactTrainStats(left, right *backend.CompactTrainAcceleratorStats) *backend.CompactTrainAcceleratorStats {
	if left == nil && right == nil {
		return nil
	}
	out := backend.CompactTrainAcceleratorStats{}
	if left != nil {
		out = *left
	}
	if right != nil {
		out.ForwardCalls += right.ForwardCalls
		out.BackwardCalls += right.BackwardCalls
		out.HandlesCreated += right.HandlesCreated
		out.HandlesReleased += right.HandlesReleased
		out.LiveHandles = right.LiveHandles
		out.GradientZeroCalls += right.GradientZeroCalls
		out.ResidentGradBytes += right.ResidentGradBytes
		out.ActivationArenaBytes += right.ActivationArenaBytes
		out.WorkspaceArenaBytes += right.WorkspaceArenaBytes
		out.UploadedBytes += right.UploadedBytes
		out.DownloadedBytes += right.DownloadedBytes
		out.PooledDownloadedBytes += right.PooledDownloadedBytes
		out.GradPooledUploadedBytes += right.GradPooledUploadedBytes
		out.StatusDownloadedBytes += right.StatusDownloadedBytes
		out.PackedBytesAvoided += right.PackedBytesAvoided
		out.HostGradUploadBytesAvoided += right.HostGradUploadBytesAvoided
		out.KernelLaunches += right.KernelLaunches
		out.CublasGemmCalls += right.CublasGemmCalls
		out.KernelSynchronizations += right.KernelSynchronizations
		out.GraphCaptures += right.GraphCaptures
		out.GraphReplays += right.GraphReplays
		out.FallbackOrUnhandled += right.FallbackOrUnhandled
		out.ForwardNanos += right.ForwardNanos
		out.BackwardNanos += right.BackwardNanos
		out.OptimizerResidentGradNanos += right.OptimizerResidentGradNanos
		out.LastShape = right.LastShape
		out.LastForwardLaunches = right.LastForwardLaunches
		out.LastBackwardLaunches = right.LastBackwardLaunches
		out.LastForwardSyncs = right.LastForwardSyncs
		out.LastBackwardSyncs = right.LastBackwardSyncs
	}
	return &out
}

func diffVectorDistillPhaseTimers(start, end EmbeddingVectorDistillPhaseTimers) EmbeddingVectorDistillPhaseTimers {
	return EmbeddingVectorDistillPhaseTimers{
		EncodeNanos:         end.EncodeNanos - start.EncodeNanos,
		ProjectionLossNanos: end.ProjectionLossNanos - start.ProjectionLossNanos,
		BackwardNanos:       end.BackwardNanos - start.BackwardNanos,
		OptimizerNanos:      end.OptimizerNanos - start.OptimizerNanos,
	}
}

func hasTrainProfileActivity(profile EmbeddingTrainProfile) bool {
	return profile.Step != 0 ||
		profile.ForwardResidency.BindSkips != 0 ||
		profile.ForwardResidency.MatMul.BindCalls != 0 ||
		profile.ForwardResidency.MatMul.UploadedBytes != 0 ||
		profile.ForwardResidency.MatMul.QuantizePasses != 0 ||
		profile.ForwardResidency.MatMul.QuantizedBytes != 0 ||
		profile.ForwardResidency.MatMul.BindNanos != 0 ||
		profile.ForwardResidency.MatMul.QuantizeNanos != 0 ||
		profile.ForwardResidency.MatMul.RunCalls != 0 ||
		profile.ForwardResidency.MatMul.BoundLeftCalls != 0 ||
		profile.ForwardResidency.MatMul.BoundRightCalls != 0 ||
		profile.ForwardResidency.MatMul.RunUploadedBytes != 0 ||
		profile.ForwardResidency.MatMul.RunDownloadedBytes != 0 ||
		profile.ForwardResidency.MatMul.RunNanos != 0 ||
		profile.CompactForwardTrainer.AttemptedCalls != 0 ||
		profile.CompactForwardTrainer.BucketCount != 0 ||
		profile.CompactForwardTrainer.FallbackOrUnhandled != 0 ||
		profile.CompactForwardTrainer.PreflightFailures != 0 ||
		profile.CompactForwardTrainer.ResidentRefCount != 0 ||
		profile.CompactForward != nil ||
		profile.CompactTrain != nil ||
		profile.VectorDistillPhases.EncodeNanos != 0 ||
		profile.VectorDistillPhases.ProjectionLossNanos != 0 ||
		profile.VectorDistillPhases.BackwardNanos != 0 ||
		profile.VectorDistillPhases.OptimizerNanos != 0 ||
		profile.Optimizer.LogicalSteps != 0 ||
		profile.Optimizer.TensorUpdateCalls != 0 ||
		profile.Optimizer.UpdateCalls != 0 ||
		profile.Optimizer.DeferredSyncUpdates != 0 ||
		profile.Optimizer.SyncCalls != 0 ||
		profile.Optimizer.ForcedSyncCalls != 0 ||
		profile.Optimizer.LastForcedSyncReason != "" ||
		profile.Optimizer.UploadedBytes != 0 ||
		profile.Optimizer.DownloadedBytes != 0 ||
		profile.Optimizer.UpdateNanos != 0 ||
		profile.Optimizer.SyncNanos != 0 ||
		profile.Activation.BindCalls != 0 ||
		profile.Activation.GELUBackwardCalls != 0 ||
		profile.Activation.SoftmaxBackwardCalls != 0 ||
		profile.Activation.LayerNormBackwardCalls != 0 ||
		profile.Activation.UploadedBytes != 0 ||
		profile.Activation.DownloadedBytes != 0 ||
		profile.Activation.RunNanos != 0 ||
		profile.Contrastive.RunCalls != 0 ||
		profile.Contrastive.UploadedBytes != 0 ||
		profile.Contrastive.DownloadedBytes != 0 ||
		profile.Contrastive.RunNanos != 0
}

func addTrainProfileDelta(left, right EmbeddingTrainProfile) EmbeddingTrainProfile {
	hasRight := hasTrainProfileActivity(right)
	out := EmbeddingTrainProfile{
		Version:               EmbeddingTrainProfileVersion,
		Step:                  left.Step + right.Step,
		ForwardBackend:        left.ForwardBackend,
		OptimizerBackend:      left.OptimizerBackend,
		ActivationBackend:     left.ActivationBackend,
		ContrastiveBackend:    left.ContrastiveBackend,
		CompactForwardBackend: left.CompactForwardBackend,
		CompactTrainBackend:   left.CompactTrainBackend,
		ForwardResidency: EmbeddingForwardResidencyStats{
			BindSkips: left.ForwardResidency.BindSkips + right.ForwardResidency.BindSkips,
			MatMul: backend.MatMulAcceleratorStats{
				BindCalls:          left.ForwardResidency.MatMul.BindCalls + right.ForwardResidency.MatMul.BindCalls,
				UploadedBytes:      left.ForwardResidency.MatMul.UploadedBytes + right.ForwardResidency.MatMul.UploadedBytes,
				QuantizePasses:     left.ForwardResidency.MatMul.QuantizePasses + right.ForwardResidency.MatMul.QuantizePasses,
				QuantizedBytes:     left.ForwardResidency.MatMul.QuantizedBytes + right.ForwardResidency.MatMul.QuantizedBytes,
				BindNanos:          left.ForwardResidency.MatMul.BindNanos + right.ForwardResidency.MatMul.BindNanos,
				QuantizeNanos:      left.ForwardResidency.MatMul.QuantizeNanos + right.ForwardResidency.MatMul.QuantizeNanos,
				BoundMatrices:      left.ForwardResidency.MatMul.BoundMatrices,
				RunCalls:           left.ForwardResidency.MatMul.RunCalls + right.ForwardResidency.MatMul.RunCalls,
				BoundLeftCalls:     left.ForwardResidency.MatMul.BoundLeftCalls + right.ForwardResidency.MatMul.BoundLeftCalls,
				BoundRightCalls:    left.ForwardResidency.MatMul.BoundRightCalls + right.ForwardResidency.MatMul.BoundRightCalls,
				RunUploadedBytes:   left.ForwardResidency.MatMul.RunUploadedBytes + right.ForwardResidency.MatMul.RunUploadedBytes,
				RunDownloadedBytes: left.ForwardResidency.MatMul.RunDownloadedBytes + right.ForwardResidency.MatMul.RunDownloadedBytes,
				RunNanos:           left.ForwardResidency.MatMul.RunNanos + right.ForwardResidency.MatMul.RunNanos,
			},
		},
		CompactForwardTrainer: embeddingCompactForwardTrainerStats{
			AttemptedCalls:      left.CompactForwardTrainer.AttemptedCalls + right.CompactForwardTrainer.AttemptedCalls,
			BucketCount:         left.CompactForwardTrainer.BucketCount + right.CompactForwardTrainer.BucketCount,
			FallbackOrUnhandled: left.CompactForwardTrainer.FallbackOrUnhandled + right.CompactForwardTrainer.FallbackOrUnhandled,
			PreflightFailures:   left.CompactForwardTrainer.PreflightFailures + right.CompactForwardTrainer.PreflightFailures,
			ResidentRefCount:    left.CompactForwardTrainer.ResidentRefCount,
		},
		CompactForward: addCompactForwardStats(left.CompactForward, right.CompactForward),
		CompactTrain:   addCompactTrainStats(left.CompactTrain, right.CompactTrain),
		VectorDistillPhases: EmbeddingVectorDistillPhaseTimers{
			EncodeNanos:         left.VectorDistillPhases.EncodeNanos + right.VectorDistillPhases.EncodeNanos,
			ProjectionLossNanos: left.VectorDistillPhases.ProjectionLossNanos + right.VectorDistillPhases.ProjectionLossNanos,
			BackwardNanos:       left.VectorDistillPhases.BackwardNanos + right.VectorDistillPhases.BackwardNanos,
			OptimizerNanos:      left.VectorDistillPhases.OptimizerNanos + right.VectorDistillPhases.OptimizerNanos,
		},
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                   left.Optimizer.LogicalSteps + right.Optimizer.LogicalSteps,
			TensorUpdateCalls:              left.Optimizer.TensorUpdateCalls + right.Optimizer.TensorUpdateCalls,
			UpdateCalls:                    left.Optimizer.UpdateCalls + right.Optimizer.UpdateCalls,
			ResidentGradUpdateCalls:        left.Optimizer.ResidentGradUpdateCalls + right.Optimizer.ResidentGradUpdateCalls,
			DeferredSyncUpdates:            left.Optimizer.DeferredSyncUpdates + right.Optimizer.DeferredSyncUpdates,
			SyncCalls:                      left.Optimizer.SyncCalls + right.Optimizer.SyncCalls,
			ForcedSyncCalls:                left.Optimizer.ForcedSyncCalls + right.Optimizer.ForcedSyncCalls,
			UploadedBytes:                  left.Optimizer.UploadedBytes + right.Optimizer.UploadedBytes,
			ResidentGradUploadBytesAvoided: left.Optimizer.ResidentGradUploadBytesAvoided + right.Optimizer.ResidentGradUploadBytesAvoided,
			DownloadedBytes:                left.Optimizer.DownloadedBytes + right.Optimizer.DownloadedBytes,
			UpdateNanos:                    left.Optimizer.UpdateNanos + right.Optimizer.UpdateNanos,
			ResidentGradUpdateNanos:        left.Optimizer.ResidentGradUpdateNanos + right.Optimizer.ResidentGradUpdateNanos,
			SyncNanos:                      left.Optimizer.SyncNanos + right.Optimizer.SyncNanos,
			ResidentParams:                 left.Optimizer.ResidentParams,
		},
		Activation: backend.ActivationAcceleratorStats{
			BindCalls:              left.Activation.BindCalls + right.Activation.BindCalls,
			GELUBackwardCalls:      left.Activation.GELUBackwardCalls + right.Activation.GELUBackwardCalls,
			SoftmaxBackwardCalls:   left.Activation.SoftmaxBackwardCalls + right.Activation.SoftmaxBackwardCalls,
			LayerNormBackwardCalls: left.Activation.LayerNormBackwardCalls + right.Activation.LayerNormBackwardCalls,
			UploadedBytes:          left.Activation.UploadedBytes + right.Activation.UploadedBytes,
			DownloadedBytes:        left.Activation.DownloadedBytes + right.Activation.DownloadedBytes,
			RunNanos:               left.Activation.RunNanos + right.Activation.RunNanos,
			BoundTensors:           left.Activation.BoundTensors,
		},
		Contrastive: backend.ContrastiveAcceleratorStats{
			RunCalls:        left.Contrastive.RunCalls + right.Contrastive.RunCalls,
			UploadedBytes:   left.Contrastive.UploadedBytes + right.Contrastive.UploadedBytes,
			DownloadedBytes: left.Contrastive.DownloadedBytes + right.Contrastive.DownloadedBytes,
			RunNanos:        left.Contrastive.RunNanos + right.Contrastive.RunNanos,
		},
	}
	if out.ForwardBackend == "" {
		out.ForwardBackend = right.ForwardBackend
	}
	if out.OptimizerBackend == "" {
		out.OptimizerBackend = right.OptimizerBackend
	}
	if out.ActivationBackend == "" {
		out.ActivationBackend = right.ActivationBackend
	}
	if out.ContrastiveBackend == "" {
		out.ContrastiveBackend = right.ContrastiveBackend
	}
	if out.CompactForwardBackend == "" {
		out.CompactForwardBackend = right.CompactForwardBackend
	}
	if out.CompactTrainBackend == "" {
		out.CompactTrainBackend = right.CompactTrainBackend
	}
	if hasRight {
		out.ForwardResidency.MatMul.BoundMatrices = right.ForwardResidency.MatMul.BoundMatrices
		out.CompactForwardTrainer.ResidentRefCount = right.CompactForwardTrainer.ResidentRefCount
		if right.Optimizer.LastForcedSyncReason != "" {
			out.Optimizer.LastForcedSyncReason = right.Optimizer.LastForcedSyncReason
		} else {
			out.Optimizer.LastForcedSyncReason = left.Optimizer.LastForcedSyncReason
		}
		out.Optimizer.ResidentParams = right.Optimizer.ResidentParams
		out.Activation.BoundTensors = right.Activation.BoundTensors
	} else {
		out.Optimizer.LastForcedSyncReason = left.Optimizer.LastForcedSyncReason
	}
	return out
}

func applyTrainProfileDelta(base, delta EmbeddingTrainProfile) EmbeddingTrainProfile {
	out := EmbeddingTrainProfile{
		Version:               EmbeddingTrainProfileVersion,
		Step:                  base.Step + delta.Step,
		ForwardBackend:        base.ForwardBackend,
		OptimizerBackend:      base.OptimizerBackend,
		ActivationBackend:     base.ActivationBackend,
		ContrastiveBackend:    base.ContrastiveBackend,
		CompactForwardBackend: base.CompactForwardBackend,
		CompactTrainBackend:   base.CompactTrainBackend,
		ForwardResidency: EmbeddingForwardResidencyStats{
			BindSkips: base.ForwardResidency.BindSkips + delta.ForwardResidency.BindSkips,
			MatMul: backend.MatMulAcceleratorStats{
				BindCalls:          base.ForwardResidency.MatMul.BindCalls + delta.ForwardResidency.MatMul.BindCalls,
				UploadedBytes:      base.ForwardResidency.MatMul.UploadedBytes + delta.ForwardResidency.MatMul.UploadedBytes,
				QuantizePasses:     base.ForwardResidency.MatMul.QuantizePasses + delta.ForwardResidency.MatMul.QuantizePasses,
				QuantizedBytes:     base.ForwardResidency.MatMul.QuantizedBytes + delta.ForwardResidency.MatMul.QuantizedBytes,
				BindNanos:          base.ForwardResidency.MatMul.BindNanos + delta.ForwardResidency.MatMul.BindNanos,
				QuantizeNanos:      base.ForwardResidency.MatMul.QuantizeNanos + delta.ForwardResidency.MatMul.QuantizeNanos,
				BoundMatrices:      base.ForwardResidency.MatMul.BoundMatrices,
				RunCalls:           base.ForwardResidency.MatMul.RunCalls + delta.ForwardResidency.MatMul.RunCalls,
				BoundLeftCalls:     base.ForwardResidency.MatMul.BoundLeftCalls + delta.ForwardResidency.MatMul.BoundLeftCalls,
				BoundRightCalls:    base.ForwardResidency.MatMul.BoundRightCalls + delta.ForwardResidency.MatMul.BoundRightCalls,
				RunUploadedBytes:   base.ForwardResidency.MatMul.RunUploadedBytes + delta.ForwardResidency.MatMul.RunUploadedBytes,
				RunDownloadedBytes: base.ForwardResidency.MatMul.RunDownloadedBytes + delta.ForwardResidency.MatMul.RunDownloadedBytes,
				RunNanos:           base.ForwardResidency.MatMul.RunNanos + delta.ForwardResidency.MatMul.RunNanos,
			},
		},
		CompactForwardTrainer: embeddingCompactForwardTrainerStats{
			AttemptedCalls:      base.CompactForwardTrainer.AttemptedCalls + delta.CompactForwardTrainer.AttemptedCalls,
			BucketCount:         base.CompactForwardTrainer.BucketCount + delta.CompactForwardTrainer.BucketCount,
			FallbackOrUnhandled: base.CompactForwardTrainer.FallbackOrUnhandled + delta.CompactForwardTrainer.FallbackOrUnhandled,
			PreflightFailures:   base.CompactForwardTrainer.PreflightFailures + delta.CompactForwardTrainer.PreflightFailures,
			ResidentRefCount:    base.CompactForwardTrainer.ResidentRefCount,
		},
		CompactForward: addCompactForwardStats(base.CompactForward, delta.CompactForward),
		CompactTrain:   addCompactTrainStats(base.CompactTrain, delta.CompactTrain),
		VectorDistillPhases: EmbeddingVectorDistillPhaseTimers{
			EncodeNanos:         base.VectorDistillPhases.EncodeNanos + delta.VectorDistillPhases.EncodeNanos,
			ProjectionLossNanos: base.VectorDistillPhases.ProjectionLossNanos + delta.VectorDistillPhases.ProjectionLossNanos,
			BackwardNanos:       base.VectorDistillPhases.BackwardNanos + delta.VectorDistillPhases.BackwardNanos,
			OptimizerNanos:      base.VectorDistillPhases.OptimizerNanos + delta.VectorDistillPhases.OptimizerNanos,
		},
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                   base.Optimizer.LogicalSteps + delta.Optimizer.LogicalSteps,
			TensorUpdateCalls:              base.Optimizer.TensorUpdateCalls + delta.Optimizer.TensorUpdateCalls,
			UpdateCalls:                    base.Optimizer.UpdateCalls + delta.Optimizer.UpdateCalls,
			ResidentGradUpdateCalls:        base.Optimizer.ResidentGradUpdateCalls + delta.Optimizer.ResidentGradUpdateCalls,
			DeferredSyncUpdates:            base.Optimizer.DeferredSyncUpdates + delta.Optimizer.DeferredSyncUpdates,
			SyncCalls:                      base.Optimizer.SyncCalls + delta.Optimizer.SyncCalls,
			ForcedSyncCalls:                base.Optimizer.ForcedSyncCalls + delta.Optimizer.ForcedSyncCalls,
			UploadedBytes:                  base.Optimizer.UploadedBytes + delta.Optimizer.UploadedBytes,
			ResidentGradUploadBytesAvoided: base.Optimizer.ResidentGradUploadBytesAvoided + delta.Optimizer.ResidentGradUploadBytesAvoided,
			DownloadedBytes:                base.Optimizer.DownloadedBytes + delta.Optimizer.DownloadedBytes,
			UpdateNanos:                    base.Optimizer.UpdateNanos + delta.Optimizer.UpdateNanos,
			ResidentGradUpdateNanos:        base.Optimizer.ResidentGradUpdateNanos + delta.Optimizer.ResidentGradUpdateNanos,
			SyncNanos:                      base.Optimizer.SyncNanos + delta.Optimizer.SyncNanos,
			ResidentParams:                 base.Optimizer.ResidentParams,
		},
		Activation: backend.ActivationAcceleratorStats{
			BindCalls:              base.Activation.BindCalls + delta.Activation.BindCalls,
			GELUBackwardCalls:      base.Activation.GELUBackwardCalls + delta.Activation.GELUBackwardCalls,
			SoftmaxBackwardCalls:   base.Activation.SoftmaxBackwardCalls + delta.Activation.SoftmaxBackwardCalls,
			LayerNormBackwardCalls: base.Activation.LayerNormBackwardCalls + delta.Activation.LayerNormBackwardCalls,
			UploadedBytes:          base.Activation.UploadedBytes + delta.Activation.UploadedBytes,
			DownloadedBytes:        base.Activation.DownloadedBytes + delta.Activation.DownloadedBytes,
			RunNanos:               base.Activation.RunNanos + delta.Activation.RunNanos,
			BoundTensors:           base.Activation.BoundTensors,
		},
		Contrastive: backend.ContrastiveAcceleratorStats{
			RunCalls:        base.Contrastive.RunCalls + delta.Contrastive.RunCalls,
			UploadedBytes:   base.Contrastive.UploadedBytes + delta.Contrastive.UploadedBytes,
			DownloadedBytes: base.Contrastive.DownloadedBytes + delta.Contrastive.DownloadedBytes,
			RunNanos:        base.Contrastive.RunNanos + delta.Contrastive.RunNanos,
		},
	}
	if out.ForwardBackend == "" {
		out.ForwardBackend = delta.ForwardBackend
	}
	if out.OptimizerBackend == "" {
		out.OptimizerBackend = delta.OptimizerBackend
	}
	if out.ActivationBackend == "" {
		out.ActivationBackend = delta.ActivationBackend
	}
	if out.ContrastiveBackend == "" {
		out.ContrastiveBackend = delta.ContrastiveBackend
	}
	if out.CompactForwardBackend == "" {
		out.CompactForwardBackend = delta.CompactForwardBackend
	}
	if out.CompactTrainBackend == "" {
		out.CompactTrainBackend = delta.CompactTrainBackend
	}
	if hasTrainProfileActivity(delta) {
		out.ForwardResidency.MatMul.BoundMatrices = delta.ForwardResidency.MatMul.BoundMatrices
		out.CompactForwardTrainer.ResidentRefCount = delta.CompactForwardTrainer.ResidentRefCount
		if delta.Optimizer.LastForcedSyncReason != "" {
			out.Optimizer.LastForcedSyncReason = delta.Optimizer.LastForcedSyncReason
		} else {
			out.Optimizer.LastForcedSyncReason = base.Optimizer.LastForcedSyncReason
		}
		out.Optimizer.ResidentParams = delta.Optimizer.ResidentParams
		out.Activation.BoundTensors = delta.Activation.BoundTensors
	} else {
		out.Optimizer.LastForcedSyncReason = base.Optimizer.LastForcedSyncReason
	}
	return out
}

func diffTrainProfile(start, end EmbeddingTrainProfile) EmbeddingTrainProfile {
	return EmbeddingTrainProfile{
		Version:               EmbeddingTrainProfileVersion,
		Step:                  end.Step - start.Step,
		ForwardBackend:        end.ForwardBackend,
		OptimizerBackend:      end.OptimizerBackend,
		ActivationBackend:     end.ActivationBackend,
		ContrastiveBackend:    end.ContrastiveBackend,
		CompactForwardBackend: end.CompactForwardBackend,
		CompactTrainBackend:   end.CompactTrainBackend,
		ForwardResidency:      diffForwardResidencyStats(start.ForwardResidency, end.ForwardResidency),
		CompactForwardTrainer: diffCompactForwardTrainerStats(start.CompactForwardTrainer, end.CompactForwardTrainer),
		CompactForward:        diffCompactForwardStats(start.CompactForward, end.CompactForward),
		CompactTrain:          diffCompactTrainStats(start.CompactTrain, end.CompactTrain),
		VectorDistillPhases:   diffVectorDistillPhaseTimers(start.VectorDistillPhases, end.VectorDistillPhases),
		Optimizer: backend.OptimizerAcceleratorStats{
			LogicalSteps:                   end.Optimizer.LogicalSteps - start.Optimizer.LogicalSteps,
			TensorUpdateCalls:              end.Optimizer.TensorUpdateCalls - start.Optimizer.TensorUpdateCalls,
			UpdateCalls:                    end.Optimizer.UpdateCalls - start.Optimizer.UpdateCalls,
			ResidentGradUpdateCalls:        end.Optimizer.ResidentGradUpdateCalls - start.Optimizer.ResidentGradUpdateCalls,
			DeferredSyncUpdates:            end.Optimizer.DeferredSyncUpdates - start.Optimizer.DeferredSyncUpdates,
			SyncCalls:                      end.Optimizer.SyncCalls - start.Optimizer.SyncCalls,
			ForcedSyncCalls:                end.Optimizer.ForcedSyncCalls - start.Optimizer.ForcedSyncCalls,
			LastForcedSyncReason:           end.Optimizer.LastForcedSyncReason,
			UploadedBytes:                  end.Optimizer.UploadedBytes - start.Optimizer.UploadedBytes,
			ResidentGradUploadBytesAvoided: end.Optimizer.ResidentGradUploadBytesAvoided - start.Optimizer.ResidentGradUploadBytesAvoided,
			DownloadedBytes:                end.Optimizer.DownloadedBytes - start.Optimizer.DownloadedBytes,
			UpdateNanos:                    end.Optimizer.UpdateNanos - start.Optimizer.UpdateNanos,
			ResidentGradUpdateNanos:        end.Optimizer.ResidentGradUpdateNanos - start.Optimizer.ResidentGradUpdateNanos,
			SyncNanos:                      end.Optimizer.SyncNanos - start.Optimizer.SyncNanos,
			ResidentParams:                 end.Optimizer.ResidentParams,
		},
		Activation: backend.ActivationAcceleratorStats{
			BindCalls:              end.Activation.BindCalls - start.Activation.BindCalls,
			GELUBackwardCalls:      end.Activation.GELUBackwardCalls - start.Activation.GELUBackwardCalls,
			SoftmaxBackwardCalls:   end.Activation.SoftmaxBackwardCalls - start.Activation.SoftmaxBackwardCalls,
			LayerNormBackwardCalls: end.Activation.LayerNormBackwardCalls - start.Activation.LayerNormBackwardCalls,
			UploadedBytes:          end.Activation.UploadedBytes - start.Activation.UploadedBytes,
			DownloadedBytes:        end.Activation.DownloadedBytes - start.Activation.DownloadedBytes,
			RunNanos:               end.Activation.RunNanos - start.Activation.RunNanos,
			BoundTensors:           end.Activation.BoundTensors,
		},
		Contrastive: backend.ContrastiveAcceleratorStats{
			RunCalls:        end.Contrastive.RunCalls - start.Contrastive.RunCalls,
			UploadedBytes:   end.Contrastive.UploadedBytes - start.Contrastive.UploadedBytes,
			DownloadedBytes: end.Contrastive.DownloadedBytes - start.Contrastive.DownloadedBytes,
			RunNanos:        end.Contrastive.RunNanos - start.Contrastive.RunNanos,
		},
	}
}
