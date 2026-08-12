package eosruntime

import "fmt"

const MultiVectorStoragePlanSchema = "manta.multivector_storage_plan.v1"

const (
	MultiVectorSidecarNone  = "none"
	MultiVectorSidecarFP16  = "fp16"
	MultiVectorSidecarDense = "dense"
)

type MultiVectorStoragePlanInput struct {
	Dim                       int
	BaselineDim               int
	Bits                      []int
	Objects                   int
	VectorsPerObject          []int
	SeriesLengths             []int
	WindowSize                int
	WindowStride              int
	SidecarStorage            string
	VectorOverheadBytes       int64
	PackedObjectOverheadBytes int64
}

type MultiVectorStoragePlan struct {
	Schema string                       `json:"schema"`
	Config MultiVectorStoragePlanConfig `json:"config"`
	Rows   []MultiVectorStoragePlanRow  `json:"rows"`
}

type MultiVectorStoragePlanConfig struct {
	Dim                       int    `json:"dim"`
	BaselineDim               int    `json:"baseline_dim"`
	Bits                      []int  `json:"bits"`
	Objects                   int    `json:"objects"`
	VectorsPerObject          []int  `json:"vectors_per_object"`
	SeriesLengths             []int  `json:"series_lengths,omitempty"`
	WindowSize                int    `json:"window_size,omitempty"`
	WindowStride              int    `json:"window_stride,omitempty"`
	SidecarStorage            string `json:"sidecar_storage"`
	VectorOverheadBytes       int64  `json:"vector_overhead_bytes"`
	PackedObjectOverheadBytes int64  `json:"packed_object_overhead_bytes"`
}

type MultiVectorStoragePlanRow struct {
	Dim                                    int     `json:"dim"`
	BaselineDim                            int     `json:"baseline_dim"`
	Bits                                   int     `json:"bits"`
	Objects                                int     `json:"objects"`
	VectorsPerObject                       int     `json:"vectors_per_object"`
	SeriesLength                           int     `json:"series_length,omitempty"`
	WindowSize                             int     `json:"window_size,omitempty"`
	WindowStride                           int     `json:"window_stride,omitempty"`
	DerivedWindowCount                     int     `json:"derived_window_count,omitempty"`
	DenseParentBytes                       int64   `json:"dense_parent_bytes"`
	DenseParentTotalBytes                  int64   `json:"dense_parent_total_bytes"`
	DenseBaselineBytes                     int64   `json:"dense_baseline_bytes"`
	DenseBaselineTotalBytes                int64   `json:"dense_baseline_total_bytes"`
	QuantizedPayloadBytes                  int64   `json:"quantized_payload_bytes"`
	SidecarStorage                         string  `json:"sidecar_storage"`
	SidecarBytesPerVector                  int64   `json:"sidecar_bytes_per_vector"`
	QuantizedVectorBytes                   int64   `json:"quantized_vector_bytes"`
	VectorOverheadBytes                    int64   `json:"vector_overhead_bytes"`
	DenseVectorStorageBytes                int64   `json:"dense_vector_storage_bytes"`
	QuantizedVectorStorageBytes            int64   `json:"quantized_vector_storage_bytes"`
	TotalQuantizedBytes                    int64   `json:"total_quantized_bytes"`
	PackedObjectOverheadBytes              int64   `json:"packed_object_overhead_bytes"`
	PackedQuantizedStorageBytes            int64   `json:"packed_quantized_storage_bytes"`
	PackedTotalQuantizedBytes              int64   `json:"packed_total_quantized_bytes"`
	DenseToQuantizedVectorRatio            float64 `json:"dense_to_quantized_vector_ratio"`
	TotalCompressionRatio                  float64 `json:"total_compression_ratio"`
	PackedTotalCompressionRatio            float64 `json:"packed_total_compression_ratio"`
	VectorsThatFitInOneDenseVector         int64   `json:"vectors_that_fit_in_one_dense_vector"`
	FitsInOneDenseVectorStorage            bool    `json:"fits_in_one_dense_vector_storage"`
	StorageMultipleOfDenseParentCost       float64 `json:"storage_multiple_of_dense_parent_cost"`
	PackedVectorsThatFitInOneDenseVector   int64   `json:"packed_vectors_that_fit_in_one_dense_vector"`
	PackedFitsInOneDenseVectorStorage      bool    `json:"packed_fits_in_one_dense_vector_storage"`
	PackedStorageMultipleOfDenseParentCost float64 `json:"packed_storage_multiple_of_dense_parent_cost"`
}

func PlanMultiVectorStorage(in MultiVectorStoragePlanInput) (MultiVectorStoragePlan, error) {
	if in.Dim <= 0 {
		return MultiVectorStoragePlan{}, fmt.Errorf("dim must be positive")
	}
	baselineDim := in.BaselineDim
	if baselineDim == 0 {
		baselineDim = in.Dim
	}
	if baselineDim < 0 {
		return MultiVectorStoragePlan{}, fmt.Errorf("baseline dim must be positive or zero to use dim")
	}
	if in.Objects <= 0 {
		return MultiVectorStoragePlan{}, fmt.Errorf("objects must be positive")
	}
	if in.VectorOverheadBytes < 0 {
		return MultiVectorStoragePlan{}, fmt.Errorf("vector overhead bytes must be non-negative")
	}
	if in.PackedObjectOverheadBytes < 0 {
		return MultiVectorStoragePlan{}, fmt.Errorf("packed object overhead bytes must be non-negative")
	}
	bits := normalizeTurboQuantRetrievalBits(in.Bits)
	if err := validateTurboQuantRetrievalBits(bits); err != nil {
		return MultiVectorStoragePlan{}, err
	}
	scenarios, vectorsPerObject, seriesLengths, windowSize, windowStride, err := multiVectorStorageScenarios(in)
	if err != nil {
		return MultiVectorStoragePlan{}, err
	}
	sidecarStorage, err := normalizeMultiVectorSidecarStorage(in.SidecarStorage)
	if err != nil {
		return MultiVectorStoragePlan{}, err
	}

	plan := MultiVectorStoragePlan{
		Schema: MultiVectorStoragePlanSchema,
		Config: MultiVectorStoragePlanConfig{
			Dim:                       in.Dim,
			BaselineDim:               baselineDim,
			Bits:                      append([]int(nil), bits...),
			Objects:                   in.Objects,
			VectorsPerObject:          append([]int(nil), vectorsPerObject...),
			SeriesLengths:             append([]int(nil), seriesLengths...),
			WindowSize:                windowSize,
			WindowStride:              windowStride,
			SidecarStorage:            sidecarStorage,
			VectorOverheadBytes:       in.VectorOverheadBytes,
			PackedObjectOverheadBytes: in.PackedObjectOverheadBytes,
		},
	}
	denseBaselineBytes := int64(baselineDim) * 4
	denseVectorStorageBytes := denseBaselineBytes + in.VectorOverheadBytes
	denseBaselineTotalBytes := denseVectorStorageBytes * int64(in.Objects)
	sidecarBytesPerVector := multiVectorSidecarBytes(in.Dim, sidecarStorage)
	for _, bitWidth := range bits {
		quantizedPayloadBytes := turboquantVectorBytes(in.Dim, bitWidth)
		quantizedVectorBytes := quantizedPayloadBytes + sidecarBytesPerVector
		quantizedVectorStorageBytes := quantizedVectorBytes + in.VectorOverheadBytes
		for _, scenario := range scenarios {
			count := scenario.VectorsPerObject
			totalQuantizedBytes := int64(in.Objects) * int64(count) * quantizedVectorStorageBytes
			packedQuantizedStorageBytes := in.PackedObjectOverheadBytes + int64(count)*quantizedVectorBytes
			packedTotalQuantizedBytes := int64(in.Objects) * packedQuantizedStorageBytes
			row := MultiVectorStoragePlanRow{
				Dim:                                    in.Dim,
				BaselineDim:                            baselineDim,
				Bits:                                   bitWidth,
				Objects:                                in.Objects,
				VectorsPerObject:                       count,
				SeriesLength:                           scenario.SeriesLength,
				WindowSize:                             scenario.WindowSize,
				WindowStride:                           scenario.WindowStride,
				DerivedWindowCount:                     scenario.DerivedWindowCount,
				DenseParentBytes:                       denseBaselineBytes,
				DenseParentTotalBytes:                  denseBaselineTotalBytes,
				DenseBaselineBytes:                     denseBaselineBytes,
				DenseBaselineTotalBytes:                denseBaselineTotalBytes,
				QuantizedPayloadBytes:                  quantizedPayloadBytes,
				SidecarStorage:                         sidecarStorage,
				SidecarBytesPerVector:                  sidecarBytesPerVector,
				QuantizedVectorBytes:                   quantizedVectorBytes,
				VectorOverheadBytes:                    in.VectorOverheadBytes,
				DenseVectorStorageBytes:                denseVectorStorageBytes,
				QuantizedVectorStorageBytes:            quantizedVectorStorageBytes,
				TotalQuantizedBytes:                    totalQuantizedBytes,
				PackedObjectOverheadBytes:              in.PackedObjectOverheadBytes,
				PackedQuantizedStorageBytes:            packedQuantizedStorageBytes,
				PackedTotalQuantizedBytes:              packedTotalQuantizedBytes,
				DenseToQuantizedVectorRatio:            ratioFloat64(float64(denseVectorStorageBytes), float64(quantizedVectorStorageBytes)),
				TotalCompressionRatio:                  ratioFloat64(float64(denseBaselineTotalBytes), float64(totalQuantizedBytes)),
				PackedTotalCompressionRatio:            ratioFloat64(float64(denseBaselineTotalBytes), float64(packedTotalQuantizedBytes)),
				VectorsThatFitInOneDenseVector:         denseVectorStorageBytes / quantizedVectorStorageBytes,
				FitsInOneDenseVectorStorage:            int64(count)*quantizedVectorStorageBytes <= denseVectorStorageBytes,
				StorageMultipleOfDenseParentCost:       ratioFloat64(float64(int64(count)*quantizedVectorStorageBytes), float64(denseVectorStorageBytes)),
				PackedVectorsThatFitInOneDenseVector:   packedVectorsThatFitInOneDenseVector(denseVectorStorageBytes, quantizedVectorBytes, in.PackedObjectOverheadBytes),
				PackedFitsInOneDenseVectorStorage:      packedQuantizedStorageBytes <= denseVectorStorageBytes,
				PackedStorageMultipleOfDenseParentCost: ratioFloat64(float64(packedQuantizedStorageBytes), float64(denseVectorStorageBytes)),
			}
			plan.Rows = append(plan.Rows, row)
		}
	}
	return plan, nil
}

type multiVectorStorageScenario struct {
	VectorsPerObject   int
	SeriesLength       int
	WindowSize         int
	WindowStride       int
	DerivedWindowCount int
}

func multiVectorStorageScenarios(in MultiVectorStoragePlanInput) ([]multiVectorStorageScenario, []int, []int, int, int, error) {
	if len(in.SeriesLengths) > 0 {
		if len(in.VectorsPerObject) > 0 {
			return nil, nil, nil, 0, 0, fmt.Errorf("use either series lengths with window settings or vectors per object, not both")
		}
		if in.WindowSize <= 0 {
			return nil, nil, nil, 0, 0, fmt.Errorf("window-size must be positive when series-lengths is set")
		}
		windowStride := in.WindowStride
		if windowStride == 0 {
			windowStride = in.WindowSize
		}
		if windowStride < 0 {
			return nil, nil, nil, 0, 0, fmt.Errorf("window-stride must be positive or zero to use window-size")
		}
		scenarios := make([]multiVectorStorageScenario, 0, len(in.SeriesLengths))
		vectorsPerObject := make([]int, 0, len(in.SeriesLengths))
		seriesLengths := make([]int, 0, len(in.SeriesLengths))
		for _, points := range in.SeriesLengths {
			windowCount, err := TimeSeriesWindowVectorCount(points, in.WindowSize, windowStride)
			if err != nil {
				return nil, nil, nil, 0, 0, err
			}
			scenarios = append(scenarios, multiVectorStorageScenario{
				VectorsPerObject:   windowCount,
				SeriesLength:       points,
				WindowSize:         in.WindowSize,
				WindowStride:       windowStride,
				DerivedWindowCount: windowCount,
			})
			vectorsPerObject = append(vectorsPerObject, windowCount)
			seriesLengths = append(seriesLengths, points)
		}
		return scenarios, vectorsPerObject, seriesLengths, in.WindowSize, windowStride, nil
	}

	vectorsPerObject := normalizeMultiVectorStorageCounts(in.VectorsPerObject)
	if len(vectorsPerObject) == 0 {
		if len(in.VectorsPerObject) > 0 {
			return nil, nil, nil, 0, 0, fmt.Errorf("vectors-per-object must contain at least one positive integer")
		}
		vectorsPerObject = []int{1}
	}
	scenarios := make([]multiVectorStorageScenario, 0, len(vectorsPerObject))
	for _, count := range vectorsPerObject {
		scenarios = append(scenarios, multiVectorStorageScenario{VectorsPerObject: count})
	}
	return scenarios, vectorsPerObject, nil, 0, 0, nil
}

func TimeSeriesWindowVectorCount(points, windowSize, stride int) (int, error) {
	if points <= 0 {
		return 0, fmt.Errorf("series length must be positive")
	}
	if windowSize <= 0 {
		return 0, fmt.Errorf("window size must be positive")
	}
	if stride <= 0 {
		return 0, fmt.Errorf("window stride must be positive")
	}
	if points <= windowSize {
		return 1, nil
	}
	return 1 + (points-windowSize+stride-1)/stride, nil
}

func normalizeMultiVectorStorageCounts(values []int) []int {
	seen := map[int]bool{}
	out := make([]int, 0, len(values))
	for _, value := range values {
		if value <= 0 || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func normalizeMultiVectorSidecarStorage(storage string) (string, error) {
	switch storage {
	case "", MultiVectorSidecarNone, "direct", "quantized", "quantized-only", TurboQuantRerankStorageCompactReconstruct, "compact", "reconstruct":
		return MultiVectorSidecarNone, nil
	case MultiVectorSidecarFP16, "f16", "half", "dense-f16":
		return MultiVectorSidecarFP16, nil
	case MultiVectorSidecarDense, "dense-f32", "fp32":
		return MultiVectorSidecarDense, nil
	default:
		return "", fmt.Errorf("unsupported sidecar storage %q; use %q, %q, or %q", storage, MultiVectorSidecarNone, MultiVectorSidecarFP16, MultiVectorSidecarDense)
	}
}

func multiVectorSidecarBytes(dim int, storage string) int64 {
	switch storage {
	case MultiVectorSidecarFP16:
		return int64(dim) * 2
	case MultiVectorSidecarDense:
		return int64(dim) * 4
	default:
		return 0
	}
}

func packedVectorsThatFitInOneDenseVector(denseVectorStorageBytes, quantizedVectorBytes, packedObjectOverheadBytes int64) int64 {
	if quantizedVectorBytes <= 0 || packedObjectOverheadBytes >= denseVectorStorageBytes {
		return 0
	}
	return (denseVectorStorageBytes - packedObjectOverheadBytes) / quantizedVectorBytes
}
