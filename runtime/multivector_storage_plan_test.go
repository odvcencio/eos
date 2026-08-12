package eosruntime

import (
	"strings"
	"testing"
)

func TestPlanMultiVectorStorageUsesTurboQuantIPPayloadBytes(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:              128,
		Bits:             []int{2, 4, 8},
		Objects:          1000,
		VectorsPerObject: []int{1, 16},
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if plan.Schema != MultiVectorStoragePlanSchema || len(plan.Rows) != 6 {
		t.Fatalf("unexpected plan identity: schema=%q rows=%d", plan.Schema, len(plan.Rows))
	}
	row := plan.Rows[0]
	if row.Bits != 2 || row.VectorsPerObject != 1 {
		t.Fatalf("unexpected first row: %+v", row)
	}
	if plan.Config.BaselineDim != 128 || row.BaselineDim != 128 {
		t.Fatalf("baseline dim = config:%d row:%d", plan.Config.BaselineDim, row.BaselineDim)
	}
	if row.DenseParentBytes != 512 || row.DenseParentTotalBytes != 512000 {
		t.Fatalf("dense bytes = parent:%d total:%d", row.DenseParentBytes, row.DenseParentTotalBytes)
	}
	if row.DenseBaselineBytes != row.DenseParentBytes || row.DenseBaselineTotalBytes != row.DenseParentTotalBytes {
		t.Fatalf("dense baseline aliases = bytes:%d total:%d", row.DenseBaselineBytes, row.DenseBaselineTotalBytes)
	}
	if row.QuantizedPayloadBytes != 40 || row.QuantizedVectorBytes != 40 {
		t.Fatalf("q2 bytes = payload:%d vector:%d", row.QuantizedPayloadBytes, row.QuantizedVectorBytes)
	}
	if row.VectorOverheadBytes != 0 || row.DenseVectorStorageBytes != 512 || row.QuantizedVectorStorageBytes != 40 {
		t.Fatalf("storage bytes = overhead:%d dense:%d quantized:%d", row.VectorOverheadBytes, row.DenseVectorStorageBytes, row.QuantizedVectorStorageBytes)
	}
	if row.PackedObjectOverheadBytes != 0 || row.PackedQuantizedStorageBytes != 40 || row.PackedTotalQuantizedBytes != 40000 {
		t.Fatalf("packed bytes = overhead:%d storage:%d total:%d", row.PackedObjectOverheadBytes, row.PackedQuantizedStorageBytes, row.PackedTotalQuantizedBytes)
	}
	if row.TotalQuantizedBytes != 40000 {
		t.Fatalf("total quantized bytes = %d", row.TotalQuantizedBytes)
	}
	if row.VectorsThatFitInOneDenseVector != 12 || !row.FitsInOneDenseVectorStorage {
		t.Fatalf("fit = %d fits=%t", row.VectorsThatFitInOneDenseVector, row.FitsInOneDenseVectorStorage)
	}
	if got, want := plan.Rows[1].FitsInOneDenseVectorStorage, false; got != want {
		t.Fatalf("16 q2 child vectors should exceed one dense parent budget")
	}
}

func TestPlanMultiVectorStorageAccountsForFP16Sidecar(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:              128,
		Bits:             []int{4},
		Objects:          1,
		VectorsPerObject: []int{1, 2},
		SidecarStorage:   MultiVectorSidecarFP16,
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	row := plan.Rows[0]
	if row.SidecarStorage != MultiVectorSidecarFP16 || row.SidecarBytesPerVector != 256 {
		t.Fatalf("sidecar = storage:%q bytes:%d", row.SidecarStorage, row.SidecarBytesPerVector)
	}
	if row.QuantizedPayloadBytes != 72 || row.QuantizedVectorBytes != 328 {
		t.Fatalf("q4 fp16 bytes = payload:%d vector:%d", row.QuantizedPayloadBytes, row.QuantizedVectorBytes)
	}
	if row.VectorsThatFitInOneDenseVector != 1 || !row.FitsInOneDenseVectorStorage {
		t.Fatalf("one q4 fp16 row fit = %d fits=%t", row.VectorsThatFitInOneDenseVector, row.FitsInOneDenseVectorStorage)
	}
	if plan.Rows[1].FitsInOneDenseVectorStorage {
		t.Fatalf("two q4 fp16 sidecar vectors should exceed one dense parent budget")
	}
}

func TestPlanMultiVectorStorageUsesLargerBaselineDimForDenseBudget(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:              128,
		BaselineDim:      3072,
		Bits:             []int{2},
		Objects:          1,
		VectorsPerObject: []int{128},
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	row := plan.Rows[0]
	if plan.Config.BaselineDim != 3072 || row.BaselineDim != 3072 {
		t.Fatalf("baseline dim = config:%d row:%d", plan.Config.BaselineDim, row.BaselineDim)
	}
	if row.DenseParentBytes != 12288 || row.DenseBaselineBytes != 12288 {
		t.Fatalf("dense baseline bytes = parent:%d baseline:%d", row.DenseParentBytes, row.DenseBaselineBytes)
	}
	if row.QuantizedPayloadBytes != 40 || row.QuantizedVectorBytes != 40 {
		t.Fatalf("q2 bytes = payload:%d vector:%d", row.QuantizedPayloadBytes, row.QuantizedVectorBytes)
	}
	if row.VectorsThatFitInOneDenseVector != 307 || !row.FitsInOneDenseVectorStorage {
		t.Fatalf("fit = %d fits=%t", row.VectorsThatFitInOneDenseVector, row.FitsInOneDenseVectorStorage)
	}
	if row.StorageMultipleOfDenseParentCost < 0.41666 || row.StorageMultipleOfDenseParentCost > 0.41667 {
		t.Fatalf("storage multiple = %.6f", row.StorageMultipleOfDenseParentCost)
	}
}

func TestPlanMultiVectorStorageAccountsForPerVectorOverhead(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:                 128,
		BaselineDim:         3072,
		Bits:                []int{2},
		Objects:             1000,
		VectorsPerObject:    []int{64, 128, 256},
		VectorOverheadBytes: 32,
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if plan.Config.VectorOverheadBytes != 32 {
		t.Fatalf("config overhead = %d", plan.Config.VectorOverheadBytes)
	}
	row := plan.Rows[0]
	if row.QuantizedPayloadBytes != 40 || row.QuantizedVectorBytes != 40 {
		t.Fatalf("raw q2 bytes = payload:%d vector:%d", row.QuantizedPayloadBytes, row.QuantizedVectorBytes)
	}
	if row.VectorOverheadBytes != 32 || row.DenseVectorStorageBytes != 12320 || row.QuantizedVectorStorageBytes != 72 {
		t.Fatalf("storage bytes = overhead:%d dense:%d quantized:%d", row.VectorOverheadBytes, row.DenseVectorStorageBytes, row.QuantizedVectorStorageBytes)
	}
	if row.DenseBaselineTotalBytes != 12320000 || row.TotalQuantizedBytes != 4608000 {
		t.Fatalf("totals = dense:%d quantized:%d", row.DenseBaselineTotalBytes, row.TotalQuantizedBytes)
	}
	if row.VectorsThatFitInOneDenseVector != 171 || !row.FitsInOneDenseVectorStorage {
		t.Fatalf("fit = %d fits=%t", row.VectorsThatFitInOneDenseVector, row.FitsInOneDenseVectorStorage)
	}
	if !plan.Rows[1].FitsInOneDenseVectorStorage {
		t.Fatalf("128 q2 child vectors with 32-byte overhead should fit in one dense baseline storage budget")
	}
	if plan.Rows[2].StorageMultipleOfDenseParentCost < 1.4960 || plan.Rows[2].StorageMultipleOfDenseParentCost > 1.4962 {
		t.Fatalf("storage multiple = %.6f", plan.Rows[2].StorageMultipleOfDenseParentCost)
	}
}

func TestPlanMultiVectorStorageAccountsForPackedParentObjectOverhead(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:                       128,
		BaselineDim:               3072,
		Bits:                      []int{2, 4, 8},
		Objects:                   1000,
		VectorsPerObject:          []int{100},
		VectorOverheadBytes:       32,
		PackedObjectOverheadBytes: 32,
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if plan.Config.PackedObjectOverheadBytes != 32 {
		t.Fatalf("config packed object overhead = %d", plan.Config.PackedObjectOverheadBytes)
	}
	tests := []struct {
		bits             int
		vectorBytes      int64
		currentFit       int64
		packedFit        int64
		packedStorage    int64
		packedTotal      int64
		packedFitsTarget bool
	}{
		{bits: 2, vectorBytes: 40, currentFit: 171, packedFit: 307, packedStorage: 4032, packedTotal: 4032000, packedFitsTarget: true},
		{bits: 4, vectorBytes: 72, currentFit: 118, packedFit: 170, packedStorage: 7232, packedTotal: 7232000, packedFitsTarget: true},
		{bits: 8, vectorBytes: 136, currentFit: 73, packedFit: 90, packedStorage: 13632, packedTotal: 13632000, packedFitsTarget: false},
	}
	for i, tt := range tests {
		row := plan.Rows[i]
		if row.Bits != tt.bits {
			t.Fatalf("row %d bits = %d, want %d", i, row.Bits, tt.bits)
		}
		if row.QuantizedVectorBytes != tt.vectorBytes {
			t.Fatalf("q%d vector bytes = %d, want %d", tt.bits, row.QuantizedVectorBytes, tt.vectorBytes)
		}
		if row.VectorsThatFitInOneDenseVector != tt.currentFit {
			t.Fatalf("q%d current fit = %d, want %d", tt.bits, row.VectorsThatFitInOneDenseVector, tt.currentFit)
		}
		if row.PackedVectorsThatFitInOneDenseVector != tt.packedFit {
			t.Fatalf("q%d packed fit = %d, want %d", tt.bits, row.PackedVectorsThatFitInOneDenseVector, tt.packedFit)
		}
		if row.PackedObjectOverheadBytes != 32 || row.PackedQuantizedStorageBytes != tt.packedStorage || row.PackedTotalQuantizedBytes != tt.packedTotal {
			t.Fatalf("q%d packed bytes = overhead:%d storage:%d total:%d", tt.bits, row.PackedObjectOverheadBytes, row.PackedQuantizedStorageBytes, row.PackedTotalQuantizedBytes)
		}
		if row.PackedFitsInOneDenseVectorStorage != tt.packedFitsTarget {
			t.Fatalf("q%d packed target fit = %t, want %t", tt.bits, row.PackedFitsInOneDenseVectorStorage, tt.packedFitsTarget)
		}
	}
}

func TestTimeSeriesWindowVectorCountCoversTailWindow(t *testing.T) {
	tests := []struct {
		name       string
		points     int
		windowSize int
		stride     int
		want       int
	}{
		{name: "short series", points: 32, windowSize: 64, stride: 16, want: 1},
		{name: "exact window", points: 64, windowSize: 64, stride: 16, want: 1},
		{name: "aligned windows", points: 256, windowSize: 64, stride: 16, want: 13},
		{name: "tail window", points: 257, windowSize: 64, stride: 16, want: 14},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := TimeSeriesWindowVectorCount(tt.points, tt.windowSize, tt.stride)
			if err != nil {
				t.Fatalf("count: %v", err)
			}
			if got != tt.want {
				t.Fatalf("count = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestPlanMultiVectorStorageDerivesVectorsFromTimeSeriesWindows(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:                 128,
		BaselineDim:         3072,
		Bits:                []int{2},
		Objects:             1000,
		SeriesLengths:       []int{256, 1024},
		WindowSize:          64,
		WindowStride:        16,
		VectorOverheadBytes: 32,
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if len(plan.Rows) != 2 {
		t.Fatalf("rows = %d", len(plan.Rows))
	}
	if got, want := plan.Config.VectorsPerObject, []int{13, 61}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("config vectors_per_object = %v, want %v", got, want)
	}
	if plan.Config.WindowSize != 64 || plan.Config.WindowStride != 16 || len(plan.Config.SeriesLengths) != 2 {
		t.Fatalf("config time series = lengths:%v window:%d stride:%d", plan.Config.SeriesLengths, plan.Config.WindowSize, plan.Config.WindowStride)
	}
	row := plan.Rows[0]
	if row.SeriesLength != 256 || row.WindowSize != 64 || row.WindowStride != 16 || row.DerivedWindowCount != 13 {
		t.Fatalf("row time series fields = %+v", row)
	}
	if row.VectorsPerObject != 13 || row.TotalQuantizedBytes != 936000 {
		t.Fatalf("row count/storage = vectors:%d total:%d", row.VectorsPerObject, row.TotalQuantizedBytes)
	}
	if !row.FitsInOneDenseVectorStorage || row.StorageMultipleOfDenseParentCost < 0.07597 || row.StorageMultipleOfDenseParentCost > 0.07598 {
		t.Fatalf("row fit/multiple = fits:%t multiple:%.6f", row.FitsInOneDenseVectorStorage, row.StorageMultipleOfDenseParentCost)
	}
}

func TestPlanMultiVectorStorageRejectsVectorsPerObjectWithSeriesLengths(t *testing.T) {
	_, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:              128,
		Bits:             []int{2},
		Objects:          1,
		VectorsPerObject: []int{13},
		SeriesLengths:    []int{256},
		WindowSize:       64,
	})
	if err == nil {
		t.Fatal("plan succeeded with vectors per object and series lengths")
	}
	if !strings.Contains(err.Error(), "not both") {
		t.Fatalf("error = %q, want conflict message", err)
	}
}

func TestPlanMultiVectorStorageDefaultsWindowStrideToWindowSize(t *testing.T) {
	plan, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:           128,
		Bits:          []int{2},
		Objects:       1,
		SeriesLengths: []int{256},
		WindowSize:    64,
	})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	row := plan.Rows[0]
	if row.WindowStride != 64 || row.DerivedWindowCount != 4 || row.VectorsPerObject != 4 {
		t.Fatalf("default stride row = %+v", row)
	}
}

func TestPlanMultiVectorStorageRejectsInvalidTimeSeriesWindow(t *testing.T) {
	for _, in := range []MultiVectorStoragePlanInput{
		{Dim: 128, Bits: []int{2}, Objects: 1, SeriesLengths: []int{256}},
		{Dim: 128, Bits: []int{2}, Objects: 1, SeriesLengths: []int{0}, WindowSize: 64},
		{Dim: 128, Bits: []int{2}, Objects: 1, SeriesLengths: []int{256}, WindowSize: 64, WindowStride: -1},
	} {
		if _, err := PlanMultiVectorStorage(in); err == nil {
			t.Fatalf("plan succeeded with invalid time series input: %+v", in)
		}
	}
}

func TestPlanMultiVectorStorageRejectsNegativePerVectorOverhead(t *testing.T) {
	_, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:                 128,
		Bits:                []int{2},
		Objects:             1,
		VectorOverheadBytes: -1,
	})
	if err == nil {
		t.Fatal("plan succeeded with negative vector overhead")
	}
}

func TestPlanMultiVectorStorageRejectsNegativePackedObjectOverhead(t *testing.T) {
	_, err := PlanMultiVectorStorage(MultiVectorStoragePlanInput{
		Dim:                       128,
		Bits:                      []int{2},
		Objects:                   1,
		PackedObjectOverheadBytes: -1,
	})
	if err == nil {
		t.Fatal("plan succeeded with negative packed object overhead")
	}
	if !strings.Contains(err.Error(), "packed object overhead bytes must be non-negative") {
		t.Fatalf("error = %q, want packed overhead message", err)
	}
}
