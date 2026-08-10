package eosruntime

import "testing"

// TestTurboquantModuleVersionNonEmpty pins the provenance source: the stamp
// must be a non-empty version string. Under `go test` the build info resolves
// the real module pin; the fallback is the upstream constant.
func TestTurboquantModuleVersionNonEmpty(t *testing.T) {
	v := turboquantModuleVersion()
	if v == "" {
		t.Fatal("turboquantModuleVersion returned empty")
	}
	if v[0] != 'v' {
		t.Fatalf("turboquantModuleVersion = %q, want a vX.Y.Z-style version", v)
	}
}
