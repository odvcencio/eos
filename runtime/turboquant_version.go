package eosruntime

import (
	"runtime/debug"
	"sync"

	"m31labs.dev/turboquant"
)

var turboquantModuleVersionOnce = sync.OnceValue(func() string {
	// Prefer the resolved module version from build info: it reflects the
	// go.mod pin (for example "v0.2.1") even when the upstream
	// turboquant.Version constant lags its own tag, which happened when the
	// v0.2.1 tag was cut one commit before the constant bump. Provenance
	// stamping must track what was actually linked, not what the dependency
	// believes about itself.
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, dep := range info.Deps {
			if dep.Path == "m31labs.dev/turboquant" {
				if dep.Replace != nil && dep.Replace.Version != "" {
					return dep.Replace.Version
				}
				if dep.Version != "" && dep.Version != "(devel)" {
					return dep.Version
				}
			}
		}
	}
	return turboquant.Version
})

// turboquantModuleVersion returns the linked turboquant module version for
// provenance stamping on quantized scoreboard rows.
func turboquantModuleVersion() string {
	return turboquantModuleVersionOnce()
}
