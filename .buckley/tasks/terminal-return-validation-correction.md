Return ONLY a valid unified git diff, with no Markdown fences, commentary, ellipses, placeholder hunks, fake index OIDs, or unrelated changes. End the response with a newline.

This is one bounded correction pass for the EOS compiler at the exact clean commit named by the harness. Touch only `compiler/semantics.go` and `compiler/compiler_test.go`. Do not touch syntax lowering, CUDA, ABI, runtime, generated files, docs, or dependencies.

Bug and required behavior:
- `Build` currently accepts statements after a return, then lowering can emit unreachable work.
- A return terminates a callable body.
- Report `unreachable statement after return` at every statement following the first return, including another return.
- Do not semantically validate unreachable statements, avoiding cascading diagnostics.
- Preserve validation of the first return and the existing no-return diagnostic.
- Add table-driven public-`Build` regression tests for a trailing let and a second return.
- Assert the source-tied rendering, which has the form `error:3:5: unreachable statement after return`.

The prior response was rejected. It put the `hasReturn` check in a Go type-switch `default`; that branch only handles unmatched concrete statement types, so it never runs for `*syntax.LetStmt`, `*syntax.ReturnStmt`, or `*syntax.ExprStmt`. It also silently continued on a second return without emitting the required diagnostic, used `fmt.Sprintf` without adding `fmt`, and emitted fake `0000000..1111111` index lines.

The guard must apply to all known statement types before the type switch. A minimal shape is:

```go
for _, stmt := range callable.Body {
	if hasReturn {
		diags = append(diags, diagnosticError(stmt.StmtSpan(), "unreachable statement after return"))
		continue
	}
	switch s := stmt.(type) {
	// existing validation cases
	}
}
```

Current `validateCallable` source follows verbatim:

```go
func validateCallable(callable *syntax.CallableDecl, env moduleTypeEnv) []syntax.Diagnostic {
	diags := []syntax.Diagnostic{}
	locals := env.forCallable(callable)
	localDecls := map[string]syntax.Span{}
	for _, param := range callable.Params {
		if prev, ok := localDecls[param.Name]; ok {
			diags = append(diags, diagnosticError(param.Span, "duplicate parameter %q; first declared at %d:%d", param.Name, prev.Line, prev.Column))
			continue
		}
		localDecls[param.Name] = param.Span
	}

	resultFields := callableResultFields(callable)
	hasReturn := false
	for _, stmt := range callable.Body {
		switch s := stmt.(type) {
		case *syntax.LetStmt:
			if prev, ok := localDecls[s.Name]; ok {
				diags = append(diags, diagnosticError(s.Span, "duplicate local %q; first declared at %d:%d", s.Name, prev.Line, prev.Column))
				continue
			}
			localDecls[s.Name] = s.Span
			typ, stmtDiags := inferSemanticExprType(s.Expr, callable, locals, env)
			diags = append(diags, stmtDiags...)
			if typ.Kind != "" {
				locals[s.Name] = typ
			}
		case *syntax.ReturnStmt:
			hasReturn = true
			exprs := returnStmtExprs(s)
			if len(exprs) != len(resultFields) {
				diags = append(diags, diagnosticError(s.Span, "return value count mismatch: got %d, want %d", len(exprs), len(resultFields)))
				continue
			}
			for i, expr := range exprs {
				typ, stmtDiags := inferSemanticExprType(expr, callable, locals, env)
				diags = append(diags, stmtDiags...)
				want := lowerType(resultFields[i].Type)
				if typ.Kind != "" && !sameType(typ, want) {
					diags = append(diags, diagnosticError(expr.ExprSpan(), "return value %d type mismatch: got %s, want %s", i+1, typeString(typ), typeString(want)))
				}
			}
		case *syntax.ExprStmt:
			_, stmtDiags := inferSemanticExprType(s.Expr, callable, locals, env)
			diags = append(diags, stmtDiags...)
			if !exprHasEffect(s.Expr) {
				diags = append(diags, diagnosticError(s.Span, "expression statement has no effect"))
			}
		}
	}
	if !hasReturn {
		diags = append(diags, diagnosticError(callable.Span, "%s %q must end with a return statement", callable.Kind, callable.Name))
	}
	return diags
}
```

`syntax.Stmt` provides `StmtSpan() Span` for all three current statement types. `compiler/compiler_test.go` already imports `strings` and `testing`; avoid adding imports that are not needed. Existing tests use `strings.Contains(err.Error(), ...)`.

Produce the smallest correct patch now.
