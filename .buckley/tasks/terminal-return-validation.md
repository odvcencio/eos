You are fixing one bounded compiler bug in the EOS repository at the exact clean commit named by the harness.

Return ONLY a valid unified git diff, with no Markdown fences, commentary, ellipses, placeholder hunks, or unrelated changes. The response must end in a newline.

Scope:
- Touch only `compiler/semantics.go` and `compiler/compiler_test.go`.
- Do not touch `syntax/cst.go`, `syntax/parser_test.go`, CUDA, ABI, runtime, generated files, docs, or dependencies.
- Keep the patch small and idiomatic Go.

Proven bug:
- The semantic validator says a callable "must end with a return statement", but it only records whether any return exists.
- Therefore this invalid source is accepted by `Build`, and lowering emits work after a return:

  pipeline pass(x: f16[N]) -> f16[N] {
      return x
      let unreachable = normalize(x)
  }

- A focused probe on this exact commit failed with: `Build accepted an unreachable statement after return`.

Required behavior:
- A return terminates a callable body.
- Reject every statement that appears after the first return, including another return, with a source-tied semantic diagnostic on the unreachable statement.
- Avoid misleading cascades from semantically validating unreachable statements.
- Preserve the existing return arity/type validation for the first return.
- Preserve the existing missing-return diagnostic for bodies with no return.
- Add focused table-driven regression coverage through the public `Build` function for at least a trailing let statement and a trailing second return. Assert the diagnostic message and its line/column so the source tie is proven.
- Do not weaken existing tests.

Relevant current source follows verbatim.

`compiler/semantics.go`:

```go
func semanticDiagnostics(file *syntax.File) []syntax.Diagnostic {
	if file == nil {
		return nil
	}

	env := newModuleTypeEnv(file)
	diags := []syntax.Diagnostic{}
	decls := map[string]syntax.Span{}
	for _, decl := range file.Decls {
		name := declName(decl)
		if name == "" {
			continue
		}
		if prev, ok := decls[name]; ok {
			diags = append(diags, diagnosticError(decl.DeclSpan(), "duplicate declaration %q; first declared at %d:%d", name, prev.Line, prev.Column))
			continue
		}
		decls[name] = decl.DeclSpan()
	}

	for _, decl := range file.Decls {
		callable, ok := decl.(*syntax.CallableDecl)
		if !ok {
			continue
		}
		diags = append(diags, validateCallable(callable, env)...)
	}
	return diags
}

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

Statement spans are exposed uniformly:

```go
type Stmt interface {
	stmtNode()
	StmtSpan() Span
}

type LetStmt struct { Name string; Expr Expr; Span Span }
func (s *LetStmt) StmtSpan() Span { return s.Span }

type ReturnStmt struct { Expr Expr; Exprs []Expr; Span Span }
func (s *ReturnStmt) StmtSpan() Span { return s.Span }

type ExprStmt struct { Expr Expr; Span Span }
func (s *ExprStmt) StmtSpan() Span { return s.Span }
```

Diagnostics render with source positions. `compiler/compiler_test.go` already imports `strings` and `testing`, and existing tests use this shape:

```go
func TestBuildRejectsReturnValueCountMismatch(t *testing.T) {
	src := []byte(`
pipeline rerank(query: f16[D], docs: q4[N, D]) -> (top_ids: i32[2], top_scores: f32[2]) {
    let scores = cosine(query, docs)
    return topk(scores, 2)
}
`)

	_, err := Build(src, Options{ModuleName: "bad_return_count"})
	if err == nil {
		t.Fatal("expected diagnostic error")
	}
	if !strings.Contains(err.Error(), "return value count mismatch: got 1, want 2") {
		t.Fatalf("unexpected diagnostic:\n%v", err)
	}
}
```

Implement the smallest correct patch and return only the unified diff.
