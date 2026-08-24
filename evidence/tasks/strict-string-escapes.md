# Strict Eos string escapes

This is a one-use patch-generation request. You have no interactive tools in
this call. Return only a complete Git unified diff whose first bytes are
`diff --git ` and whose final byte is a newline. Do not use Markdown fences or
put prose before or after the diff.

Implement one bounded correctness fix in exactly these files:

- `syntax/cst.go`
- `syntax/parser_test.go`

Eos's generated grammar intentionally accepts a generic escaped byte in a
`string_literal`: `([^"\\]|\\.)*`. Do not modify the grammar or regenerate
`syntax/grammar.bin`. Semantic lowering must instead reject literals that
`strconv.Unquote` considers malformed. Today `unquoteEosString` silently strips
quotes after any unquote error, so `\q` reaches both `ParamDecl.Binding` from
`@weight(...)` and `StringExpr.Value` without a diagnostic.

Required behavior:

1. Lower a string through one small shared helper that returns failure after
   appending a `SeverityError` diagnostic on the exact `string_literal` node.
   The message must clearly identify an invalid Eos string literal.
2. A malformed `@weight` literal must not produce its `ParamDecl`.
3. A malformed expression literal must not produce a `StringExpr` (and its
   enclosing statement may consequently be omitted), while the rest of the
   current lowering behavior stays unchanged.
4. Preserve valid escaped quote and escaped backslash decoding for both weight
   bindings and ordinary string expressions.
5. Add focused parser tests for invalid `\q` at both call sites, source-local
   error diagnostics, and valid quote/backslash decoding. Keep tests
   deterministic and use current helpers/types; do not add dependencies.

Exact relevant source at this base follows.

`syntax/cst.go` imports `fmt`, `strconv`, and `strings`. Its parameter lowering
currently ends with:

```go
weight := firstNamedChildOfType(l.Walker, bindingNode, "string_literal")
if weight == nil {
	l.errorf(bindingNode, "expected @weight annotation")
	return nil
}
return &ParamDecl{
	Name:      l.Text(nameNode),
	Type:      typ,
	Binding:   unquoteEosString(l.Text(weight)),
	Trainable: l.Field(n, "trainable") != nil,
	Span:      spanOf(n),
}
```

Its expression switch currently contains:

```go
case "string_literal":
	return &StringExpr{Value: unquoteEosString(l.Text(n)), Span: spanOf(n)}
```

Diagnostics are added with this existing source-local helper:

```go
func (l *cstLowerer) errorf(n *gotreesitter.Node, format string, args ...any) {
	l.diags = append(l.diags, Diagnostic{
		Severity: SeverityError,
		Message:  fmt.Sprintf(format, args...),
		Span:     spanOf(n),
	})
}
```

The unsafe helper at the end of the file is:

```go
func unquoteEosString(raw string) string {
	value, err := strconv.Unquote(raw)
	if err == nil {
		return value
	}
	return strings.Trim(raw, `"`)
}
```

`Parse(moduleName, src)` returns `(*File, []Diagnostic)`. `File.Decls` contains
`*ParamDecl` and `*CallableDecl`; `ParamDecl.Binding` and `StringExpr.Value` are
strings. `Diagnostic` has `Severity`, `Message`, and `Span`; `Span` has byte
`Start`/`End` plus one-based `Line`/`Column`. `syntax/parser_test.go` is package
`syntax` and already imports `strings` and `testing`, so append focused tests
without rewriting existing ones.

Do not claim to run tests. Do not change APIs outside this package, add docs,
regenerate blobs, or touch any other file.
