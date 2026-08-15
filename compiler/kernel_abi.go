package compiler

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	gotreesitter "github.com/odvcencio/gotreesitter"
	"github.com/odvcencio/gotreesitter/grammars"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

const (
	kernelABIParserFull      = "gotreesitter-cpp-normalized-v1"
	kernelABIParserSignature = "gotreesitter-cpp-signature-v1"
)

// extractKernelABI parses the generated CUDA/Metal source with the pure-Go
// C++ grammar and returns the typed source-level launch contract. CUDA and
// Metal use a small number of vendor qualifiers that are not part of the C++
// grammar shipped by gotreesitter; maskKernelExtensions replaces those tokens
// with spaces (preserving byte offsets) before parsing, then reads annotations
// from the original source.
func extractKernelABI(backend eosartifact.BackendKind, entry, source string) (*eosartifact.KernelABI, error) {
	if backend != eosartifact.BackendCUDA && backend != eosartifact.BackendMetal {
		return nil, nil
	}
	if entry == "" {
		return nil, fmt.Errorf("kernel ABI entry is empty")
	}
	if source == "" {
		return nil, fmt.Errorf("kernel ABI source is empty for %q", entry)
	}

	normalized := maskKernelExtensions(backend, []byte(source))
	parser := gotreesitter.NewParser(grammars.CppLanguage())
	tree, err := parser.Parse(normalized)
	if err != nil {
		return nil, fmt.Errorf("parse %s kernel %q: %w", backend, entry, err)
	}
	root := tree.RootNode()
	parserMode := kernelABIParserFull
	function := findKernelFunction(root, grammars.CppLanguage(), normalized, entry)
	if root == nil || root.HasError() || function == nil {
		// Generated bodies may contain backend builtins that are valid for the
		// target compiler but not represented by the C++ grammar. Reparse a
		// signature-only view so the ABI remains grammar-derived while Prism
		// remains responsible for validating the complete backend source.
		signatureSource := maskKernelBody(normalized, entry)
		signatureTree, signatureErr := parser.Parse(signatureSource)
		if signatureErr != nil {
			return nil, fmt.Errorf("parse %s kernel %q signature: %w", backend, entry, signatureErr)
		}
		signatureRoot := signatureTree.RootNode()
		function = findKernelFunction(signatureRoot, grammars.CppLanguage(), signatureSource, entry)
		if signatureRoot == nil || signatureRoot.HasError() || function == nil {
			if signatureRoot == nil {
				return nil, fmt.Errorf("parse %s kernel %q signature produced no syntax tree", backend, entry)
			}
			return nil, fmt.Errorf("parse %s kernel %q signature has syntax errors: %s", backend, entry, signatureRoot.SExpr(grammars.CppLanguage()))
		}
		parserMode = kernelABIParserSignature
	}

	declarator := function.ChildByFieldName("declarator", grammars.CppLanguage())
	functionDeclarator := findFunctionDeclarator(declarator, grammars.CppLanguage())
	if functionDeclarator == nil {
		return nil, fmt.Errorf("kernel %q has no function declarator", entry)
	}
	parameters := functionDeclarator.ChildByFieldName("parameters", grammars.CppLanguage())
	if parameters == nil {
		return nil, fmt.Errorf("kernel %q has no parameter list", entry)
	}

	hash := sha256.Sum256([]byte(source))
	abi := &eosartifact.KernelABI{
		Version:      eosartifact.KernelABIVersion,
		Entry:        entry,
		Parser:       parserMode,
		SourceSHA256: hex.EncodeToString(hash[:]),
	}
	argIndex := 0
	for i := 0; i < parameters.NamedChildCount(); i++ {
		parameter := parameters.NamedChild(i)
		if parameter == nil || parameter.Type(grammars.CppLanguage()) != "parameter_declaration" {
			continue
		}
		arg, err := kernelABIArg(backend, argIndex, parameter, parameters, source, normalized)
		if err != nil {
			return nil, fmt.Errorf("kernel %q parameter %d: %w", entry, argIndex, err)
		}
		abi.Args = append(abi.Args, arg)
		argIndex++
	}
	return abi, nil
}

func kernelABIArg(backend eosartifact.BackendKind, index int, parameter, parameters *gotreesitter.Node, source string, normalized []byte) (eosartifact.KernelABIArg, error) {
	lang := grammars.CppLanguage()
	declarator := parameter.ChildByFieldName("declarator", lang)
	nameNode := findDeclaratorIdentifier(declarator, lang)
	if nameNode == nil {
		return eosartifact.KernelABIArg{}, fmt.Errorf("parameter has no named declarator")
	}
	name := nameNode.Text(normalized)
	if name == "" {
		return eosartifact.KernelABIArg{}, fmt.Errorf("parameter name is empty")
	}
	normalizedText := strings.TrimSpace(parameter.Text(normalized))
	nameOffset := strings.LastIndex(normalizedText, name)
	if nameOffset < 0 {
		return eosartifact.KernelABIArg{}, fmt.Errorf("parameter %q is not present in declarator text", name)
	}
	typeText := strings.Join(strings.Fields(strings.TrimSpace(normalizedText[:nameOffset])), " ")
	if typeText == "" {
		return eosartifact.KernelABIArg{}, fmt.Errorf("parameter %q has no type", name)
	}

	raw := sourceSlice(source, parameter.StartByte(), parameter.EndByte())
	if end := scanParameterAttributeEnd(source, int(parameter.EndByte()), int(parameters.EndByte())); end > int(parameter.EndByte()) {
		raw = sourceSlice(source, parameter.StartByte(), uint32(end))
	}
	pointer := hasNodeType(declarator, lang, "pointer_declarator")
	kind := "value"
	if pointer {
		kind = "pointer"
	}
	addressSpace := kernelABIAddressSpace(backend, raw, pointer)
	access := "value"
	if pointer {
		if hasWord(raw, "const") {
			access = "read"
		} else {
			access = "write"
		}
	}
	location := kernelABILocation(backend, raw, pointer)
	return eosartifact.KernelABIArg{
		Index:        index,
		Name:         name,
		Type:         typeText,
		Kind:         kind,
		AddressSpace: addressSpace,
		Access:       access,
		Location:     location,
	}, nil
}

func findKernelFunction(root *gotreesitter.Node, lang *gotreesitter.Language, source []byte, entry string) *gotreesitter.Node {
	if root == nil {
		return nil
	}
	if root.Type(lang) == "function_definition" {
		declarator := root.ChildByFieldName("declarator", lang)
		functionDeclarator := findFunctionDeclarator(declarator, lang)
		if functionDeclarator != nil {
			name := findDeclaratorIdentifier(functionDeclarator.ChildByFieldName("declarator", lang), lang)
			if name != nil && name.Text(source) == entry {
				return root
			}
		}
	}
	for i := 0; i < root.NamedChildCount(); i++ {
		if found := findKernelFunction(root.NamedChild(i), lang, source, entry); found != nil {
			return found
		}
	}
	return nil
}

func findFunctionDeclarator(node *gotreesitter.Node, lang *gotreesitter.Language) *gotreesitter.Node {
	if node == nil {
		return nil
	}
	if node.Type(lang) == "function_declarator" {
		return node
	}
	if child := node.ChildByFieldName("declarator", lang); child != nil {
		if found := findFunctionDeclarator(child, lang); found != nil {
			return found
		}
	}
	for i := 0; i < node.NamedChildCount(); i++ {
		if found := findFunctionDeclarator(node.NamedChild(i), lang); found != nil {
			return found
		}
	}
	return nil
}

func findDeclaratorIdentifier(node *gotreesitter.Node, lang *gotreesitter.Language) *gotreesitter.Node {
	if node == nil {
		return nil
	}
	typ := node.Type(lang)
	if typ == "identifier" || typ == "field_identifier" {
		return node
	}
	if child := node.ChildByFieldName("declarator", lang); child != nil {
		if found := findDeclaratorIdentifier(child, lang); found != nil {
			return found
		}
	}
	for i := 0; i < node.NamedChildCount(); i++ {
		if found := findDeclaratorIdentifier(node.NamedChild(i), lang); found != nil {
			return found
		}
	}
	return nil
}

func hasNodeType(node *gotreesitter.Node, lang *gotreesitter.Language, want string) bool {
	if node == nil {
		return false
	}
	if node.Type(lang) == want {
		return true
	}
	for i := 0; i < node.NamedChildCount(); i++ {
		if hasNodeType(node.NamedChild(i), lang, want) {
			return true
		}
	}
	return false
}

func sourceSlice(source string, start, end uint32) string {
	if start > end || int(end) > len(source) {
		return ""
	}
	return source[int(start):int(end)]
}

func scanParameterAttributeEnd(source string, start, limit int) int {
	if start < 0 || limit > len(source) || start > limit {
		return start
	}
	i := start
	for i < limit {
		switch source[i] {
		case ' ', '\t', '\r', '\n':
			i++
		default:
			if i+1 >= limit || source[i] != '[' || source[i+1] != '[' {
				return start
			}
			end := strings.Index(source[i+2:limit], "]]")
			if end < 0 {
				return start
			}
			return i + 2 + end + 2
		}
	}
	return start
}

func kernelABIAddressSpace(backend eosartifact.BackendKind, raw string, pointer bool) string {
	for _, space := range []string{"device", "constant", "threadgroup", "thread"} {
		if hasWord(raw, space) {
			return space
		}
	}
	if backend == eosartifact.BackendCUDA && pointer {
		return "global"
	}
	return ""
}

func kernelABILocation(backend eosartifact.BackendKind, raw string, pointer bool) string {
	if backend == eosartifact.BackendCUDA {
		if pointer {
			return "global"
		}
		return "value"
	}
	for _, attr := range bracketAttributes(raw) {
		attr = strings.TrimSpace(attr)
		if strings.HasPrefix(attr, "buffer(") && strings.HasSuffix(attr, ")") {
			return "buffer:" + strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(attr, "buffer("), ")"))
		}
		if strings.Contains(attr, "thread_position_in_grid") {
			return "builtin:thread_position_in_grid"
		}
	}
	return ""
}

func bracketAttributes(raw string) []string {
	var out []string
	for {
		start := strings.Index(raw, "[[")
		if start < 0 {
			return out
		}
		rest := raw[start+2:]
		end := strings.Index(rest, "]]")
		if end < 0 {
			return out
		}
		out = append(out, rest[:end])
		raw = rest[end+2:]
	}
}

func hasWord(text, word string) bool {
	for start := 0; ; {
		i := strings.Index(text[start:], word)
		if i < 0 {
			return false
		}
		i += start
		beforeOK := i == 0 || !isIdentifierByte(text[i-1])
		after := i + len(word)
		afterOK := after >= len(text) || !isIdentifierByte(text[after])
		if beforeOK && afterOK {
			return true
		}
		start = i + len(word)
		if start >= len(text) {
			return false
		}
	}
}

func isIdentifierByte(b byte) bool {
	return b == '_' || b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9'
}

func maskKernelExtensions(backend eosartifact.BackendKind, source []byte) []byte {
	masked := append([]byte(nil), source...)
	if backend == eosartifact.BackendMetal {
		maskKernelAttributes(masked)
		for _, word := range []string{"kernel", "device", "constant", "threadgroup", "thread"} {
			maskWord(masked, word)
		}
	} else if backend == eosartifact.BackendCUDA {
		for _, word := range []string{"__global__", "__device__", "__host__", "__forceinline__", "__restrict__"} {
			maskWord(masked, word)
		}
	}
	return masked
}

func maskWord(source []byte, word string) {
	for i := 0; i+len(word) <= len(source); {
		if string(source[i:i+len(word)]) != word || i > 0 && isIdentifierByte(source[i-1]) || i+len(word) < len(source) && isIdentifierByte(source[i+len(word)]) {
			i++
			continue
		}
		for j := i; j < i+len(word); j++ {
			if source[j] != '\n' {
				source[j] = ' '
			}
		}
		i += len(word)
	}
}

func maskKernelAttributes(source []byte) {
	for i := 0; i+1 < len(source); {
		if source[i] != '[' || source[i+1] != '[' {
			i++
			continue
		}
		end := -1
		for j := i + 2; j+1 < len(source); j++ {
			if source[j] == ']' && source[j+1] == ']' {
				end = j + 2
				break
			}
		}
		if end < 0 {
			return
		}
		for j := i; j < end; j++ {
			if source[j] != '\n' {
				source[j] = ' '
			}
		}
		i = end
	}
}

func maskKernelBody(source []byte, entry string) []byte {
	masked := append([]byte(nil), source...)
	start := findKernelEntryOffset(masked, entry)
	if start < 0 {
		return masked
	}
	open := -1
	for i := start + len(entry); i < len(masked); i++ {
		if masked[i] == '{' {
			open = i
			break
		}
	}
	if open < 0 {
		return masked
	}
	close := matchingBrace(masked, open)
	if close < 0 {
		return masked
	}
	for i := open + 1; i < close; i++ {
		if masked[i] != '\n' {
			masked[i] = ' '
		}
	}
	return masked
}

func findKernelEntryOffset(source []byte, entry string) int {
	for start := 0; ; {
		i := strings.Index(string(source[start:]), entry+"(")
		if i < 0 {
			return -1
		}
		i += start
		if (i == 0 || !isIdentifierByte(source[i-1])) && (i+len(entry) >= len(source) || !isIdentifierByte(source[i+len(entry)])) {
			return i
		}
		start = i + len(entry)
		if start >= len(source) {
			return -1
		}
	}
}

func matchingBrace(source []byte, open int) int {
	depth := 0
	var quote byte
	escaped := false
	lineComment := false
	blockComment := false
	for i := open; i < len(source); i++ {
		b := source[i]
		if lineComment {
			if b == '\n' {
				lineComment = false
			}
			continue
		}
		if blockComment {
			if b == '*' && i+1 < len(source) && source[i+1] == '/' {
				blockComment = false
				i++
			}
			continue
		}
		if quote != 0 {
			if escaped {
				escaped = false
				continue
			}
			if b == '\\' {
				escaped = true
				continue
			}
			if b == quote {
				quote = 0
			}
			continue
		}
		if b == '/' && i+1 < len(source) && source[i+1] == '/' {
			lineComment = true
			i++
			continue
		}
		if b == '/' && i+1 < len(source) && source[i+1] == '*' {
			blockComment = true
			i++
			continue
		}
		if b == '\'' || b == '"' {
			quote = b
			continue
		}
		if b == '{' {
			depth++
		} else if b == '}' {
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}
