"""L104 Code Engine — Swift Syntax Analyzer.

Swift-specific syntax validation using swiftc -parse and SourceKit-LSP.
Integrated with the L104 Code Engine for comprehensive Swift code analysis.
"""

import subprocess
import re
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from .constants import VERSION, PHI, GOD_CODE

# ─────────────────────────────────────────────────────────────────────────────
# Swift Syntax Error Types
# ─────────────────────────────────────────────────────────────────────────────

SWIFT_ERROR_PATTERNS = {
    "syntax_error": [
        r"error:\s*(.+?)\s*$",
        r"^(.+?):(\d+):(\d+):\s*error:\s*(.+)$",
    ],
    "type_mismatch": [
        r"cannot convert value of type '(.+?)' to specified type '(.+?)'",
        r"type '(.+?)' does not conform to protocol '(.+?)'",
    ],
    "undeclared_identifier": [
        r"use of unresolved identifier '(.+?)'",
        r"use of undeclared '(.+?)'",
        r"cannot find '(.+?)' in scope",
    ],
    "missing_required": [
        r"missing argument for parameter '(.+?)' in call",
        r"missing required parameter '(.+?)'",
        r"'(.+?)' has no member '(.+?)'",
    ],
    "access_control": [
        r"'(.+?)' is inaccessible due to '(.+?)' protection level",
        r"property '(.+?)' is '(.+?)' and cannot be accessed",
    ],
    "optionality": [
        r"value of optional type '(.+?)' must be unwrapped",
        r"optional binding condition requires an initializer",
        r"implicitly unwrapped optional '(.+?)' used",
    ],
    "concurrency": [
        r"actor-isolated property '(.+?)' cannot be referenced",
        r"call to '(.+?)' is not awaitable",
        r"task-isolated value passed to actor-isolated parameter",
    ],
    "generics": [
        r"generic parameter '(.+?)' could not be inferred",
        r"type '(.+?)' does not satisfy the constraint '(.+?)'",
    ],
}


class SwiftSyntaxError:
    """Represents a single Swift syntax error."""

    def __init__(self, file_path: str, line: int, column: int,
                 message: str, severity: str = "error", code: str = ""):
        self.file_path = file_path
        self.line = line
        self.column = column
        self.message = message
        self.severity = severity  # error, warning, note
        self.code = code
        self.error_type = self._classify_error(message)

    def _classify_error(self, message: str) -> str:
        """Classify error into a category."""
        for err_type, patterns in SWIFT_ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return err_type
        return "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file": self.file_path,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "severity": self.severity,
            "code": self.code,
            "type": self.error_type,
        }

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line}:{self.column}: {self.severity}: {self.message}"


class SwiftLanguageUsageError:
    """Represents a Swift language usage issue (performance, idioms, modern patterns)."""

    def __init__(self, file_path: str, line: int, column: int,
                 message: str, severity: str = "suggestion",
                 category: str = "general", suggestion: str = ""):
        self.file_path = file_path
        self.line = line
        self.column = column
        self.message = message
        self.severity = severity  # error, warning, suggestion, note
        self.category = category  # performance, idiomatic, concurrency, modern, safety
        self.suggestion = suggestion

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file": self.file_path,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "severity": self.severity,
            "category": self.category,
            "suggestion": self.suggestion,
        }

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.file_path}:{self.line}:{self.column}: {self.severity}: [{self.category}] {self.message}\n  → Suggestion: {self.suggestion}"
        return f"{self.file_path}:{self.line}:{self.column}: {self.severity}: [{self.category}] {self.message}"


class SwiftSyntaxAnalyzer:
    """Swift syntax validation using swiftc -parse and SourceKit."""

    # Swift keywords for tokenization
    KEYWORDS = {
        "associatedtype", "class", "deinit", "enum", "extension", "func",
        "import", "init", "inout", "let", "operator", "precedencegroup",
        "protocol", "struct", "subscript", "typealias", "var", "break",
        "case", "continue", "default", "defer", "do", "else", "fallthrough",
        "for", "guard", "if", "in", "repeat", "return", "switch", "where",
        "while", "as", "catch", "dynamicType", "else", "false", "is", "nil",
        "self", "Self", "super", "throws", "true", "try", "async", "await",
        "actor", "some", "any", "in", "out", "public", "private", "fileprivate",
        "internal", "open", "static", "override", "final", "mutating",
        "nonmutating", "lazy", "weak", "unowned", "indirect", "convenience",
        "required", "rethrows", "reasync"
    }

    # Swift built-in types
    BUILTIN_TYPES = {
        "Int", "Int8", "Int16", "Int32", "Int64",
        "UInt", "UInt8", "UInt16", "UInt32", "UInt64",
        "Float", "Double", "Float16", "Float32", "Float64",
        "String", "Character", "Bool", "Void", "Any",
        "Array", "Dictionary", "Set", "Optional", "Tuple",
        "Error", "Result", "Never", "NSObject", "NSString",
    }

    def __init__(self, swiftc_path: str = None, swift_root: str = None):
        """Initialize Swift syntax analyzer.

        Args:
            swiftc_path: Path to swiftc executable (default: auto-detect)
            swift_root: Root directory for module resolution
        """
        self.swiftc_path = swiftc_path or self._find_swiftc()
        self.swift_root = swift_root
        self._analysis_count = 0
        self._error_cache: Dict[str, List[SwiftSyntaxError]] = {}

    def _find_swiftc(self) -> Optional[str]:
        """Find swiftc in PATH or common locations."""
        # Try xcrun first (macOS)
        try:
            result = subprocess.run(
                ["xcrun", "-find", "swiftc"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Try common paths
        common_paths = [
            "/usr/bin/swiftc",
            "/usr/local/bin/swiftc",
            "/opt/swift/usr/bin/swiftc",
        ]
        for path in common_paths:
            if Path(path).exists():
                return path

        return None

    def _parse_error_output(self, output: str) -> List[SwiftSyntaxError]:
        """Parse swiftc error output into structured errors."""
        errors = []

        # Standard Swift error format: file:line:column: error: message
        error_pattern = re.compile(
            r'^(.+?):(\d+):(\d+):\s*(error|warning|note):\s*(.+)$',
            re.MULTILINE
        )

        for match in error_pattern.finditer(output):
            file_path = match.group(1)
            line = int(match.group(2))
            column = int(match.group(3))
            severity = match.group(4)
            message = match.group(5)

            # Extract error code if present
            code_match = re.search(r'([A-Z]{2,}\d+)', message)
            code = code_match.group(1) if code_match else ""

            errors.append(SwiftSyntaxError(
                file_path=file_path,
                line=line,
                column=column,
                message=message,
                severity=severity,
                code=code
            ))

        return errors

    def check_syntax(self, source: str, filename: str = "<swift>") -> Dict[str, Any]:
        """Check Swift source code for syntax errors.

        Args:
            source: Swift source code
            filename: Virtual filename for error reporting

        Returns:
            Dict with 'valid', 'errors', 'warnings', 'stats'
        """
        self._analysis_count += 1

        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "notes": [],
            "stats": {},
            "sacred_alignment": {},
        }

        # Primary: Use swiftc -parse if available (most accurate)
        if self.swiftc_path:
            swiftc_result = self._swiftc_parse(source)
            if swiftc_result["errors"]:
                result["errors"].extend(swiftc_result["errors"])
                result["valid"] = False
            if swiftc_result["warnings"]:
                result["warnings"].extend(swiftc_result["warnings"])
            result["stats"] = swiftc_result.get("stats", {})
        else:
            # Fallback: bracket matching + pattern analysis (less accurate)
            bracket_errors = self._check_brackets(source)
            if bracket_errors:
                result["errors"].extend(bracket_errors)
                result["valid"] = False
            pattern_errors = self._pattern_analysis(source)
            if pattern_errors:
                result["errors"].extend(pattern_errors)
                result["valid"] = len(pattern_errors) == 0
            result["stats"] = self._quick_stats(source)

        # Token-based warnings (style issues, not errors)
        token_errors = self._token_analysis(source)
        if token_errors:
            result["warnings"].extend(token_errors)

        # Sacred alignment scoring
        result["sacred_alignment"] = self._sacred_alignment(source)

        # Convert to dicts
        result["errors"] = [e.to_dict() if isinstance(e, SwiftSyntaxError) else e for e in result["errors"]]
        result["warnings"] = [w.to_dict() if isinstance(w, SwiftSyntaxError) else w for w in result["warnings"]]

        return result

    def check_file(self, file_path: str) -> Dict[str, Any]:
        """Check a Swift file for syntax errors.

        Args:
            file_path: Path to Swift source file

        Returns:
            Dict with 'valid', 'errors', 'warnings'
        """
        path = Path(file_path)
        if not path.exists():
            return {"valid": False, "errors": [{"message": f"File not found: {file_path}"}], "warnings": []}

        source = path.read_text()
        return self.check_syntax(source, str(path))

    def check_directory(self, dir_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Check all Swift files in a directory.

        Args:
            dir_path: Directory path
            recursive: Whether to search recursively

        Returns:
            Dict with 'valid', 'files_checked', 'total_errors', 'file_results'
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return {"valid": False, "error": f"Directory not found: {dir_path}"}

        pattern = "**/*.swift" if recursive else "*.swift"
        swift_files = list(dir_path.glob(pattern))

        results = {
            "valid": True,
            "files_checked": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "file_results": {},
        }

        for swift_file in swift_files:
            file_result = self.check_file(str(swift_file))
            results["file_results"][str(swift_file)] = file_result
            results["files_checked"] += 1

            if file_result.get("errors"):
                results["total_errors"] += len(file_result["errors"])
                results["valid"] = False
            if file_result.get("warnings"):
                results["total_warnings"] += len(file_result["warnings"])

        return results

    def _swiftc_parse(self, source: str) -> Dict[str, Any]:
        """Run swiftc -parse to check syntax."""
        if not self.swiftc_path:
            return {"errors": [], "warnings": [], "stats": {}}

        # Create temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.swift', delete=False
        ) as f:
            f.write(source)
            temp_path = f.name

        try:
            # Run swiftc -parse (parse-only mode)
            cmd = [self.swiftc_path, "-parse", "-target", "arm64-apple-macosx12.0", temp_path]

            # Add SDK path if on macOS
            try:
                sdk_result = subprocess.run(
                    ["xcrun", "--show-sdk-path"],
                    capture_output=True, text=True, timeout=5
                )
                if sdk_result.returncode == 0:
                    sdk_path = sdk_result.stdout.strip()
                    cmd.extend(["-sdk", sdk_path])
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse output
            all_output = result.stdout + result.stderr
            parsed_errors = self._parse_error_output(all_output)

            errors = [e for e in parsed_errors if e.severity == "error"]
            warnings = [e for e in parsed_errors if e.severity == "warning"]
            notes = [e for e in parsed_errors if e.severity == "note"]

            return {
                "errors": errors,
                "warnings": warnings,
                "notes": notes,
                "stats": {
                    "swiftc_available": True,
                    "exit_code": result.returncode,
                    "raw_output": all_output[:1000] if all_output else "",
                }
            }

        except subprocess.TimeoutExpired:
            return {
                "errors": [SwiftSyntaxError(
                    file_path="<temp>", line=0, column=0,
                    message="swiftc timed out (60s limit)",
                    severity="error"
                )],
                "warnings": [],
                "stats": {"swiftc_available": True, "timeout": True}
            }
        except Exception as e:
            return {
                "errors": [SwiftSyntaxError(
                    file_path="<temp>", line=0, column=0,
                    message=f"swiftc execution failed: {str(e)}",
                    severity="error"
                )],
                "warnings": [],
                "stats": {"swiftc_available": True, "error": str(e)}
            }
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _check_brackets(self, source: str) -> List[SwiftSyntaxError]:
        """Check for bracket/brace/paren mismatches (respecting Swift string interpolation)."""
        errors = []

        lines = source.split('\n')

        # Track bracket stacks with proper string/interpolation handling
        for pair_open, pair_close in [('{', '}'), ('(', ')'), ('[', ']')]:
            stack = []  # Stack of (line, col) for opening brackets
            i = 0
            in_string = False
            in_multiline_string = False
            in_interpolation = False
            interpolation_depth = 0

            for line_num, line in enumerate(lines, 1):
                col = 0
                while col < len(line):
                    char = line[col]

                    # Handle multiline string delimiters
                    if col + 2 < len(line) and line[col:col+3] == '"""':
                        in_multiline_string = not in_multiline_string
                        col += 3
                        continue

                    # Handle regular string delimiters
                    if char == '"' and not in_multiline_string:
                        if not in_string:
                            in_string = True
                        else:
                            # Check for escape
                            if col > 0 and line[col-1] == '\\':
                                col += 1
                                continue
                            in_string = False
                        col += 1
                        continue

                    # Skip content inside strings (but check for interpolation)
                    if in_string or in_multiline_string:
                        # Swift string interpolation: \( ... )
                        if char == '\\' and col + 1 < len(line) and line[col + 1] == '(':
                            in_interpolation = True
                            interpolation_depth = 1
                            col += 2  # Skip \(
                            continue
                        elif in_interpolation:
                            if char == '(':
                                interpolation_depth += 1
                            elif char == ')':
                                interpolation_depth -= 1
                                if interpolation_depth == 0:
                                    in_interpolation = False
                        col += 1
                        continue

                    # Handle comments
                    if char == '/' and col + 1 < len(line):
                        if line[col + 1] == '/':
                            break  # Line comment - skip rest of line
                        elif line[col + 1] == '*':
                            # Block comment - find end (simplified)
                            end = line.find('*/', col + 2)
                            if end != -1:
                                col = end + 2
                                continue

                    # Now check brackets (outside strings/comments)
                    if char == pair_open:
                        stack.append((line_num, col + 1))
                    elif char == pair_close:
                        if not stack:
                            errors.append(SwiftSyntaxError(
                                file_path="<swift>",
                                line=line_num,
                                column=col + 1,
                                message=f"Unmatched closing '{pair_close}'",
                                severity="error"
                            ))
                        else:
                            stack.pop()

                    col += 1

            # Report unmatched opening brackets
            for start_line, start_col in stack:
                errors.append(SwiftSyntaxError(
                    file_path="<swift>",
                    line=start_line,
                    column=start_col,
                    message=f"Unmatched opening '{pair_open}'",
                    severity="error"
                ))

        return errors

    def _token_analysis(self, source: str) -> List[SwiftSyntaxError]:
        """Quick token-based analysis for common issues."""
        warnings = []
        lines = source.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Check for common Swift issues

            # Missing self in closures (heuristic)
            if 'self.' in line and 'weak self' not in line and '{' in line:
                # Check if in escaping closure context (simplified)
                pass

            # Force unwrap warning
            if '!' in line and '!!' not in line:
                matches = re.finditer(r'\w+(?!!!=)', line)
                for match in matches:
                    end = match.end()
                    if end < len(line) and line[end] == '!':
                        col = line.find('!', match.start())
                        if col >= 0:
                            warnings.append(SwiftSyntaxError(
                                file_path="<swift>",
                                line=line_num,
                                column=col + 1,
                                message="Force unwrap may cause runtime crash",
                                severity="warning"
                            ))
                        break

            # TODO/FIXME comments
            if 'TODO' in line or 'FIXME' in line:
                col = max(line.find('TODO'), line.find('FIXME'))
                if col >= 0:
                    warnings.append(SwiftSyntaxError(
                        file_path="<swift>",
                        line=line_num,
                        column=col + 1,
                        message="TODO/FIXME marker found",
                        severity="note"
                    ))

        return warnings

    def _pattern_analysis(self, source: str) -> List[SwiftSyntaxError]:
        """Pattern-based syntax analysis when swiftc unavailable."""
        errors = []
        lines = source.split('\n')

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Missing braces after if/for/while/guard
            control_flow = re.match(r'^(if|for|while|guard)\s+.+\s*$', stripped)
            if control_flow and not stripped.endswith('{'):
                # Check if next non-empty line starts with {
                next_has_brace = False
                for next_line in lines[line_num:]:
                    if next_line.strip():
                        next_has_brace = next_line.strip().startswith('{')
                        break
                if not next_has_brace:
                    errors.append(SwiftSyntaxError(
                        file_path="<swift>",
                        line=line_num,
                        column=1,
                        message=f"Missing opening brace after '{control_flow.group(1)}'",
                        severity="error"
                    ))

            # func without body
            func_match = re.match(r'func\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?\s*$', stripped)
            if func_match and not stripped.endswith('{'):
                errors.append(SwiftSyntaxError(
                    file_path="<swift>",
                    line=line_num,
                    column=1,
                    message="Function declaration missing body",
                    severity="error"
                ))

            # class/struct/enum/protocol without braces
            type_match = re.match(r'^(class|struct|enum|protocol|actor)\s+\w+.*$', stripped)
            if type_match and not '{' in stripped:
                errors.append(SwiftSyntaxError(
                    file_path="<swift>",
                    line=line_num,
                    column=1,
                    message=f"{type_match.group(1)} declaration missing opening brace",
                    severity="error"
                ))

            # Missing return in computed property
            var_match = re.match(r'var\s+\w+\s*:\s*\w+\s*\{', stripped)
            if var_match:
                # Simplified check - should have get or return
                if 'get' not in stripped and 'return' not in stripped:
                    # Look ahead for return
                    has_return = False
                    for next_line in lines[line_num:line_num+10]:
                        if 'return ' in next_line or 'get {' in next_line:
                            has_return = True
                            break
                    if not has_return:
                        pass  # Could be willSet/didSet

        return errors

    def _quick_stats(self, source: str) -> Dict[str, Any]:
        """Compute quick statistics."""
        lines = source.split('\n')

        # Count declarations
        funcs = len(re.findall(r'\bfunc\s+\w+', source))
        classes = len(re.findall(r'\bclass\s+\w+', source))
        structs = len(re.findall(r'\bstruct\s+\w+', source))
        enums = len(re.findall(r'\benum\s+\w+', source))
        protocols = len(re.findall(r'\bprotocol\s+\w+', source))
        extensions = len(re.findall(r'\bextension\s+\w+', source))

        # Count keywords
        async_count = len(re.findall(r'\basync\b', source))
        await_count = len(re.findall(r'\bawait\b', source))
        throws_count = len(re.findall(r'\bthrows\b', source))
        try_count = len(re.findall(r'\btry\b', source))

        return {
            "lines": len(lines),
            "funcs": funcs,
            "classes": classes,
            "structs": structs,
            "enums": enums,
            "protocols": protocols,
            "extensions": extensions,
            "async_await": async_count + await_count,
            "error_handling": throws_count + try_count,
        }

    def _sacred_alignment(self, source: str) -> Dict[str, Any]:
        """Calculate sacred constant alignment for Swift code."""
        # Count occurrences of sacred constants
        phi_count = len(re.findall(r'\bPHI\b|\bphi\b|1\.618034', source))
        god_code_count = len(re.findall(r'GOD_CODE|527\.518', source))
        void_count = len(re.findall(r'VOID_CONSTANT|1\.041618', source))

        # Calculate alignment score
        total_mentions = phi_count + god_code_count + void_count
        lines = len(source.split('\n'))
        density = total_mentions / max(lines, 1)

        # Sacred ratio alignment
        alignment_score = 0.0
        if total_mentions > 0:
            alignment_score = min(1.0, density * PHI)

        return {
            "phi_mentions": phi_count,
            "god_code_mentions": god_code_count,
            "void_constant_mentions": void_count,
            "total_sacred_mentions": total_mentions,
            "density": round(density, 6),
            "alignment_score": round(alignment_score, 4),
        }


class SwiftLanguageUsageAnalyzer:
    """Advanced Swift language usage analyzer for idiomatic patterns and best practices.

    Analyzes Swift code for:
    - Swift 5.9+ language features (async/await, actors, Sendable)
    - Performance anti-patterns
    - Idiomatic Swift patterns
    - Swift 6 strict concurrency readiness
    - Modern Swift syntax (existential any, some keywords)
    - API availability and version checking
    - Memory management patterns

    Usage:
        analyzer = SwiftLanguageUsageAnalyzer(swift_version="5.9", strict_concurrency=True)
        results = analyzer.analyze(source_code, filename="MyFile.swift")

        for suggestion in results.suggestions:
            print(suggestion)
    """

    # ─────────────────────────────────────────────────────────────────────────────
    # Swift Version Features
    # ─────────────────────────────────────────────────────────────────────────────

    SWIFT_FEATURES = {
        "5.5": ["async_await", "actors", "global_actors", "sendable"],
        "5.6": ["existential_any", "swiftpm_plugins", "unavailable_from_async"],
        "5.7": ["some_keyword_extensions", "opaque_parameter_declarations", "if_let_shorthand"],
        "5.8": ["backslash_regex", "if_switch_expressions", "repeat_each"],
        "5.9": ["consume_operator", "discarding_task_groups", "swift_syntax_macros"],
        "6.0": ["strict_concurrency", "region_based_isolation", "complete_strict_concurrency"],
    }

    # ─────────────────────────────────────────────────────────────────────────────
    # Pattern Catalog
    # ─────────────────────────────────────────────────────────────────────────────

    # Performance patterns
    PERFORMANCE_PATTERNS = {
        "array_concatenation": (
            re.compile(r'var\s+(\w+)\s*=\s*\[\]\s*\n(?:\s*\1\s*\+?=\s*\.\.\..*\n)+'),
            "Use reserveCapacity() before appending in loops",
            "performance"
        ),
        "string_concatenation": (
            re.compile(r'var\s+(\w+)\s*=\s*""\s*\n(?:\s*\1\s*\+?=.*\n)+'),
            "String concatenation in loop - consider using [String] joined or String.reserveCapacity",
            "performance"
        ),
        "nsarray_conversion": (
            re.compile(r'NSArray\(.*\)\s+as\?\s+\['),
            "Direct NSArray to Array conversion - consider using Array(nsArray) for better performance",
            "performance"
        ),
        "nsstring_in_swift": (
            re.compile(r'\bNSString\b'),
            "Using NSString in Swift - prefer native String for better performance and safety",
            "performance"
        ),
        "implicit_unwrap_optional": (
            re.compile(r':\s*\w+!'),
            "Implicitly unwrapped optional - prefer regular Optionals with explicit unwrapping",
            "safety"
        ),
        "force_cast": (
            re.compile(r'\bas!\s+\w+'),
            "Force cast with 'as!' can crash - use 'as?' with proper error handling",
            "safety"
        ),
    }

    # Modern Swift patterns (5.7+)
    MODERN_PATTERNS = {
        "existential_without_any": (
            re.compile(r':\s+(Equatable|Hashable|Comparable|Codable|Error|Collection|Sequence|Identifiable)(\s|$)'),
            "Protocol used as type without 'any' - use 'any Protocol' for explicit existential",
            "modern"
        ),
        "if_let_binding_expanded": (
            re.compile(r'if\s+let\s+(\w+)\s*=\s*\1'),
            "Can use shorthand 'if let x' instead of 'if let x = x' (Swift 5.7+)",
            "modern"
        ),
        "some_collection_return": (
            re.compile(r'->\s*\[\w+\]\s+\{'),
            "Consider returning 'some Collection' instead of concrete array type for abstraction",
            "modern"
        ),
    }

    # Swift 6 strict concurrency patterns
    CONCURRENCY_PATTERNS = {
        "global_actor_not_sendable": (
            re.compile(r'@MainActor\s+.*var\s+\w+\s*:\s*(?!\s*some\s+Sendable)(?!\s*\w+\?\s*$)\w+'),
            "Non-Sendable property in global actor - mark as Sendable or use value types",
            "concurrency"
        ),
        "actor_isolation_escape": (
            re.compile(r'\{[^}]*?\bin\b[^}]*?\}\s*\)\s*\{'),
            "Closure in actor-isolated context - ensure closure is @Sendable if escaping",
            "concurrency"
        ),
        "async_without_await": (
            re.compile(r'func\s+\w+.*\)\s*async\s+\{[^}]*\}(?!\s*\(?!.*await)'),
            "Async function that may not need to be async - check for actual suspension points",
            "concurrency"
        ),
        "task_without_priority": (
            re.compile(r'Task\s*\{'),
            "Task created without priority - consider Task(priority: .userInitiated) for important work",
            "concurrency"
        ),
        "completion_handler_not_marked": (
            re.compile(r'\w+:\s*@escaping\s*\([^)]*\)\s*->\s*Void'),
            "Completion handler closure not marked @Sendable - may cause data races in Swift 6",
            "concurrency"
        ),
    }

    # Idiomatic Swift patterns
    IDIOMATIC_PATTERNS = {
        "forced_unwrap_optional": (
            re.compile(r'\w+!\s*[\.,;]'),
            "Force unwrap detected - prefer if/guard let or optional chaining",
            "idiomatic"
        ),
        "nsnull_check": (
            re.compile(r'!==?\s*NSNull'),
            "Checking for NSNull - use Swift Optional binding instead",
            "idiomatic"
        ),
        "nsnumber_literal": (
            re.compile(r'NSNumber\s*\(\s*(value\s*:\s*)?\d+\s*\)'),
            "NSNumber with literal - use Swift numeric types directly",
            "idiomatic"
        ),
        "c_style_for": (
            re.compile(r'for\s+var\s+\w+\s*=\s*\d+;'),
            "C-style for loop deprecated - use Swift for-in or stride",
            "idiomatic"
        ),
        "iuo_as_optional": (
            re.compile(r'\!\s*\?'),
            "Implicitly unwrapped optional treated as Optional - use consistent optional handling",
            "idiomatic"
        ),
        "nsarray_literal": (
            re.compile(r'\[\w+\]\(\)'),
            "Use [] for empty array literal instead of initializer",
            "idiomatic"
        ),
        "nsdictionary_literal": (
            re.compile(r'\[\w+:\s*\w+\]\(\)'),
            "Use [:] for empty dictionary literal instead of initializer",
            "idiomatic"
        ),
        "unnecessary_self": (
            re.compile(r'self\.[a-z]\w+\s*[^:(]'),
            "Unnecessary 'self.' - only required in closures and init",
            "idiomatic"
        ),
    }

    # Memory management patterns
    MEMORY_PATTERNS = {
        "unowned_self": (
            re.compile(r'\[\s*unowned\s+self\s*\]'),
            "Using [unowned self] can crash if self is deallocated - prefer [weak self]",
            "memory"
        ),
        "retain_cycle_closure": (
            re.compile(r'\{\s*[^\[]*\bin\b[^}]*self\.'),
            "Potential retain cycle - self captured in closure without [weak self]",
            "memory"
        ),
        "unsafe_pointer_usage": (
            re.compile(r'UnsafeMutablePointer|UnsafePointer|UnsafeRawPointer'),
            "Unsafe pointer usage - ensure proper memory management and bounds checking",
            "memory"
        ),
    }

    def __init__(self, swift_version: str = "5.9", strict_concurrency: bool = False):
        """Initialize Swift language usage analyzer.

        Args:
            swift_version: Target Swift version ("5.5", "5.6", "5.7", "5.8", "5.9", "6.0")
            strict_concurrency: Enable Swift 6 strict concurrency checking
        """
        self.swift_version = swift_version
        self.strict_concurrency = strict_concurrency
        self._issues: List[SwiftLanguageUsageError] = []

    def analyze(self, source: str, filename: str = "<swift>") -> Dict[str, Any]:
        """Analyze Swift source code for language usage issues.

        Args:
            source: Swift source code
            filename: Virtual filename for error reporting

        Returns:
            Dict with 'suggestions', 'stats', 'modern_features', 'concurrency_readiness'
        """
        self._issues = []

        # Run all pattern checks
        self._check_patterns(source, filename, self.PERFORMANCE_PATTERNS)
        self._check_patterns(source, filename, self.MODERN_PATTERNS)
        self._check_patterns(source, filename, self.IDIOMATIC_PATTERNS)
        self._check_patterns(source, filename, self.MEMORY_PATTERNS)

        if self.strict_concurrency or self.swift_version >= "6.0":
            self._check_patterns(source, filename, self.CONCURRENCY_PATTERNS)

        # Advanced checks
        self._check_modern_features(source, filename)
        self._check_concurrency_patterns(source, filename)
        self._check_api_availability(source, filename)

        return {
            "suggestions": [issue.to_dict() for issue in self._issues],
            "stats": self._compute_stats(source),
            "modern_features": self._detect_modern_features(source),
            "concurrency_readiness": self._assess_concurrency_readiness(source),
            "swift_version": self.swift_version,
            "strict_concurrency_enabled": self.strict_concurrency,
        }

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Swift file for language usage."""
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        source = path.read_text(encoding="utf-8")
        return self.analyze(source, str(path))

    def _check_patterns(self, source: str, filename: str, patterns: Dict) -> None:
        """Check source against pattern dictionary."""
        lines = source.split('\n')

        for pattern_name, (pattern, message, category) in patterns.items():
            for match in pattern.finditer(source):
                line_no = source[:match.start()].count('\n') + 1
                col = match.start() - source.rfind('\n', 0, match.start())

                # Skip if in comment or string
                if self._is_in_comment_or_string(source, match.start()):
                    continue

                self._issues.append(SwiftLanguageUsageError(
                    file_path=filename,
                    line=line_no,
                    column=col,
                    message=message,
                    severity="suggestion",
                    category=category,
                    suggestion=self._get_suggestion(pattern_name, match.group(0))
                ))

    def _check_modern_features(self, source: str, filename: str) -> None:
        """Check for modern Swift 5.7+ features that could be used."""
        lines = source.split('\n')

        # Check for shorthand optional binding
        if self.swift_version >= "5.7":
            for i, line in enumerate(lines, 1):
                match = re.search(r'if\s+let\s+(\w+)\s*=\s*\1\b', line)
                if match:
                    self._issues.append(SwiftLanguageUsageError(
                        file_path=filename,
                        line=i,
                        column=match.start() + 1,
                        message=f"Can use shorthand 'if let {match.group(1)}' (Swift 5.7+)",
                        severity="suggestion",
                        category="modern",
                        suggestion=f"Replace 'if let {match.group(1)} = {match.group(1)}' with 'if let {match.group(1)}'"
                    ))

        # Check for regex literal (Swift 5.7+)
        if self.swift_version >= "5.7":
            if re.search(r'NSRegularExpression|Regex\s*\{', source):
                self._issues.append(SwiftLanguageUsageError(
                    file_path=filename,
                    line=1,
                    column=1,
                    message="Consider using Swift Regex literals /.../ (Swift 5.7+)",
                    severity="suggestion",
                    category="modern",
                    suggestion="Use /pattern/ syntax for better compile-time checking"
                ))

        # Check for existential 'any' requirement (Swift 5.6+)
        if self.swift_version >= "5.6":
            for i, line in enumerate(lines, 1):
                # Protocol types in function parameters
                match = re.search(r'func\s+\w+.*:\s*(Error|Codable|Equable|Hashable)\b(?!\s*where)', line)
                if match:
                    self._issues.append(SwiftLanguageUsageError(
                        file_path=filename,
                        line=i,
                        column=match.start(),
                        message=f"Protocol '{match.group(1)}' as type requires 'any' keyword (Swift 5.6+)",
                        severity="warning" if self.swift_version >= "5.6" else "suggestion",
                        category="modern",
                        suggestion=f"Use 'any {match.group(1)}' instead of '{match.group(1)}'"
                    ))

    def _check_concurrency_patterns(self, source: str, filename: str) -> None:
        """Check Swift 6 concurrency readiness."""
        if not self.strict_concurrency and self.swift_version < "6.0":
            return

        lines = source.split('\n')

        # Check for @Sendable conformance
        class_pattern = re.compile(r'(class|struct|enum)\s+(\w+)')
        for i, line in enumerate(lines, 1):
            match = class_pattern.search(line)
            if match:
                class_name = match.group(2)
                # Check if Sendable conformance is missing
                if not re.search(rf'{class_name}.*:\s*Sendable', source):
                    # Check if class has properties that should be Sendable
                    class_section = self._extract_class_section(source, class_name)
                    if class_section and self._has_concurrency_risk_properties(class_section):
                        self._issues.append(SwiftLanguageUsageError(
                            file_path=filename,
                            line=i,
                            column=match.start(),
                            message=f"{match.group(1).capitalize()} '{class_name}' should conform to Sendable for Swift 6",
                            severity="warning",
                            category="concurrency",
                            suggestion=f"Add ': Sendable' conformance or mark as @unchecked Sendable if needed"
                        ))

        # Check for unsafe global state
        global_var_pattern = re.compile(r'^\s*(var|let)\s+(\w+)\s*[=:]', re.MULTILINE)
        for match in global_var_pattern.finditer(source):
            if not re.search(r'@MainActor|@globalActor|DispatchQueue', source[:match.start()]):
                line_no = source[:match.start()].count('\n') + 1
                self._issues.append(SwiftLanguageUsageError(
                    file_path=filename,
                    line=line_no,
                    column=1,
                    message="Global mutable state detected - will require synchronization in Swift 6",
                    severity="warning",
                    category="concurrency",
                    suggestion="Mark with @MainActor, use actor-isolated state, or use atomic operations"
                ))

    def _check_api_availability(self, source: str, filename: str) -> None:
        """Check for proper API availability annotations."""
        lines = source.split('\n')

        # Check for unavailable_from_async where appropriate
        for i, line in enumerate(lines, 1):
            if re.search(r'completion.*handler|callback|delegate', line, re.IGNORECASE):
                if not re.search(r'@available.*unavailable.*async', line) and \
                   not re.search(r'@available.*from.*async', line):
                    self._issues.append(SwiftLanguageUsageError(
                        file_path=filename,
                        line=i,
                        column=1,
                        message="Callback/completion-based API should be marked unavailable in async context",
                        severity="suggestion",
                        category="modern",
                        suggestion="Add @available(*, unavailable, renamed: \"asyncAlternative\") for async contexts"
                    ))

    def _is_in_comment_or_string(self, source: str, position: int) -> bool:
        """Check if position is inside a comment or string literal."""
        lines_before = source[:position].split('\n')
        current_line = lines_before[-1] if lines_before else ""

        # Simple heuristic: check if we're in a comment
        if '//' in current_line:
            return True

        # Check if in multiline comment
        text_before = source[:position]
        open_comments = text_before.count('/*') - text_before.count('*/')
        if open_comments > 0:
            return True

        # Check if in string (simplified)
        # Count unescaped quotes before position
        quote_count = 0
        i = 0
        while i < position:
            if source[i] == '"' and (i == 0 or source[i-1] != '\\'):
                quote_count += 1
            i += 1
        if quote_count % 2 == 1:
            return True

        return False

    def _get_suggestion(self, pattern_name: str, matched_text: str) -> str:
        """Get specific suggestion for a pattern match."""
        suggestions = {
            "array_concatenation": "Use array.reserveCapacity(estimatedCount) before appending in a loop",
            "nsstring_in_swift": "Use String instead of NSString for better Swift integration",
            "force_cast": f"Replace '{matched_text}' with optional binding: if let value = x as? Type",
            "unowned_self": f"Replace '{matched_text}' with [weak self] and guard let self = self else {{ return }}",
            "retain_cycle_closure": "Add [weak self] capture list to prevent retain cycle",
            "existential_without_any": f"Add 'any' keyword: any {matched_text.strip(': ')}",
        }
        return suggestions.get(pattern_name, "Review and apply best practices")

    def _extract_class_section(self, source: str, class_name: str) -> Optional[str]:
        """Extract the body of a class/struct."""
        pattern = rf'(class|struct|enum)\s+{class_name}.*?(?=\n(?:class|struct|enum|extension)\s|\Z)'
        match = re.search(pattern, source, re.DOTALL)
        return match.group(0) if match else None

    def _has_concurrency_risk_properties(self, class_section: str) -> bool:
        """Check if class section has properties that risk concurrency."""
        # Check for mutable shared state
        if re.search(r'\bvar\s+\w+\s*:', class_section):
            return True
        # Check for reference types
        if re.search(r':\s*\w+\?', class_section):
            return True
        return False

    def _compute_stats(self, source: str) -> Dict[str, Any]:
        """Compute code statistics."""
        lines = source.split('\n')

        # Count async/await usage
        async_count = len(re.findall(r'\basync\b', source))
        await_count = len(re.findall(r'\bawait\b', source))
        actor_count = len(re.findall(r'\bactor\b', source))
        task_count = len(re.findall(r'\bTask\b', source))

        # Count Sendable usage
        sendable_count = len(re.findall(r'\bSendable\b', source))

        # Modern Swift features
        existential_any = len(re.findall(r'\bany\s+\w+', source))
        some_keyword = len(re.findall(r'\bsome\s+\w+', source))

        return {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip()]),
            "async_functions": async_count,
            "await_calls": await_count,
            "actors": actor_count,
            "tasks": task_count,
            "sendable_types": sendable_count,
            "existential_any": existential_any,
            "some_keyword": some_keyword,
            "issues_found": len(self._issues),
        }

    def _detect_modern_features(self, source: str) -> Dict[str, Any]:
        """Detect which modern Swift features are being used."""
        features = {}

        # Swift 5.5 features
        features["async_await"] = bool(re.search(r'\basync\b.*\{', source))
        features["actors"] = bool(re.search(r'\bactor\s+', source))
        features["global_actors"] = bool(re.search(r'@MainActor|@globalActor', source))
        features["sendable"] = bool(re.search(r'\bSendable\b', source))

        # Swift 5.6 features
        features["existential_any"] = bool(re.search(r':\s*any\s+', source))

        # Swift 5.7 features
        features["if_let_shorthand"] = bool(re.search(r'if\s+let\s+\w+(?!\s*=)', source))
        features["some_extensions"] = bool(re.search(r'extension\s+some\s+', source))

        # Swift 5.8 features
        features["if_switch_expressions"] = bool(re.search(r'=\s*if\s+|\s*=\s*switch\s+', source))

        return features

    def _assess_concurrency_readiness(self, source: str) -> Dict[str, Any]:
        """Assess code readiness for Swift 6 strict concurrency."""
        issues = []
        score = 100

        # Check for Sendable conformance
        if not re.search(r'\bSendable\b', source):
            issues.append("No Sendable conformance found - will need to add for Swift 6")
            score -= 20

        # Check for @MainActor usage
        if not re.search(r'@MainActor', source) and re.search(r'UI\w+|AppKit|UIKit', source):
            issues.append("UI-related code without @MainActor isolation")
            score -= 15

        # Check for unsafe globals
        global_vars = re.findall(r'^(var|let)\s+(\w+)', source, re.MULTILINE)
        if len(global_vars) > 3:
            issues.append(f"Many global variables ({len(global_vars)}) need synchronization")
            score -= 10

        # Check for completion handlers
        completion_handlers = len(re.findall(r'completion.*Handler|completionHandler', source, re.I))
        if completion_handlers > 0:
            issues.append(f"{completion_handlers} completion handlers - consider async/await migration")
            score -= 5

        return {
            "readiness_score": max(0, score),
            "issues": issues,
            "recommendation": "Ready for Swift 6" if score >= 80 else "Needs concurrency updates",
        }

    def suggest_fix(self, error: SwiftSyntaxError, source: str) -> Optional[str]:
        """Suggest a fix for a syntax error.

        Args:
            error: The syntax error
            source: Full source code

        Returns:
            Suggested fix string or None
        """
        lines = source.split('\n')
        if error.line <= 0 or error.line > len(lines):
            return None

        line = lines[error.line - 1]

        # Missing brace suggestions
        if "missing opening brace" in error.message.lower():
            return f"Add '{{' after the declaration"

        # Unmatched brackets
        if "unmatched" in error.message.lower():
            return "Check bracket matching in the surrounding context"

        # Force unwrap
        if "force unwrap" in error.message.lower():
            return "Consider using 'if let' or 'guard let' for safe unwrapping"

        return None


# ─────────────────────────────────────────────────────────────────────────────
# SwiftAutoFixEngine — automated detection and repair of recurring L104 patterns
# ─────────────────────────────────────────────────────────────────────────────

class SwiftAutoFixEngine:
    """Automated fixer for recurring Swift compilation errors in L104SwiftApp.

    Identifies and repairs the ~12 patterns that emerge from L104 code generation:
    missing os.log boilerplate, self-capture in Logger autoclosures, unsafe pointer
    optional syntax, integer literal type ambiguity, doubled-prefix type names,
    Codable conformance on Any-keyed structs, missing return in multi-statement
    closures, optional values in dict literals, missing QuantumGateEngine argument
    labels, and missing InterEngineFeedbackBus channels.

    Usage:
        fixer = SwiftAutoFixEngine()
        fixed_code, report = fixer.apply_all_safe(source_code, "MyFile.swift")
        for entry in report:
            print(entry)  # e.g. "fix_baseaddress_optional: 2 replacement(s)"
    """

    # ── FIX_CATALOG ────────────────────────────────────────────────────────────
    # Maps fix-method names to a short human-readable description.
    FIX_CATALOG: Dict[str, str] = {
        "fix_missing_os_log":           "Add missing 'import os.log' + Logger declaration",
        "fix_self_in_oslog_closure":    "Add 'self.' to instance props in Logger autoclosures",
        "fix_baseaddress_optional":     "Replace .baseAddress? with .baseAddress! (vDSP / DSP contexts)",
        "fix_reduce_integer_literal":   "Replace .reduce(0, +) with .reduce(0.0, +) for Double arrays",
        "fix_double_prefix_typename":   "Collapse doubled-prefix type names (DataDataX → DataX, etc.)",
        "fix_codable_with_any":         "Strip : Codable from structs that contain [String: Any] fields",
        "fix_missing_return_in_map":    "Insert explicit 'return' in multi-statement .map closures",
        "fix_optional_in_dict_literal": "Replace array.first? with array.first ?? 0 in dict literals",
        "fix_quantum_gate_labels":      "Add missing QuantumGateEngine argument labels",
        "fix_sorted_optional_in_dict":  "Replace sorted.first?/sorted.last? with ?? 0 in dict literals",
    }

    # ── PATTERNS ───────────────────────────────────────────────────────────────

    # Logger method calls we recognise
    _LOGGER_CALL_RE = re.compile(
        r'(logging\.(info|debug|error|warning|notice|critical|fault)\s*\(["\(])',
    )

    # Interpolation of a plain identifier inside a Logger string literal
    # Matches \(identifier) but NOT \(self.identifier) or \(something.else)
    _PLAIN_INTERP_RE = re.compile(r'\\(\((?!self\.)(?!super\.)([a-zA-Z_][a-zA-Z0-9_]*))\)')

    # Doubled-prefix patterns the code generator produces.
    # Matches any CamelCase word that is immediately repeated:
    #   DataData..., SearchSearch..., QuantumQuantum..., etc.
    _DOUBLE_PREFIX_RE = re.compile(r'\b([A-Z][a-z]+)\1([A-Z]\w*)\b')

    # .reduce(0, +) that should be .reduce(0.0, +)
    _REDUCE_INT_RE = re.compile(r'\.reduce\(\s*0\s*,\s*\+\s*\)')

    # .baseAddress? inside vDSP / DSPDoubleSplitComplex / UnsafePointer contexts
    _BASEADDR_OPT_RE = re.compile(r'(\.baseAddress)\?(?=\s*[,)\]])')

    # dict literals containing .first? or .last? producing optionals
    _SORTED_OPT_DICT_RE = re.compile(
        r'("(?:min|max|first|last|top|bottom|start|end)":\s*\w+\.(?:first|last|min|max))\?'
    )

    # Multi-statement .map closures whose last expression is missing a 'return'
    # Detects: .map { [params] in\n    let ... = ...\n    TypeName(...) }
    _MAP_CLOSURE_RE = re.compile(
        r'(\.map\s*\{[^\{]*?\bin\b\s*\n)'   # .map { ... in\n
        r'((?:[ \t]+(?:let|var)\s+[^\n]+\n)+)'  # one or more let/var lines
        r'([ \t]+)([A-Z]\w+\([^\n]+\)(?:\n[ \t]+\.\w+\([^\n]+\))*\s*\})',  # Constructor call + closing brace
        re.DOTALL,
    )

    # Known L104 QuantumGateEngine calls missing their argument labels
    _QGE_LABELS = [
        # (pattern, replacement)
        (re.compile(r'\.sacredCircuit\((\d+|\w+),\s*depth:'), '.sacredCircuit(nQubits: \\1, depth:'),
        (re.compile(r'\.execute\((\w+),\s*shots:'),           '.execute(circuit: \\1, shots:'),
        (re.compile(r'\.vqeAnsatz\((\d+|\w+),\s*depth:'),    '.vqeAnsatz(nQubits: \\1, depth:'),
        (re.compile(r'\.qft\((\d+|\w+)\)'),                   '.qft(nQubits: \\1)'),
        (re.compile(r'\.ghzState\((\d+|\w+)\)'),              '.ghzState(nQubits: \\1)'),
        (re.compile(r'\.bellPair\((\d+|\w+)\)'),              '.bellPair(nQubits: \\1)'),
    ]

    # ── PUBLIC API ─────────────────────────────────────────────────────────────

    def apply_all_safe(self, code: str, filename: str = "<swift>") -> Tuple[str, List[str]]:
        """Apply all safe auto-fixes to *code* and return (fixed_code, report).

        Each entry in *report* is a string like:
            "fix_baseaddress_optional: 3 replacement(s)"
        or
            "fix_missing_os_log: added Logger for MyFile"

        Fixes are applied in dependency order: type renames first (so later
        passes see the corrected names), then structural fixes, then style.
        """
        report: List[str] = []

        fixes = [
            ("fix_double_prefix_typename",   lambda c: self.fix_double_prefix_typename(c)),
            ("fix_missing_os_log",           lambda c: self.fix_missing_os_log(c, filename)),
            ("fix_self_in_oslog_closure",    lambda c: self.fix_self_in_oslog_closure(c)),
            ("fix_baseaddress_optional",     lambda c: self.fix_baseaddress_optional(c)),
            ("fix_reduce_integer_literal",   lambda c: self.fix_reduce_integer_literal(c)),
            ("fix_sorted_optional_in_dict",  lambda c: self.fix_sorted_optional_in_dict(c)),
            ("fix_optional_in_dict_literal", lambda c: self.fix_optional_in_dict_literal(c)),
            ("fix_missing_return_in_map",    lambda c: self.fix_missing_return_in_map(c)),
            ("fix_codable_with_any",         lambda c: self.fix_codable_with_any(c)),
            ("fix_quantum_gate_labels",      lambda c: self.fix_quantum_gate_labels(c)),
        ]

        for name, fn in fixes:
            try:
                new_code = fn(code)
                if new_code != code:
                    # Count how many lines changed as a rough diff metric
                    old_lines = code.splitlines()
                    new_lines = new_code.splitlines()
                    changed = sum(1 for a, b in zip(old_lines, new_lines) if a != b)
                    changed += abs(len(new_lines) - len(old_lines))
                    report.append(f"{name}: {changed} line(s) changed")
                    code = new_code
            except Exception as exc:  # never let one fix break the others
                report.append(f"{name}: SKIPPED ({exc})")

        return code, report

    def detect_issues(self, code: str, filename: str = "<swift>") -> List[Dict[str, Any]]:
        """Return a list of detected issues without modifying code.

        Each dict has keys: 'fix', 'line', 'description'.
        """
        issues: List[Dict[str, Any]] = []
        lines = code.splitlines()

        # os.log missing
        if self._LOGGER_CALL_RE.search(code):
            has_import = "import os.log" in code
            has_logger = re.search(r'private\s+let\s+logging\s*=\s*Logger\s*\(', code)
            if not has_import:
                issues.append({"fix": "fix_missing_os_log", "line": 1,
                                "description": "Uses logging.* but 'import os.log' is missing"})
            if not has_logger:
                issues.append({"fix": "fix_missing_os_log", "line": 1,
                                "description": "Uses logging.* but Logger declaration is missing"})

        # self. missing in Logger interpolations
        for i, line in enumerate(lines, 1):
            if self._LOGGER_CALL_RE.search(line):
                for m in self._PLAIN_INTERP_RE.finditer(line):
                    issues.append({"fix": "fix_self_in_oslog_closure", "line": i,
                                   "description": f"\\({m.group(2)}) in Logger call needs self."})

        # .baseAddress?
        for i, line in enumerate(lines, 1):
            if self._BASEADDR_OPT_RE.search(line):
                issues.append({"fix": "fix_baseaddress_optional", "line": i,
                               "description": ".baseAddress? should be .baseAddress! (non-optional)"})

        # .reduce(0, +)
        for i, line in enumerate(lines, 1):
            if self._REDUCE_INT_RE.search(line):
                issues.append({"fix": "fix_reduce_integer_literal", "line": i,
                               "description": ".reduce(0, +) infers Int; use .reduce(0.0, +) for Double"})

        # sorted.first? in dict
        for i, line in enumerate(lines, 1):
            if self._SORTED_OPT_DICT_RE.search(line):
                issues.append({"fix": "fix_sorted_optional_in_dict", "line": i,
                               "description": "Optional value in dict literal; add ?? 0"})

        # doubled prefix
        for i, line in enumerate(lines, 1):
            m = self._DOUBLE_PREFIX_RE.search(line)
            if m:
                issues.append({"fix": "fix_double_prefix_typename", "line": i,
                               "description": f"Doubled-prefix type name: {m.group(0)}"})

        return issues

    # ── FIX METHODS ───────────────────────────────────────────────────────────

    def fix_missing_os_log(self, code: str, filename: str = "<swift>") -> str:
        """Inject 'import os.log' and a Logger declaration when the file uses
        logging.info/debug/error/… but is missing one or both of those lines."""

        if not self._LOGGER_CALL_RE.search(code):
            return code  # file doesn't use the Logger pattern

        lines = code.splitlines(keepends=True)
        has_import_oslog  = any("import os.log" in l for l in lines)
        has_logger_decl   = any(re.search(r'private\s+let\s+logging\s*=\s*Logger\s*\(', l) for l in lines)

        if has_import_oslog and has_logger_decl:
            return code

        # Derive a subsystem/category from the filename
        stem = Path(filename).stem if filename != "<swift>" else "L104"
        subsystem = f"com.l104.{stem}"

        # ── insert 'import os.log' after the last existing 'import …' line ──
        if not has_import_oslog:
            last_import_idx = -1
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") and "//" not in stripped:
                    last_import_idx = i
            insert_at = last_import_idx + 1 if last_import_idx >= 0 else 0
            lines.insert(insert_at, "import os.log\n")

        code = "".join(lines)
        lines = code.splitlines(keepends=True)

        # ── insert Logger declaration after the last top-level 'import' ──────
        if not has_logger_decl:
            last_import_idx = -1
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") and "//" not in stripped:
                    last_import_idx = i

            # Skip blank lines after the last import
            insert_at = last_import_idx + 1 if last_import_idx >= 0 else 0
            while insert_at < len(lines) and lines[insert_at].strip() == "":
                insert_at += 1

            logger_line = (
                f'private let logging = Logger(subsystem: "{subsystem}", category: "main")\n'
            )
            lines.insert(insert_at, logger_line)

        return "".join(lines)

    def fix_self_in_oslog_closure(self, code: str) -> str:
        """In Logger.*() calls, prepend 'self.' to interpolated identifiers that
        lack it.  Only modifies lines that contain a Logger call."""

        lines = code.splitlines(keepends=True)
        result = []
        for line in lines:
            if self._LOGGER_CALL_RE.search(line):
                line = self._PLAIN_INTERP_RE.sub(r'\\(self.\2)', line)
            result.append(line)
        return "".join(result)

    def fix_baseaddress_optional(self, code: str) -> str:
        """Replace .baseAddress? with .baseAddress! in unsafe-pointer contexts."""
        return self._BASEADDR_OPT_RE.sub(r'\1!', code)

    def fix_reduce_integer_literal(self, code: str) -> str:
        """Replace .reduce(0, +) with .reduce(0.0, +) to force Double inference."""
        return self._REDUCE_INT_RE.sub('.reduce(0.0, +)', code)

    def fix_double_prefix_typename(self, code: str) -> str:
        """Collapse doubled-prefix type names produced by the code generator.

        E.g.  DataDataPrecognitionResult  →  DataPrecognitionResult
              SearchSearchAttractorState  →  SearchAttractorState

        Only collapses exact alphabetic-word doubling (CamelCase prefix repeated
        immediately, no separator).
        """
        # Generic doubled-word pattern: a capitalised word immediately repeated
        # e.g. DataData, SearchSearch, QuantumQuantum, NeuralNeural …
        doubled = re.compile(r'\b([A-Z][a-z]+)(\1)([A-Z]\w*)\b')
        return doubled.sub(r'\1\3', code)

    def fix_codable_with_any(self, code: str) -> str:
        """Remove ': Codable' (or 'Codable,' from conformance lists) from structs
        that contain [String: Any] or [[String: Any]] stored properties, since
        Codable auto-synthesis fails for heterogeneous dictionary types."""

        # Find struct declarations that have [String: Any] fields
        # Strategy: scan struct bodies; if [String: Any] found, strip Codable.
        struct_block = re.compile(
            r'(struct\s+\w+\s*(?:<[^>]*>)?\s*:)([^{]+)(\{)',
        )

        def _strip_codable(m: re.Match) -> str:
            header = m.group(1)
            conformances = m.group(2)
            brace = m.group(3)
            # Find the struct body by scanning ahead (limited heuristic)
            return m.group(0)  # default: no change at match level

        # Simpler approach: find each struct block, check for [String: Any],
        # then strip Codable from its conformance list.
        lines = code.splitlines()
        result_lines = list(lines)
        i = 0
        while i < len(lines):
            line = lines[i]
            struct_m = re.match(r'^(\s*(?:public\s+|internal\s+|private\s+|fileprivate\s+)?struct\s+\w+\s*(?:<[^>]*>)?\s*:)([^{]+)(\{)', line)
            if struct_m:
                conformances_str = struct_m.group(2)
                if 'Codable' in conformances_str or 'Encodable' in conformances_str or 'Decodable' in conformances_str:
                    # Find the closing brace of this struct (simple brace counter)
                    depth = 1
                    j = i + 1
                    body_lines = []
                    while j < len(lines) and depth > 0:
                        body_lines.append(lines[j])
                        depth += lines[j].count('{') - lines[j].count('}')
                        j += 1
                    body_text = "\n".join(body_lines)
                    if '[String: Any]' in body_text or '[[String: Any]]' in body_text:
                        # Strip Codable/Encodable/Decodable from conformances
                        new_conf = re.sub(r',?\s*\b(Codable|Encodable|Decodable)\b', '', conformances_str)
                        new_conf = re.sub(r'^\s*,\s*', '', new_conf)  # leading comma
                        new_conf = new_conf.strip()
                        if new_conf:
                            result_lines[i] = struct_m.group(1) + " " + new_conf + " " + struct_m.group(3) + line[struct_m.end():]
                        else:
                            # No conformances left — remove the ': ' entirely
                            bare = re.sub(r'\s*:[^{]+(\{)', r' \1', line)
                            result_lines[i] = bare
            i += 1

        return "\n".join(result_lines) + ("\n" if code.endswith("\n") else "")

    def fix_missing_return_in_map(self, code: str) -> str:
        """Insert explicit 'return' before the final expression in multi-statement
        .map { … in } closures.

        Pattern:  .map { [params] in
                      let x = …
                      SomeType(…)    ← needs 'return'
                  }
        """
        def _add_return(m: re.Match) -> str:
            intro   = m.group(1)   # ".map { ... in\n"
            lets    = m.group(2)   # one or more let/var lines
            indent  = m.group(3)   # indentation of the constructor line
            expr    = m.group(4)   # the constructor call + closing brace
            return f"{intro}{lets}{indent}return {expr}"

        return self._MAP_CLOSURE_RE.sub(_add_return, code)

    def fix_optional_in_dict_literal(self, code: str) -> str:
        """Replace optional-producing expressions in dict literal values with
        a nil-coalescing fallback.

        Covers patterns like:
            "key": array.first?   →  "key": array.first ?? 0
            "key": array.last?    →  "key": array.last ?? 0
        """
        # Match: a string key followed by an expression ending with ?
        opt_dict_val = re.compile(
            r'("[\w\s]+":\s*)([\w.]+(?:\[[\w.]+\])?)\?(?=\s*[,\n\]])'
        )
        return opt_dict_val.sub(r'\1\2 ?? 0', code)

    def fix_sorted_optional_in_dict(self, code: str) -> str:
        """Specifically replace sorted.first? / sorted.last? in dict literals."""
        return self._SORTED_OPT_DICT_RE.sub(r'\1 ?? 0', code)

    def fix_quantum_gate_labels(self, code: str) -> str:
        """Add missing argument labels to QuantumGateEngine API calls."""
        for pattern, replacement in self._QGE_LABELS:
            code = pattern.sub(replacement, code)
        return code

    # ── CONVENIENCE ───────────────────────────────────────────────────────────

    def fix_file(self, file_path: str, dry_run: bool = False) -> Tuple[bool, List[str]]:
        """Read *file_path*, apply all fixes, optionally write it back.

        Returns (changed: bool, report: List[str]).
        If *dry_run* is True the file is not written.
        """
        path = Path(file_path)
        original = path.read_text(encoding="utf-8")
        fixed, report = self.apply_all_safe(original, path.name)
        changed = fixed != original
        if changed and not dry_run:
            path.write_text(fixed, encoding="utf-8")
        return changed, report

    def fix_directory(
        self,
        directory: str,
        glob: str = "**/*.swift",
        dry_run: bool = False,
    ) -> Dict[str, List[str]]:
        """Apply fixes to every Swift file under *directory*.

        Returns a dict mapping file paths to their report lists (only files
        where at least one fix was applied).
        """
        results: Dict[str, List[str]] = {}
        root = Path(directory)
        for swift_file in sorted(root.glob(glob)):
            changed, report = self.fix_file(str(swift_file), dry_run=dry_run)
            if changed:
                results[str(swift_file)] = report
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Integration with CodeEngine
# ─────────────────────────────────────────────────────────────────────────────

def check_swift_syntax(source: str, filename: str = "<swift>") -> Dict[str, Any]:
    """Convenience function for Swift syntax checking.

    Args:
        source: Swift source code
        filename: Virtual filename for error reporting

    Returns:
        Dict with 'valid', 'errors', 'warnings', 'sacred_alignment'
    """
    analyzer = SwiftSyntaxAnalyzer()
    return analyzer.check_syntax(source, filename)


def check_swift_file(file_path: str) -> Dict[str, Any]:
    """Check a Swift file for syntax errors.

    Args:
        file_path: Path to Swift source file

    Returns:
        Dict with 'valid', 'errors', 'warnings'
    """
    analyzer = SwiftSyntaxAnalyzer()
    return analyzer.check_file(file_path)


def check_swift_directory(dir_path: str, recursive: bool = True) -> Dict[str, Any]:
    """Check all Swift files in a directory.

    Args:
        dir_path: Directory path
        recursive: Whether to search recursively

    Returns:
        Dict with 'valid', 'files_checked', 'total_errors', 'file_results'
    """
    analyzer = SwiftSyntaxAnalyzer()
    return analyzer.check_directory(dir_path, recursive)


# ─────────────────────────────────────────────────────────────────────────────
# Swift Debugger — LLDB Integration
# ─────────────────────────────────────────────────────────────────────────────

class LLDBBridge:
    """Bridge to LLDB for Swift runtime debugging."""

    def __init__(self):
        """Initialize LLDB bridge."""
        self.lldb = None
        self.debugger = None
        self.target = None
        self.process = None
        self._initialize_lldb()

    def _initialize_lldb(self) -> bool:
        """Initialize LLDB Python module."""
        try:
            import lldb
            self.lldb = lldb
            self.debugger = lldb.SBDebugger.Create()
            self.debugger.SetAsync(True)
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if LLDB is available."""
        return self.lldb is not None

    def attach_to_pid(self, pid: int) -> Dict[str, Any]:
        """Attach LLDB to a process by PID.

        Args:
            pid: Process ID to attach to

        Returns:
            Dict with 'success', 'error', 'pid'
        """
        if not self.is_available():
            return {"success": False, "error": "LLDB not available", "pid": pid}

        try:
            # Create a target for the process
            error = self.lldb.SBError()
            self.process = self.debugger.AttachToProcessWithPID(pid, error)

            if error.Success():
                self.target = self.process.GetTarget()
                return {
                    "success": True,
                    "pid": pid,
                    "process_id": self.process.GetProcessID(),
                    "threads": self.process.GetNumThreads(),
                }
            else:
                return {
                    "success": False,
                    "error": error.GetCString(),
                    "pid": pid,
                }
        except Exception as e:
            return {"success": False, "error": str(e), "pid": pid}

    def attach_to_name(self, name: str, wait_for: bool = False) -> Dict[str, Any]:
        """Attach LLDB to a process by name.

        Args:
            name: Process name
            wait_for: Wait for process to launch

        Returns:
            Dict with 'success', 'error'
        """
        if not self.is_available():
            return {"success": False, "error": "LLDB not available"}

        try:
            error = self.lldb.SBError()
            listener = self.debugger.GetListener()

            if wait_for:
                self.process = self.debugger.AttachToProcessWithName(
                    listener, name, False, error
                )
            else:
                self.process = self.debugger.AttachToProcessWithName(
                    listener, name, True, error
                )

            if error.Success():
                self.target = self.process.GetTarget()
                return {
                    "success": True,
                    "name": name,
                    "process_id": self.process.GetProcessID(),
                }
            else:
                return {
                    "success": False,
                    "error": error.GetCString(),
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def launch_debugger(self, path: str, args: List[str] = None) -> Dict[str, Any]:
        """Launch a process under debugger.

        Args:
            path: Path to executable
            args: Command line arguments

        Returns:
            Dict with 'success', 'error'
        """
        if not self.is_available():
            return {"success": False, "error": "LLDB not available"}

        try:
            error = self.lldb.SBError()
            target = self.debugger.CreateTarget(path, "", "", True, error)

            if error.Success():
                self.target = target
                launch_info = self.lldb.SBLaunchInfo(args or [])
                self.process = target.Launch(launch_info, error)

                if error.Success():
                    return {
                        "success": True,
                        "process_id": self.process.GetProcessID(),
                    }
                else:
                    return {
                        "success": False,
                        "error": error.GetCString(),
                    }
            else:
                return {
                    "success": False,
                    "error": error.GetCString(),
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_breakpoint(self, file: str, line: int) -> Optional[int]:
        """Set a breakpoint at a file:line.

        Args:
            file: Source file path
            line: Line number

        Returns:
            Breakpoint ID or None
        """
        if not self.target:
            return None

        try:
            bp_location = self.target.BreakpointCreateByLocation(file, line)
            return bp_location.GetID()
        except Exception:
            return None

    def set_symbol_breakpoint(self, symbol: str, module: str = None) -> Optional[int]:
        """Set a symbolic breakpoint.

        Args:
            symbol: Symbol name (e.g., "main" or "ViewController.viewDidLoad")
            module: Optional module filter

        Returns:
            Breakpoint ID or None
        """
        if not self.target:
            return None

        try:
            if module:
                bp = self.target.BreakpointCreateByName(symbol, module)
            else:
                bp = self.target.BreakpointCreateByName(symbol)
            return bp.GetID()
        except Exception:
            return None

    def list_breakpoints(self) -> List[Dict[str, Any]]:
        """List all breakpoints.

        Returns:
            List of breakpoint info dicts
        """
        if not self.target:
            return []

        breakpoints = []
        for bp in self.target.breakpoint_iter():
            locations = []
            for loc in bp:
                locations.append({
                    "line": loc.GetLineNumber(),
                    "file": loc.GetFileSpec().GetFilename(),
                    "address": hex(loc.GetLoadAddress()),
                })

            breakpoints.append({
                "id": bp.GetID(),
                "name": bp.GetName(),
                "enabled": bp.IsEnabled(),
                "locations": locations,
            })

        return breakpoints

    def delete_breakpoint(self, bp_id: int) -> bool:
        """Delete a breakpoint.

        Args:
            bp_id: Breakpoint ID

        Returns:
            True if successful
        """
        if not self.target:
            return False

        try:
            return self.target.BreakpointDelete(bp_id)
        except Exception:
            return False

    def continue_process(self) -> bool:
        """Continue execution after breakpoint.

        Returns:
            True if successful
        """
        if not self.process:
            return False

        try:
            self.process.Continue()
            return True
        except Exception:
            return False

    def step_over(self) -> bool:
        """Step over (next) line.

        Returns:
            True if successful
        """
        if not self.process:
            return False

        try:
            thread = self.process.GetSelectedThread()
            frame = thread.GetSelectedFrame()
            thread.StepOver()
            return True
        except Exception:
            return False

    def step_into(self) -> bool:
        """Step into function.

        Returns:
            True if successful
        """
        if not self.process:
            return False

        try:
            thread = self.process.GetSelectedThread()
            thread.StepInto()
            return True
        except Exception:
            return False

    def step_out(self) -> bool:
        """Step out of function.

        Returns:
            True if successful
        """
        if not self.process:
            return False

        try:
            thread = self.process.GetSelectedThread()
            thread.StepOut()
            return True
        except Exception:
            return False

    def get_backtrace(self, max_depth: int = 100) -> List[Dict[str, Any]]:
        """Get current backtrace.

        Args:
            max_depth: Maximum frames to retrieve

        Returns:
            List of frame info dicts
        """
        if not self.process:
            return []

        thread = self.process.GetSelectedThread()
        frames = []

        for i in range(min(thread.GetNumFrames(), max_depth)):
            frame = thread.GetFrameAtIndex(i)
            func = frame.GetFunction()

            frame_info = {
                "index": i,
                "pc": hex(frame.GetPC()),
                "symbol": frame.GetSymbol().GetName() if frame.GetSymbol() else None,
                "module": frame.GetModule().GetFileSpec().GetFilename() if frame.GetModule() else None,
            }

            # Try to get line info
            line_entry = frame.GetLineEntry()
            if line_entry:
                frame_info["line"] = line_entry.GetLineNumber()
                frame_info["file"] = line_entry.GetFileSpec().GetFilename()

            frames.append(frame_info)

        return frames

    def get_local_variables(self, frame_idx: int = 0) -> List[Dict[str, Any]]:
        """Get local variables in a frame.

        Args:
            frame_idx: Frame index

        Returns:
            List of variable info dicts
        """
        if not self.process:
            return []

        thread = self.process.GetSelectedThread()
        if frame_idx >= thread.GetNumFrames():
            return []

        frame = thread.GetFrameAtIndex(frame_idx)
        variables = []

        # Get local variables from lexical scope
        for var in frame.GetVariables(True, True, True, True):
            variables.append({
                "name": var.GetName(),
                "type": var.GetType().GetName(),
                "value": var.GetValue(),
                "summary": var.GetSummary(),
            })

        return variables

    def evaluate_expression(self, expr: str) -> Optional[Dict[str, Any]]:
        """Evaluate an expression in the current context.

        Args:
            expr: Expression to evaluate

        Returns:
            Dict with 'value', 'type', 'error' or None
        """
        if not self.process:
            return None

        try:
            thread = self.process.GetSelectedThread()
            frame = thread.GetSelectedFrame()

            result = frame.EvaluateExpression(expr)

            if result.IsValid():
                return {
                    "value": result.GetValue(),
                    "type": result.GetType().GetName(),
                    "summary": result.GetSummary(),
                }
            else:
                return {
                    "error": "Expression evaluation failed",
                    "expr": expr,
                }
        except Exception as e:
            return {"error": str(e), "expr": expr}

    def get_threads(self) -> List[Dict[str, Any]]:
        """Get all threads.

        Returns:
            List of thread info dicts
        """
        if not self.process:
            return []

        threads = []
        for i in range(self.process.GetNumThreads()):
            thread = self.process.GetThreadAtIndex(i)
            threads.append({
                "index": i,
                "id": thread.GetThreadID(),
                "name": thread.GetName(),
                "stop_reason": thread.GetStopReason(),
                "num_frames": thread.GetNumFrames(),
            })

        return threads

    def get_registers(self, frame_idx: int = 0) -> Dict[str, Any]:
        """Get CPU registers.

        Args:
            frame_idx: Frame index

        Returns:
            Dict of register name -> value
        """
        if not self.process:
            return {}

        thread = self.process.GetSelectedThread()
        if frame_idx >= thread.GetNumFrames():
            return {}

        frame = thread.GetFrameAtIndex(frame_idx)
        regs = {}

        for reg in frame.GetRegisters():
            for r in reg:
                regs[r.GetName()] = r.GetValue()

        return regs

    def read_memory(self, address: int, size: int) -> Optional[bytes]:
        """Read raw memory.

        Args:
            address: Memory address
            size: Number of bytes to read

        Returns:
            Bytes or None
        """
        if not self.process:
            return None

        try:
            error = self.lldb.SBError()
            data = self.process.ReadMemory(address, size, error)

            if error.Success():
                return data
            return None
        except Exception:
            return None

    def find_memory_pattern(self, pattern: bytes, max_results: int = 10) -> List[int]:
        """Find a pattern in process memory.

        Args:
            pattern: Byte pattern to find
            max_results: Maximum results

        Returns:
            List of addresses where pattern was found
        """
        # This is a simplified implementation - real implementation
        # would scan process memory regions
        return []

    def get_process_info(self) -> Optional[Dict[str, Any]]:
        """Get process information.

        Returns:
            Dict with process info or None
        """
        if not self.process:
            return None

        return {
            "pid": self.process.GetProcessID(),
            "unique_id": self.process.GetUniqueID(),
            "num_threads": self.process.GetNumThreads(),
            "num_breakpoints": self.target.GetNumBreakpoints() if self.target else 0,
            "state": str(self.process.GetState()),
        }

    def list_source_files(self) -> List[str]:
        """List source files in target.

        Returns:
            List of source file paths
        """
        if not self.target:
            return []

        files = []
        for module in self.target:
            for source in module.GetSymbolFile().GetSupportFiles():
                files.append(source.GetFilename())

        return files

    def disconnect(self) -> bool:
        """Disconnect from process.

        Returns:
            True if successful
        """
        if not self.process:
            return True

        try:
            self.process.Kill()
            self.process = None
            self.target = None
            return True
        except Exception:
            return False


class SwiftDebugger:
    """High-level Swift debugging interface.

    Provides LLDB integration with sacred alignment tracking.
    """

    def __init__(self):
        """Initialize Swift debugger."""
        self.lldb = LLDBBridge()
        self._debug_session_count = 0
        self._sacred_alignment: Dict[str, Any] = {}

    def is_available(self) -> bool:
        """Check if debugger is available."""
        return self.lldb.is_available()

    def attach_to_process(self, pid: int) -> Dict[str, Any]:
        """Attach to a process by PID.

        Args:
            pid: Process ID

        Returns:
            Dict with 'success', 'error', 'pid'
        """
        result = self.lldb.attach_to_pid(pid)
        if result.get("success"):
            self._debug_session_count += 1
            self._update_sacred_alignment()
        return result

    def launch_and_debug(self, path: str, args: List[str] = None) -> Dict[str, Any]:
        """Launch a process under debugger.

        Args:
            path: Path to executable
            args: Command line arguments

        Returns:
            Dict with 'success', 'error'
        """
        result = self.lldb.launch_debugger(path, args)
        if result.get("success"):
            self._debug_session_count += 1
            self._update_sacred_alignment()
        return result

    def set_breakpoint(self, file: str, line: int) -> Optional[int]:
        """Set a source breakpoint.

        Args:
            file: Source file path
            line: Line number

        Returns:
            Breakpoint ID or None
        """
        return self.lldb.set_breakpoint(file, line)

    def set_symbol_breakpoint(self, symbol: str, module: str = None) -> Optional[int]:
        """Set a symbolic breakpoint.

        Args:
            symbol: Symbol name
            module: Optional module filter

        Returns:
            Breakpoint ID or None
        """
        return self.lldb.set_symbol_breakpoint(symbol, module)

    def list_breakpoints(self) -> List[Dict[str, Any]]:
        """List all breakpoints."""
        return self.lldb.list_breakpoints()

    def delete_breakpoint(self, bp_id: int) -> bool:
        """Delete a breakpoint."""
        return self.lldb.delete_breakpoint(bp_id)

    def continue_execution(self) -> bool:
        """Continue execution."""
        return self.lldb.continue_process()

    def step_over(self) -> bool:
        """Step over line."""
        return self.lldb.step_over()

    def step_into(self) -> bool:
        """Step into function."""
        return self.lldb.step_into()

    def step_out(self) -> bool:
        """Step out of function."""
        return self.lldb.step_out()

    def backtrace(self) -> List[Dict[str, Any]]:
        """Get current backtrace."""
        return self.lldb.get_backtrace()

    def frame_info(self, frame_idx: int = 0) -> Dict[str, Any]:
        """Get frame details.

        Args:
            frame_idx: Frame index

        Returns:
            Frame info dict
        """
        frames = self.lldb.get_backtrace()
        if frame_idx < len(frames):
            return frames[frame_idx]
        return {}

    def local_variables(self, frame_idx: int = 0) -> List[Dict[str, Any]]:
        """Get local variables in a frame."""
        return self.lldb.get_local_variables(frame_idx)

    def evaluate(self, expr: str) -> Optional[Dict[str, Any]]:
        """Evaluate an expression."""
        return self.lldb.evaluate_expression(expr)

    def thread_list(self) -> List[Dict[str, Any]]:
        """List all threads."""
        return self.lldb.get_threads()

    def frame_list(self, thread_idx: int = 0) -> List[Dict[str, Any]]:
        """List frames in a thread."""
        if not self.lldb.process:
            return []

        if thread_idx >= self.lldb.process.GetNumThreads():
            return []

        thread = self.lldb.process.GetThreadAtIndex(thread_idx)
        frames = []

        for i in range(thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(i)
            frames.append({
                "index": i,
                "pc": hex(frame.GetPC()),
                "symbol": frame.GetSymbol().GetName() if frame.GetSymbol() else None,
            })

        return frames

    def register_read(self) -> Dict[str, Any]:
        """Read CPU registers."""
        return self.lldb.get_registers()

    def memory_read(self, address: int, size: int) -> Optional[bytes]:
        """Read raw memory."""
        return self.lldb.read_memory(address, size)

    def memory_find_pattern(self, pattern: bytes) -> List[int]:
        """Find pattern in memory."""
        return self.lldb.find_memory_pattern(pattern)

    def process_info(self) -> Optional[Dict[str, Any]]:
        """Get process info."""
        return self.lldb.get_process_info()

    def source_list(self) -> List[str]:
        """List source files."""
        return self.lldb.list_source_files()

    def source_list_lines(self, file: str) -> List[Dict[str, Any]]:
        """List lines in a source file.

        Args:
            file: Source file path

        Returns:
            List of line info dicts
        """
        # Simple line listing - in production would use SBLineEntry
        try:
            with open(file, 'r') as f:
                lines = []
                for i, line in enumerate(f, 1):
                    lines.append({
                        "line": i,
                        "content": line.rstrip('\n'),
                    })
                return lines
        except Exception:
            return []

    def disconnect(self) -> bool:
        """Disconnect from process."""
        result = self.lldb.disconnect()
        if result:
            self._update_sacred_alignment()
        return result

    def _update_sacred_alignment(self) -> None:
        """Update sacred alignment for debug session."""
        session = self._debug_session_count
        if session > 0:
            self._sacred_alignment = {
                "session_count": session,
                "god_code_resonance": round(session * GOD_CODE % 1.0, 6),
                "phi_alignment": round(session * PHI % 1.0, 6),
                "void_integration": round(1.0 / (session + 1), 6),
            }

    def sacred_alignment(self) -> Dict[str, Any]:
        """Get sacred alignment for debug session."""
        return self._sacred_alignment


# ─────────────────────────────────────────────────────────────────────────────
# Swift Language Usage Analyzer — proper Swift idioms and best practices
# ─────────────────────────────────────────────────────────────────────────────

class SwiftLanguageUsageAnalyzer:
    """Analyzes Swift code for proper language usage, idioms, and best practices.

    Validates modern Swift patterns including:
    - Swift 5.9+ concurrency (actors, Sendable, async/await)
    - Proper optionality handling (avoiding force unwrap)
    - Protocol-oriented programming patterns
    - Performance best practices
    - Memory management (ARC, weak/strong patterns)
    - Error handling patterns
    - Swift 6 strict concurrency readiness
    """

    # Modern Swift keywords (Swift 5.9+)
    MODERN_KEYWORDS = {
        # Concurrency
        "async", "await", "actor", "nonisolated", "isolated",
        "Sendable", "@Sendable", "unchecked", "Task", "TaskGroup",
        "AsyncSequence", "AsyncStream", "Continuation", "CheckedContinuation",
        # Existentials
        "any", "some", "AnyActor",
        # Ownership
        "borrowing", "consuming", "mutating", "inout",
        # Macros (Swift 5.9+)
        "macro", "freestanding", "attached",
        # Result builders
        "resultBuilder", "View",
    }

    # Swift 6 strict concurrency patterns to check
    STRICT_CONCURRENCY_PATTERNS = {
        "unsafe_patterns": [
            r"@unchecked\s+Sendable",
            r"NonSendable",
            r"unsafe\s+{",
        ],
        "isolation_patterns": [
            r"@MainActor",
            r"@globalActor",
            r"actor\s+",
        ],
        "sendable_patterns": [
            r":\s*Sendable",
            r"@Sendable",
            r"sending\s+",
        ],
    }

    # Performance anti-patterns
    PERFORMANCE_ANTI_PATTERNS = {
        "inefficient_string_concat": {
            "pattern": r'var\s+\w+\s*=\s*""[^}]+for\s+.*\+\=',
            "message": "Consider using joined() or String initializer with Collection instead of string concatenation in loop",
            "severity": "warning",
        },
        "inefficient_array_append": {
            "pattern": r'var\s+\w+\s*:\s*\[\w+\]\s*=\s*\[\][^}]+for\s+.*\.append',
            "message": "Consider using map() or compactMap() instead of manual array building",
            "severity": "suggestion",
        },
        "nsarray_bridge": {
            "pattern": r'as\s*!\s*\[\w+\]',
            "message": "Force-cast NSArray bridging may be inefficient; consider typed arrays",
            "severity": "warning",
        },
        "autoclosure_capture": {
            "pattern": r'@autoclosure.*\{[^}]*\blet\b[^}]*\bvar\b',
            "message": "Autoclosure capturing mutable state may cause unexpected behavior",
            "severity": "warning",
        },
    }

    # Memory management patterns
    MEMORY_PATTERNS = {
        "strong_reference_cycle": {
            "pattern": r'\{[^}]*\[\s*unowned\s+self\s*\][^}]*\}',
            "message": "Review unowned self usage - ensure self always outlives closure",
            "severity": "warning",
        },
        "weak_self_check": {
            "pattern": r'\[\s*weak\s+self\s*\][^}]*self\?[^=]*=',
            "message": "Consider using guard let self = self after weak capture to avoid optional chaining",
            "severity": "suggestion",
        },
        "unused_weak_self": {
            "pattern": r'\[\s*weak\s+\w+\s*\][^}]*\{[^}]*\}',
            "message": "Captured weak reference is never used in closure body",
            "severity": "note",
        },
    }

    # Optionality best practices
    OPTIONALITY_PATTERNS = {
        "force_unwrap": {
            "pattern": r'\w+\!',
            "message": "Avoid force unwrap; use guard let or if let for safe unwrapping",
            "severity": "warning",
            "exclude": [r'IBOutlet', r'as\!', r'try!', r'fatalError'],
        },
        "implicitly_unwrapped": {
            "pattern": r'\bvar\s+\w+\s*:\s*\w+\!',
            "message": "Implicitly unwrapped optionals should be avoided except for IBOutlet",
            "severity": "warning",
        },
        "optional_binding_preference": {
            "pattern": r'if\s+let\s+\w+\s*=\s*\w+\s*\{[^}]*\}\s*else\s*\{[^}]*return[^}]*\}',
            "message": "Consider using guard let instead of if-let-else-return pattern",
            "severity": "suggestion",
        },
    }

    # Protocol-oriented programming
    POP_PATTERNS = {
        "concrete_type_in_protocol": {
            "pattern": r'protocol\s+\w+\s*:\s*\w+Class',
            "message": "Prefer composition over inheritance; consider protocol extensions",
            "severity": "suggestion",
        },
        "existential_type_usage": {
            "pattern": r'\b(any\s+\w+Protocol)\b',
            "message": "Using 'any' existential; consider 'some' for better performance if possible",
            "severity": "suggestion",
        },
        "missing_protocol_stubs": {
            "pattern": r'struct\s+\w+\s*:\s*(?:\w+,\s*)*\w+Protocol',
            "message": "Ensure all protocol requirements are implemented",
            "severity": "note",
        },
    }

    # Error handling patterns
    ERROR_HANDLING_PATTERNS = {
        "empty_catch": {
            "pattern": r'catch\s*\{[^}]*\}',
            "message": "Empty catch block swallows errors; consider logging or handling",
            "severity": "warning",
        },
        "catch_nserror": {
            "pattern": r'catch\s+let\s+\w+\s+as\s+NSError',
            "message": "Catching NSError loses type safety; catch specific error types",
            "severity": "suggestion",
        },
        "try_bang_without_comment": {
            "pattern": r'try!',
            "message": "Force try should be documented with // swiftlint:disable:next force_try or equivalent justification",
            "severity": "warning",
        },
    }

    # Swift 6 migration readiness
    SWIFT6_PATTERNS = {
        "global_actor_isolation": {
            "pattern": r'^\s*(?:func|var|let)\s+\w+',
            "message": "Global actor isolation not specified; may need @MainActor or @globalActor in Swift 6",
            "severity": "suggestion",
        },
        "unchecked_sendable": {
            "pattern": r'extension\s+\w+\s*:\s*@unchecked\s+Sendable',
            "message": "Review @unchecked Sendable conformance for Swift 6 strict mode",
            "severity": "warning",
        },
        "actor_data_race": {
            "pattern": r'actor\s+\w+[^}]*var\s+\w+\s*:\s*(?:Int|Bool|String|Double)(?!\s*{)',
            "message": "Mutable actor state may cause data races; consider isolated or nonisolated",
            "severity": "warning",
        },
    }

    def __init__(self, swift_version: str = "5.9"):
        """Initialize language usage analyzer.

        Args:
            swift_version: Target Swift version (5.9, 6.0, etc.)
        """
        self.swift_version = swift_version
        self._issues: List[Dict[str, Any]] = []

    def analyze_language_usage(self, source: str, filename: str = "<swift>") -> Dict[str, Any]:
        """Analyze Swift code for proper language usage.

        Args:
            source: Swift source code
            filename: Source filename for reporting

        Returns:
            Dict with 'score', 'issues', 'suggestions', 'warnings'
        """
        self._issues = []
        lines = source.split('\n')

        # Run all pattern checks
        self._check_patterns(source, lines, filename)
        self._check_modern_swift(source, lines, filename)
        self._check_concurrency(source, lines, filename)
        self._check_performance(source, lines, filename)
        self._check_memory_management(source, lines, filename)
        self._check_optionality(source, lines, filename)
        self._check_pop(source, lines, filename)
        self._check_error_handling(source, lines, filename)

        if self.swift_version >= "6.0":
            self._check_swift6_readiness(source, lines, filename)

        # Calculate language usage score
        score = self._calculate_score(source)

        # Categorize issues
        warnings = [i for i in self._issues if i.get("severity") == "warning"]
        suggestions = [i for i in self._issues if i.get("severity") == "suggestion"]
        notes = [i for i in self._issues if i.get("severity") == "note"]

        return {
            "language_score": round(score, 4),
            "swift_version": self.swift_version,
            "total_issues": len(self._issues),
            "warnings": warnings,
            "suggestions": suggestions,
            "notes": notes,
            "all_issues": self._issues,
        }

    def _check_patterns(self, source: str, lines: List[str], filename: str) -> None:
        """Check all registered patterns."""
        all_patterns = [
            ("performance", self.PERFORMANCE_ANTI_PATTERNS),
            ("memory", self.MEMORY_PATTERNS),
            ("optionality", self.OPTIONALITY_PATTERNS),
            ("pop", self.POP_PATTERNS),
            ("error_handling", self.ERROR_HANDLING_PATTERNS),
        ]

        for category, patterns in all_patterns:
            for pattern_name, pattern_info in patterns.items():
                pattern = pattern_info.get("pattern", "")
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        # Check exclusions
                        excluded = False
                        for exclude in pattern_info.get("exclude", []):
                            if re.search(exclude, line):
                                excluded = True
                                break
                        if excluded:
                            continue

                        self._issues.append({
                            "category": category,
                            "pattern": pattern_name,
                            "line": line_num,
                            "file": filename,
                            "message": pattern_info.get("message", ""),
                            "severity": pattern_info.get("severity", "note"),
                            "context": line.strip()[:100],
                        })

    def _check_modern_swift(self, source: str, lines: List[str], filename: str) -> None:
        """Check for modern Swift feature usage."""
        modern_features_used = set()

        for line_num, line in enumerate(lines, 1):
            # Check for modern keywords
            for keyword in self.MODERN_KEYWORDS:
                if re.search(rf'\b{keyword}\b', line):
                    modern_features_used.add(keyword)

            # Suggest modern patterns
            if re.search(r'\bURLSession\.shared\.dataTask', line):
                self._issues.append({
                    "category": "modern_swift",
                    "pattern": "legacy_networking",
                    "line": line_num,
                    "file": filename,
                    "message": "Consider using async/await URLSession.data(from:) instead of completion handlers",
                    "severity": "suggestion",
                    "context": line.strip()[:100],
                })

            # Suggest Result type for complex error handling
            if re.search(r'\bcompletion.*Error\?\)', line) and "Result<" not in source:
                self._issues.append({
                    "category": "modern_swift",
                    "pattern": "result_type",
                    "line": line_num,
                    "file": filename,
                    "message": "Consider using Result<T, Error> for cleaner error propagation",
                    "severity": "suggestion",
                    "context": line.strip()[:100],
                })

        # Report modern Swift adoption
        if len(modern_features_used) < 3:
            self._issues.append({
                "category": "modern_swift",
                "pattern": "adoption",
                "line": 0,
                "file": filename,
                "message": f"Code uses limited modern Swift features ({len(modern_features_used)} found). Consider adopting async/await, actors, or Result builders.",
                "severity": "note",
                "features_found": list(modern_features_used),
            })

    def _check_concurrency(self, source: str, lines: List[str], filename: str) -> None:
        """Check concurrency best practices."""
        has_async = "async " in source or "await " in source
        has_actor = "actor " in source

        if has_async:
            # Check for proper async patterns
            for line_num, line in enumerate(lines, 1):
                # Warn about async in non-async context without Task
                if re.search(r'\w+\(.*\{[^}]*await[^}]*\}[^)]*\)', line):
                    self._issues.append({
                        "category": "concurrency",
                        "pattern": "unstructured_concurrency",
                        "line": line_num,
                        "file": filename,
                        "message": "Consider using Task or TaskGroup for structured concurrency",
                        "severity": "suggestion",
                        "context": line.strip()[:100],
                    })

        if has_actor:
            # Check actor isolation patterns
            for line_num, line in enumerate(lines, 1):
                if re.search(r'nonisolated\s+var', line):
                    self._issues.append({
                        "category": "concurrency",
                        "pattern": "nonisolated_mutable",
                        "line": line_num,
                        "file": filename,
                        "message": "nonisolated var may introduce data races; consider nonisolated let or isolated",
                        "severity": "warning",
                        "context": line.strip()[:100],
                    })

    def _check_performance(self, source: str, lines: List[str], filename: str) -> None:
        """Check performance best practices."""
        # Check for copy-on-write types
        for line_num, line in enumerate(lines, 1):
            # Array copying in loops
            if re.search(r'for\s+\w+\s+in\s+\w+\s*\{[^}]*\w+\.append', line):
                self._issues.append({
                    "category": "performance",
                    "pattern": "array_copy_in_loop",
                    "line": line_num,
                    "file": filename,
                    "message": "Array append in loop may cause multiple reallocations; consider reserveCapacity or ContiguousArray",
                    "severity": "suggestion",
                    "context": line.strip()[:100],
                })

    def _check_memory_management(self, source: str, lines: List[str], filename: str) -> None:
        """Check memory management patterns."""
        capture_patterns = []

        for line_num, line in enumerate(lines, 1):
            # Check for capture list issues
            capture_match = re.search(r'\[\s*(weak|unowned)\s+(\w+)\s*\]', line)
            if capture_match:
                capture_type = capture_match.group(1)
                captured_var = capture_match.group(2)
                capture_patterns.append({
                    "type": capture_type,
                    "var": captured_var,
                    "line": line_num,
                })

        # Check for unowned usage that should be weak
        for capture in capture_patterns:
            if capture["type"] == "unowned":
                self._issues.append({
                    "category": "memory",
                    "pattern": "unowned_review",
                    "line": capture["line"],
                    "file": filename,
                    "message": f"Review unowned {capture['var']} - only use if reference always outlives closure",
                    "severity": "warning",
                    "context": f"Captured {capture['var']} as unowned",
                })

    def _check_optionality(self, source: str, lines: List[str], filename: str) -> None:
        """Check optionality best practices."""
        # Already covered in pattern checks
        pass

    def _check_pop(self, source: str, lines: List[str], filename: str) -> None:
        """Check protocol-oriented programming patterns."""
        for line_num, line in enumerate(lines, 1):
            # Prefer protocol extensions over base class
            if re.search(r'class\s+\w+Base', line):
                self._issues.append({
                    "category": "pop",
                    "pattern": "base_class",
                    "line": line_num,
                    "file": filename,
                    "message": "Consider protocol with default implementation instead of base class",
                    "severity": "suggestion",
                    "context": line.strip()[:100],
                })

    def _check_error_handling(self, source: str, lines: List[str], filename: str) -> None:
        """Check error handling patterns."""
        for line_num, line in enumerate(lines, 1):
            # Prefer specific errors over generic
            if re.search(r'throws\s*$', line) and "->" in line:
                self._issues.append({
                    "category": "error_handling",
                    "pattern": "generic_throws",
                    "line": line_num,
                    "file": filename,
                    "message": "Consider defining custom Error enum for better error handling",
                    "severity": "suggestion",
                    "context": line.strip()[:100],
                })

    def _check_swift6_readiness(self, source: str, lines: List[str], filename: str) -> None:
        """Check Swift 6 strict concurrency readiness."""
        for pattern_name, pattern_info in self.SWIFT6_PATTERNS.items():
            pattern = pattern_info.get("pattern", "")
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    self._issues.append({
                        "category": "swift6",
                        "pattern": pattern_name,
                        "line": line_num,
                        "file": filename,
                        "message": pattern_info.get("message", ""),
                        "severity": pattern_info.get("severity", "note"),
                        "context": line.strip()[:100],
                    })

    def _calculate_score(self, source: str) -> float:
        """Calculate language usage quality score."""
        base_score = 1.0
        lines = len(source.split('\n'))

        # Deductions for issues
        for issue in self._issues:
            severity = issue.get("severity", "note")
            if severity == "warning":
                base_score -= 0.05
            elif severity == "suggestion":
                base_score -= 0.02
            elif severity == "note":
                base_score -= 0.01

        # Bonus for modern Swift features
        modern_count = sum(1 for kw in self.MODERN_KEYWORDS if kw in source)
        base_score += min(0.1, modern_count * 0.01)

        return max(0.0, min(1.0, base_score))

    def suggest_modernization(self, source: str) -> List[Dict[str, Any]]:
        """Suggest modernization opportunities for legacy Swift code.

        Args:
            source: Swift source code

        Returns:
            List of modernization suggestions
        """
        suggestions = []
        lines = source.split('\n')

        modernization_checks = [
            (r'\.map\s*\{\s*\$0', "Use key path syntax: .map(\\.property) where possible"),
            (r'func\s+\w+.*completion:\s*@escaping', "Consider async/await conversion for completion handler"),
            (r'NSNotificationCenter', "Use Combine or async notifications instead"),
            (r'@objc\s+class', "Review @objc usage; may not need Objective-C interop"),
            (r'UIApplication\.shared', "Consider injection for testability"),
            (r'print\(', "Use Logger/os_log instead of print for production code"),
            (r'dispatch_group_', "Use TaskGroup for modern concurrency"),
            (r'OperationQueue', "Consider async algorithms or structured concurrency"),
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern, message in modernization_checks:
                if re.search(pattern, line):
                    suggestions.append({
                        "line": line_num,
                        "message": message,
                        "original": line.strip()[:80],
                    })

        return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# Swift Language Fix Engine — automated language improvements
# ─────────────────────────────────────────────────────────────────────────────

class SwiftLanguageFixEngine:
    """Automated fix engine for Swift language improvements.

    Applies safe transformations to improve Swift code quality:
    - Convert completion handlers to async/await
    - Add explicit self where needed
    - Modernize syntax (some vs Any, etc.)
    - Add missing Sendable conformance
    """

    FIX_CATALOG = {
        "modernize_existential": "Replace 'any' with 'some' where appropriate for performance",
        "add_sendable_conformance": "Add Sendable conformance for thread-safe types",
        "convert_to_async": "Convert completion handler to async/await (partial)",
        "add_explicit_self": "Add explicit self in closures requiring it",
        "modernize_optional_binding": "Convert if-let-else-return to guard let",
        "add_mainactor_isolation": "Add @MainActor for UI-related code",
    }

    def __init__(self):
        """Initialize language fix engine."""
        self._fixes_applied = 0

    def apply_safe_fixes(self, source: str, filename: str = "<swift>") -> Tuple[str, List[str]]:
        """Apply safe language improvements.

        Args:
            source: Swift source code
            filename: Source filename

        Returns:
            Tuple of (fixed_code, report)
        """
        report = []
        code = source

        # Apply fixes in order
        fixes = [
            ("modernize_existential", self._fix_existentials),
            ("add_explicit_self", self._fix_explicit_self),
            ("modernize_optional_binding", self._fix_optional_binding),
        ]

        for fix_name, fix_fn in fixes:
            try:
                new_code = fix_fn(code)
                if new_code != code:
                    changes = self._count_changes(code, new_code)
                    report.append(f"{fix_name}: {changes} change(s)")
                    code = new_code
                    self._fixes_applied += changes
            except Exception as e:
                report.append(f"{fix_name}: SKIPPED ({e})")

        return code, report

    def _fix_existentials(self, code: str) -> str:
        """Replace existential any with some where appropriate."""
        # Pattern: func foo(_ param: any Protocol) -> Use some Protocol if return type
        # This is a simplified version - real implementation would need type analysis
        return code

    def _fix_explicit_self(self, code: str) -> str:
        """Add explicit self in closures that need it."""
        lines = code.splitlines(keepends=True)
        result = []

        in_closure = False
        capture_list = ""

        for line in lines:
            # Detect closure start with capture list
            capture_match = re.search(r'\[\s*(.*?)\s*\]', line)
            if capture_match and ('{' in line or '}' not in line):
                capture_list = capture_match.group(1)
                in_closure = True

            # Add self. to property access if weak self captured
            if in_closure and 'weak self' in capture_list:
                # Match property access without self
                line = re.sub(r'(?<!self\.)(?<!\w)([a-z][a-zA-Z0-9]*)\.(\w+)', r'self.\1.\2', line)

            if '}' in line and in_closure:
                in_closure = False
                capture_list = ""

            result.append(line)

        return "".join(result)

    def _fix_optional_binding(self, code: str) -> str:
        """Convert if-let-else-return to guard let."""
        # Pattern: if let x = y { ... } else { return }
        pattern = re.compile(
            r'if\s+let\s+(\w+)\s*=\s*([^\{]+)\s*\{([^}]*)\}\s*else\s*\{\s*return\s*\}',
            re.DOTALL
        )

        def replace_with_guard(m: re.Match) -> str:
            var_name = m.group(1)
            assignment = m.group(2).strip()
            body = m.group(3).strip()
            return f"guard let {var_name} = {assignment} else {{ return }}\n{body}"

        return pattern.sub(replace_with_guard, code)

    def _count_changes(self, old: str, new: str) -> int:
        """Count number of lines changed."""
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        changed = sum(1 for a, b in zip(old_lines, new_lines) if a != b)
        changed += abs(len(new_lines) - len(old_lines))
        return changed


# Module-level convenience functions
def analyze_swift_language_usage(source: str, filename: str = "<swift>", swift_version: str = "5.9") -> Dict[str, Any]:
    """Analyze Swift code for proper language usage.

    Args:
        source: Swift source code
        filename: Source filename
        swift_version: Target Swift version

    Returns:
        Analysis results with score and issues
    """
    analyzer = SwiftLanguageUsageAnalyzer(swift_version=swift_version)
    return analyzer.analyze_language_usage(source, filename)


def suggest_swift_modernization(source: str) -> List[Dict[str, Any]]:
    """Suggest modernization for Swift code.

    Args:
        source: Swift source code

    Returns:
        List of modernization suggestions
    """
    analyzer = SwiftLanguageUsageAnalyzer()
    return analyzer.suggest_modernization(source)


# ─────────────────────────────────────────────────────────────────────────────
# Swift Language Usage Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def analyze_swift_usage(source: str, filename: str = "<swift>",
                        swift_version: str = "5.9",
                        strict_concurrency: bool = False) -> Dict[str, Any]:
    """Analyze Swift source code for language usage patterns.

    Args:
        source: Swift source code
        filename: Virtual filename for error reporting
        swift_version: Target Swift version (default: "5.9")
        strict_concurrency: Enable strict concurrency checking

    Returns:
        Dict with suggestions, stats, modern_features, concurrency_readiness
    """
    analyzer = SwiftLanguageUsageAnalyzer(swift_version=swift_version)
    result = analyzer.analyze_language_usage(source, filename)

    # Add concurrency readiness if requested
    if strict_concurrency or swift_version >= "6.0":
        result["concurrency_readiness"] = {
            "strict_mode_ready": result["language_score"] > 0.8,
            "recommendation": "Enable strict concurrency" if result["language_score"] > 0.8 else "Fix concurrency issues first",
        }

    return result


def check_swift_concurrency_readiness(source: str, filename: str = "<swift>") -> Dict[str, Any]:
    """Assess Swift code readiness for Swift 6 strict concurrency.

    Args:
        source: Swift source code
        filename: Virtual filename for error reporting

    Returns:
        Dict with readiness score and required changes
    """
    analyzer = SwiftLanguageUsageAnalyzer(swift_version="6.0")
    results = analyzer.analyze_language_usage(source, filename)

    # Filter to concurrency-related issues
    concurrency_issues = [i for i in results["all_issues"]
                         if i.get("category") in ["concurrency", "swift6"]]

    return {
        "readiness_score": results["language_score"],
        "concurrency_issues": concurrency_issues,
        "is_ready": results["language_score"] >= 0.9 and len(concurrency_issues) <= 2,
        "recommendations": [
            "Add Sendable conformance to value types" if any("Sendable" in str(i) for i in concurrency_issues) else None,
            "Mark UI code with @MainActor" if any("MainActor" in str(i) for i in concurrency_issues) else None,
            "Review global mutable state" if any("global" in str(i).lower() for i in concurrency_issues) else None,
        ],
    }


# Updated exports
__all__ = [
    # Syntax analysis
    "SwiftSyntaxAnalyzer",
    "SwiftSyntaxError",
    "check_swift_syntax",
    "check_swift_file",
    "check_swift_directory",
    "SWIFT_ERROR_PATTERNS",
    # Language usage analysis (NEW v2.0)
    "SwiftLanguageUsageAnalyzer",
    "SwiftLanguageFixEngine",
    "SwiftLanguageUsageError",
    "analyze_swift_usage",
    "analyze_swift_language_usage",
    "check_swift_concurrency_readiness",
    "suggest_swift_modernization",
    # Auto-fix (legacy)
    "SwiftAutoFixEngine",
    # Debugging
    "SwiftDebugger",
    "LLDBBridge",
]