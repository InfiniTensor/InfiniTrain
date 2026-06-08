import argparse
import io
import os
import re
import tokenize
from pathlib import Path


CPP_SUFFIXES = {
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".mlu",
    ".cl",
}
PY_SUFFIXES = {".py"}
EXCLUDED_DIRS = {".git", "build", "cmake-build-debug", "cmake-build-release", "third_party"}
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
THREAD_LOCAL_RE = re.compile(r"\bthread_local\b.*?;", re.DOTALL)
TLS_NAME_RE = re.compile(r"(?:(?:[A-Za-z_]\w*)::)*(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*(?==|;|,|\{|\()")


def iter_files(paths):
    for path in paths:
        if path.is_file():
            yield path
            continue
        if not path.is_dir():
            print(f"error: {path} is not a file or directory")
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
            for filename in sorted(filenames):
                yield Path(dirpath) / filename


def has_cjk(text):
    return CJK_RE.search(text) is not None


def blank_like(text):
    chars = []
    for ch in text:
        chars.append("\n" if ch == "\n" else " ")
    return "".join(chars)


def scan_cpp(text):
    stripped = []
    comments = []
    line = 1
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if ch == "\n":
            stripped.append(ch)
            line += 1
            i += 1
            continue

        if ch == "R" and nxt == '"':
            match = re.match(r'R"([^\s\\()]*)\(', text[i:])
            if match:
                end_token = ")" + match.group(1) + '"'
                body_start = i + len(match.group(0))
                end = text.find(end_token, body_start)
                raw_end = n if end == -1 else end + len(end_token)
                segment = text[i:raw_end]
                stripped.append(blank_like(segment))
                line += segment.count("\n")
                i = raw_end
                continue

        if ch == "/" and nxt == "/":
            start_line = line
            i += 2
            stripped.append("  ")
            start = i
            while i < n and text[i] != "\n":
                stripped.append(" ")
                i += 1
            comments.append((start_line, text[start:i]))
            continue

        if ch == "/" and nxt == "*":
            start_line = line
            i += 2
            stripped.append("  ")
            comment = []
            while i < n:
                if i + 1 < n and text[i] == "*" and text[i + 1] == "/":
                    stripped.append("  ")
                    i += 2
                    break
                comment.append(text[i])
                if text[i] == "\n":
                    stripped.append("\n")
                    line += 1
                else:
                    stripped.append(" ")
                i += 1
            comments.append((start_line, "".join(comment)))
            continue

        if ch in {'"', "'"}:
            quote = ch
            stripped.append(" ")
            i += 1
            while i < n:
                current = text[i]
                if current == "\n":
                    stripped.append("\n")
                    line += 1
                    i += 1
                    break
                if current == "\\":
                    stripped.append(" ")
                    i += 1
                    if i < n:
                        if text[i] == "\n":
                            stripped.append("\n")
                            line += 1
                        else:
                            stripped.append(" ")
                        i += 1
                    continue
                stripped.append(" ")
                i += 1
                if current == quote:
                    break
            continue

        stripped.append(ch)
        i += 1

    return "".join(stripped), comments


def remove_template_args(statement):
    result = []
    depth = 0
    for ch in statement:
        if ch == "<":
            depth += 1
            result.append(" ")
        elif ch == ">" and depth > 0:
            depth -= 1
            result.append(" ")
        elif depth > 0:
            result.append("\n" if ch == "\n" else " ")
        else:
            result.append(ch)
    return "".join(result)


def declaration_prefix(statement):
    limit = len(statement)
    for token in ("=", "{"):
        pos = statement.find(token)
        if pos != -1:
            limit = min(limit, pos)
    return statement[:limit] + ";"


def check_thread_local_names(path, stripped_text):
    errors = []
    for match in THREAD_LOCAL_RE.finditer(stripped_text):
        statement = declaration_prefix(remove_template_args(match.group(0)))
        line = stripped_text.count("\n", 0, match.start()) + 1
        for name_match in TLS_NAME_RE.finditer(statement):
            name = name_match.group("name")
            if not name.startswith("tls_"):
                errors.append(f"{path}:{line}: thread_local variable '{name}' must use the tls_ prefix")
    return errors


def check_cpp_comments(path, comments):
    errors = []
    for line, comment in comments:
        if has_cjk(comment):
            errors.append(f"{path}:{line}: comments must be written in English")
    return errors


def check_python_comments(path, text):
    errors = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and has_cjk(token.string):
                errors.append(f"{path}:{token.start[0]}: comments must be written in English")
    except tokenize.TokenError as exc:
        errors.append(f"{path}: could not tokenize Python file: {exc}")
    return errors


def check_file(path):
    suffix = path.suffix
    if suffix not in CPP_SUFFIXES and suffix not in PY_SUFFIXES:
        return []

    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in CPP_SUFFIXES:
        stripped_text, comments = scan_cpp(text)
        return check_thread_local_names(path, stripped_text) + check_cpp_comments(path, comments)
    return check_python_comments(path, text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="+", type=Path, required=True, help="Files or directories to check.")
    args = parser.parse_args()

    errors = []
    for file in iter_files(args.path):
        errors.extend(check_file(file))

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print("Style check passed.")


if __name__ == "__main__":
    main()
