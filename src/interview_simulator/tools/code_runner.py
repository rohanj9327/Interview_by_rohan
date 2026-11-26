"""Run candidate code (Python / Java) against a problem's test cases.

LOCAL USE ONLY: code runs in a subprocess inside a temp dir with timeouts and memory
caps, which is fine on your own machine but is NOT a security sandbox. Replace with a
container-based runner (Docker etc.) before deploying for other users.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from interview_simulator.config import get_settings
from interview_simulator.models import CodingProblem, Language, RunResult, Signature, TestResult
from interview_simulator.tools.harness.java_gen import JAVA_TYPES, generate_main

HARNESS_DIR = Path(__file__).parent / "harness"
RESULT_RE = re.compile(r"^@@RESULT (.*)$", re.MULTILINE)

PY_TYPES = {
    "int": "int", "long": "int", "double": "float", "bool": "bool", "string": "str",
    "int[]": "list[int]", "double[]": "list[float]", "bool[]": "list[bool]", "string[]": "list[str]",
    "int[][]": "list[list[int]]", "string[][]": "list[list[str]]",
}

JAVA_DEFAULT_IMPORTS = "import java.util.*;\nimport java.util.stream.*;\n\n"


# --------------------------------------------------------------------------------------
# Starter code
# --------------------------------------------------------------------------------------


def starter_code(signature: Signature, language: Language) -> str:
    if language == "python":
        params = ", ".join(f"{p.name}: {PY_TYPES[p.type]}" for p in signature.params)
        return (
            "class Solution:\n"
            f"    def {signature.function_name}(self, {params}) -> {PY_TYPES[signature.return_type]}:\n"
            "        # Write your solution here\n"
            "        pass\n"
        )
    params = ", ".join(f"{JAVA_TYPES[p.type]} {p.name}" for p in signature.params)
    return (
        "import java.util.*;\n\n"
        "class Solution {\n"
        f"    public {JAVA_TYPES[signature.return_type]} {signature.function_name}({params}) {{\n"
        "        // Write your solution here\n"
        f"        {_java_default_return(signature.return_type)}\n"
        "    }\n"
        "}\n"
    )


def _java_default_return(typ: str) -> str:
    return {
        "int": "return 0;", "long": "return 0L;", "double": "return 0.0;", "bool": "return false;",
    }.get(typ, "return null;")


# --------------------------------------------------------------------------------------
# Output comparison
# --------------------------------------------------------------------------------------


def _close(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def outputs_match(expected, actual, typ: str) -> bool:
    if typ == "double":
        return _close(expected, actual)
    if typ == "double[]":
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_close(e, a) for e, a in zip(expected, actual))
        )
    if typ == "bool":
        return isinstance(actual, bool) and actual == expected
    return actual == expected


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


def _limit_memory():  # runs in the child process (POSIX only)
    try:
        import resource

        limit = get_settings().code_memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError, ImportError):
        pass  # macOS often ignores/refuses RLIMIT_AS; timeouts still apply


def _parse_results(stdout: str) -> dict[int, dict]:
    parsed = {}
    for match in RESULT_RE.finditer(stdout):
        try:
            item = json.loads(match.group(1))
            parsed[item["i"]] = item
        except (json.JSONDecodeError, KeyError):
            continue
    return parsed


def _strip_result_lines(stdout: str) -> str:
    return RESULT_RE.sub("", stdout).strip()


def _run_python(workdir: Path, code: str, spec: dict, timeout: float) -> tuple[subprocess.CompletedProcess | None, str | None]:
    (workdir / "solution.py").write_text(code)
    (workdir / "tests.json").write_text(json.dumps(spec))
    shutil.copy(HARNESS_DIR / "python_harness.py", workdir / "harness.py")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "harness.py"],
            cwd=workdir, capture_output=True, text=True, timeout=timeout, check=False,
            preexec_fn=_limit_memory, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        return None, exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    marker = "@@COMPILE_ERROR "
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return proc, "COMPILE:" + json.loads(line[len(marker):])
    return proc, None


def _run_java(workdir: Path, code: str, signature: Signature, tests: list[str], per_test: float, timeout: float):
    if not code.lstrip().startswith("package"):
        code = JAVA_DEFAULT_IMPORTS + code
    (workdir / "Solution.java").write_text(code)
    (workdir / "Main.java").write_text(generate_main(signature, tests, per_test))
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        return None, "COMPILE:Java (JDK) is not installed or not on PATH."
    comp = subprocess.run(
        [javac, "-nowarn", "-encoding", "UTF-8", "Solution.java", "Main.java"],
        cwd=workdir, capture_output=True, text=True, timeout=60, check=False,
    )
    if comp.returncode != 0:
        err = comp.stderr.replace(str(workdir) + "/", "")
        return None, "COMPILE:" + err.strip()
    mem = get_settings().code_memory_limit_mb
    try:
        proc = subprocess.run(
            [java, f"-Xmx{mem}m", "-Xss64m", "-cp", ".", "Main"],
            cwd=workdir, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return None, exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    return proc, None


def run_code(
    problem: CodingProblem,
    code: str,
    language: Language,
    include_hidden: bool = True,
) -> RunResult:
    """Run `code` against the problem's visible (+ hidden) tests."""
    settings = get_settings()
    per_test = float(settings.code_run_timeout_sec)
    tests = [(t, False) for t in problem.visible_tests]
    if include_hidden:
        tests += [(t, True) for t in problem.hidden_tests]
    args_list = [t.args_json for t, _ in tests]
    overall_timeout = per_test * len(tests) + 15

    with tempfile.TemporaryDirectory(prefix="interview_code_") as tmp:
        workdir = Path(tmp)
        if language == "python":
            spec = {
                "function_name": problem.signature.function_name,
                "params": [p.name for p in problem.signature.params],
                "timeout_sec": per_test,
                "tests": args_list,
            }
            proc, problem_text = _run_python(workdir, code, spec, overall_timeout)
        else:
            proc, problem_text = _run_java(
                workdir, code, problem.signature, args_list, per_test, overall_timeout
            )

    if problem_text and problem_text.startswith("COMPILE:"):
        return RunResult(language=language, compiled=False, compile_error=problem_text[len("COMPILE:"):])

    timed_out = proc is None
    stdout = problem_text if timed_out else proc.stdout
    stderr = "" if timed_out else proc.stderr
    parsed = _parse_results(stdout or "")

    results = []
    for i, (test, hidden) in enumerate(tests):
        item = parsed.get(i)
        expected = json.loads(test.expected_json)
        if item is None:
            results.append(TestResult(
                index=i, hidden=hidden, passed=False, args_json=test.args_json,
                expected_json=test.expected_json,
                error="Time limit exceeded" if timed_out else "No result (program crashed?)",
            ))
            continue
        output = item.get("output")
        passed = item.get("error") is None and outputs_match(expected, output, problem.signature.return_type)
        results.append(TestResult(
            index=i, hidden=hidden, passed=passed, args_json=test.args_json,
            expected_json=test.expected_json,
            output_json=json.dumps(output) if item.get("error") is None else None,
            error=item.get("error"), time_ms=round(item.get("ms", 0.0), 2),
        ))
    return RunResult(
        language=language, results=results, timed_out=timed_out,
        stderr=(stderr or "")[-2000:] + ("\n" + _strip_result_lines(stdout or ""))[-1000:] if stdout else stderr,
    )


def validate_problem(problem: CodingProblem) -> list[str]:
    """Check the problem's reference solutions pass all tests. Returns a list of errors."""
    errors = []
    try:
        for t in problem.visible_tests + problem.hidden_tests:
            args = json.loads(t.args_json)
            json.loads(t.expected_json)
            missing = [p.name for p in problem.signature.params if p.name not in args]
            if missing:
                errors.append(f"test args missing {missing}: {t.args_json}")
    except json.JSONDecodeError as exc:
        return [f"invalid test JSON: {exc}"]
    if errors:
        return errors
    if not problem.visible_tests or not problem.hidden_tests:
        return ["problem needs both visible and hidden tests"]
    for language, code in (("python", problem.reference_python), ("java", problem.reference_java)):
        result = run_code(problem, code, language)  # type: ignore[arg-type]
        if not result.compiled:
            errors.append(f"{language} reference does not compile: {result.compile_error[:500]}")
        elif not result.all_passed:
            failed = [r for r in result.results if not r.passed][:3]
            details = "; ".join(
                f"args={r.args_json} expected={r.expected_json} got={r.output_json} err={r.error}" for r in failed
            )
            errors.append(f"{language} reference fails {result.total - result.passed}/{result.total}: {details}")
    return errors
