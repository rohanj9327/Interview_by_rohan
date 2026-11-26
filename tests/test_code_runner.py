import pytest

from interview_simulator.prompts.question_bank import FALLBACK_PROBLEMS, MERGE_INTERVALS, TWO_SUM
from interview_simulator.tools.code_runner import (
    outputs_match,
    run_code,
    starter_code,
    validate_problem,
)
from interview_simulator.tools.harness.java_gen import java_literal


@pytest.mark.parametrize("problem", FALLBACK_PROBLEMS, ids=lambda p: p.title)
def test_builtin_problems_validate(problem):
    assert validate_problem(problem) == []


def test_python_wrong_answer_and_exception():
    r = run_code(TWO_SUM, "class Solution:\n    def twoSum(self, nums, target):\n        return [0, 1]\n", "python")
    assert r.compiled and 0 < r.passed < r.total
    r = run_code(TWO_SUM, "class Solution:\n    def twoSum(self, nums, target):\n        return nums[99]\n", "python")
    assert r.passed == 0 and all("IndexError" in x.error for x in r.results)


def test_python_syntax_error():
    r = run_code(TWO_SUM, "class Solution:\n    def twoSum(self, nums, target)\n        return []\n", "python")
    assert not r.compiled and "SyntaxError" in r.compile_error


def test_python_prints_do_not_break_parsing():
    code = "class Solution:\n    def merge(self, intervals):\n        print('debug', intervals)\n        return [[1,6],[8,10],[15,18]]\n"
    r = run_code(MERGE_INTERVALS, code, "python", include_hidden=False)
    assert r.results[0].passed


def test_java_compile_error_and_wrong_answer():
    r = run_code(TWO_SUM, "class Solution { public int[] twoSum(int[] a, int t) { return 5 } }", "java")
    assert not r.compiled and "error" in r.compile_error
    r = run_code(TWO_SUM, "class Solution { public int[] twoSum(int[] a, int t) { return new int[]{0, 1}; } }", "java")
    assert r.compiled and 0 < r.passed < r.total


def test_python_timeout(monkeypatch):
    from interview_simulator.config import get_settings
    monkeypatch.setattr(get_settings(), "code_run_timeout_sec", 1)
    r = run_code(TWO_SUM, "class Solution:\n    def twoSum(self, n, t):\n        while True: pass\n", "python", include_hidden=False)
    assert r.passed == 0 and all(x.error == "Time limit exceeded" for x in r.results)


def test_starter_code_compiles_in_both_languages():
    for lang in ("python", "java"):
        r = run_code(MERGE_INTERVALS, starter_code(MERGE_INTERVALS.signature, lang), lang, include_hidden=False)
        assert r.compiled


def test_java_literals():
    assert java_literal([[1, 2], [3]], "int[][]") == "new int[][]{{1, 2}, {3}}"
    assert java_literal(['a"b'], "string[]") == 'new String[]{"a\\"b"}'
    assert java_literal(5, "long") == "5L"
    assert java_literal(True, "bool") == "true"


def test_outputs_match_doubles_and_bools():
    assert outputs_match(0.3, 0.1 + 0.2, "double")
    assert outputs_match([1.0, 2.0], [1.0000000001, 2.0], "double[]")
    assert not outputs_match(True, 1, "bool")
