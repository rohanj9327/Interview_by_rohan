"""Runs inside a temp dir next to solution.py and tests.json. Do not import app code here.

Prints one line per test:  @@RESULT {"i": 0, "output": <json>, "error": null, "ms": 1.2}
"""

import json
import os
import signal
import sys
import time
import traceback


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _to_jsonable(value):
    if isinstance(value, (tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def main():
    with open("tests.json") as fh:
        spec = json.load(fh)
    signal.signal(signal.SIGALRM, _on_alarm)
    sys.path.insert(0, os.getcwd())  # -I drops the script dir from sys.path
    try:
        import solution
        cls = solution.Solution
    except SyntaxError as exc:
        msg = "".join(traceback.format_exception_only(type(exc), exc)).replace(os.getcwd() + "/", "")
        print("@@COMPILE_ERROR " + json.dumps(msg))
        return
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, AttributeError):
            msg = "Your code must define `class Solution` with the required method."
        print("@@COMPILE_ERROR " + json.dumps(msg))
        return
    for i, args_json in enumerate(spec["tests"]):
        args = json.loads(args_json)
        positional = [args[name] for name in spec["params"]]
        out, err = None, None
        t0 = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, spec["timeout_sec"])
        try:
            result = getattr(cls(), spec["function_name"])(*positional)
            out = _to_jsonable(result)
        except _Timeout:
            err = "Time limit exceeded"
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        ms = (time.perf_counter() - t0) * 1000
        try:
            payload = json.dumps({"i": i, "output": out, "error": err, "ms": ms})
        except (TypeError, ValueError):
            payload = json.dumps({"i": i, "output": None, "error": f"Unserializable return value: {out!r}"[:300], "ms": ms})
        sys.stdout.flush()
        print("\n@@RESULT " + payload, flush=True)


if __name__ == "__main__":
    main()
