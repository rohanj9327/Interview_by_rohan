"""Generate a Java test driver (Main.java) with test inputs embedded as Java literals.

Embedding literals avoids needing a JSON parser on the Java side. Results are printed
as JSON lines prefixed with @@RESULT and compared in Python.
"""

from __future__ import annotations

import json

from interview_simulator.models import Signature

JAVA_TYPES = {
    "int": "int", "long": "long", "double": "double", "bool": "boolean", "string": "String",
    "int[]": "int[]", "double[]": "double[]", "bool[]": "boolean[]", "string[]": "String[]",
    "int[][]": "int[][]", "string[][]": "String[][]",
}


def java_literal(value, typ: str) -> str:
    if typ == "int":
        return str(int(value))
    if typ == "long":
        return f"{int(value)}L"
    if typ == "double":
        return repr(float(value))
    if typ == "bool":
        return "true" if value else "false"
    if typ == "string":
        return json.dumps(value)  # JSON string escapes are valid Java escapes
    if typ.endswith("[][]"):
        inner = typ[:-2]
        rows = ", ".join("{" + ", ".join(java_literal(v, inner[:-2]) for v in row) + "}" for row in value)
        return f"new {JAVA_TYPES[typ]}{{{rows}}}"
    if typ.endswith("[]"):
        elem = typ[:-2]
        return f"new {JAVA_TYPES[typ]}{{{', '.join(java_literal(v, elem) for v in value)}}}"
    raise ValueError(f"Unsupported type {typ}")


_HELPERS = r"""
    static String str(String s) {
        if (s == null) return "null";
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c)); else b.append(c);
            }
        }
        return b.append('"').toString();
    }
    static String toJson(int v) { return Integer.toString(v); }
    static String toJson(long v) { return Long.toString(v); }
    static String toJson(double v) { return Double.isFinite(v) ? Double.toString(v) : "null"; }
    static String toJson(boolean v) { return v ? "true" : "false"; }
    static String toJson(String v) { return str(v); }
    static String toJson(int[] a) { if (a == null) return "null"; StringJoiner j = new StringJoiner(",", "[", "]"); for (int v : a) j.add(toJson(v)); return j.toString(); }
    static String toJson(long[] a) { if (a == null) return "null"; StringJoiner j = new StringJoiner(",", "[", "]"); for (long v : a) j.add(toJson(v)); return j.toString(); }
    static String toJson(double[] a) { if (a == null) return "null"; StringJoiner j = new StringJoiner(",", "[", "]"); for (double v : a) j.add(toJson(v)); return j.toString(); }
    static String toJson(boolean[] a) { if (a == null) return "null"; StringJoiner j = new StringJoiner(",", "[", "]"); for (boolean v : a) j.add(toJson(v)); return j.toString(); }
    static String toJson(String[] a) { if (a == null) return "null"; StringJoiner j = new StringJoiner(",", "[", "]"); for (String v : a) j.add(toJson(v)); return j.toString(); }
    static String toJson(int[][] a) { if (a == null) return "null"; StringJoiner j = new StringJoiner(",", "[", "]"); for (int[] v : a) j.add(toJson(v)); return j.toString(); }
    static String toJson(String[][] a) { if (a == null) return "null"; StringJoiner j = new StringJoiner(",", "[", "]"); for (String[] v : a) j.add(toJson(v)); return j.toString(); }

    static void run(int i, long timeoutMs, Callable<String> task) {
        ExecutorService ex = Executors.newSingleThreadExecutor(r -> { Thread t = new Thread(r); t.setDaemon(true); return t; });
        long t0 = System.nanoTime();
        String out = null, err = null;
        Future<String> f = ex.submit(task);
        try {
            out = f.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            f.cancel(true); err = "Time limit exceeded";
        } catch (ExecutionException e) {
            Throwable c = e.getCause();
            err = c.getClass().getSimpleName() + (c.getMessage() != null ? ": " + c.getMessage() : "");
        } catch (Throwable e) {
            err = e.toString();
        } finally {
            ex.shutdownNow();
        }
        double ms = (System.nanoTime() - t0) / 1e6;
        System.out.flush();
        System.out.println("\n@@RESULT {\"i\":" + i + ",\"output\":" + (out == null ? "null" : out)
            + ",\"error\":" + str(err) + ",\"ms\":" + ms + "}");
        System.out.flush();
    }
"""


def generate_main(signature: Signature, tests_args_json: list[str], timeout_sec: float) -> str:
    calls = []
    for i, args_json in enumerate(tests_args_json):
        args = json.loads(args_json)
        literals = ", ".join(java_literal(args[p.name], p.type) for p in signature.params)
        calls.append(
            f"        run({i}, {int(timeout_sec * 1000)}L, () -> toJson(new Solution()."
            f"{signature.function_name}({literals})));"
        )
    body = "\n".join(calls)
    return (
        "import java.util.*;\nimport java.util.concurrent.*;\n\n"
        "public class Main {\n"
        "    public static void main(String[] args) {\n"
        f"{body}\n"
        "        System.exit(0);\n"
        "    }\n"
        f"{_HELPERS}"
        "}\n"
    )
