"""Built-in fallbacks: common SWE questions and validated coding problems.

Used when web research is unavailable or when LLM problem generation fails validation.
"""

from interview_simulator.models import CodingProblem, Param, ResearchedQuestion, Signature, TestCase

COMMON_QUESTIONS: list[ResearchedQuestion] = [
    ResearchedQuestion(question=q, category=c, source_url="bank")
    for c, q in [
        ("behavioral", "Tell me about a time you disagreed with a teammate on a technical decision."),
        ("behavioral", "Describe a project that failed or went off track. What did you learn?"),
        ("behavioral", "Tell me about a time you had to deliver under a tight deadline."),
        ("behavioral", "Describe a time you took ownership of a problem outside your responsibilities."),
        ("fundamentals", "What happens when you type a URL into a browser and press enter?"),
        ("fundamentals", "Explain the difference between a process and a thread."),
        ("fundamentals", "How does a hash map work internally, and what is its worst-case complexity?"),
        ("fundamentals", "What are database indexes and when can they hurt performance?"),
        ("fundamentals", "Explain optimistic vs pessimistic locking."),
        ("system_design", "Design a URL shortener."),
        ("system_design", "Design a rate limiter for a public API."),
        ("system_design", "Design a notification service that sends email, SMS and push messages."),
        ("resume", "Walk me through the most technically challenging project on your resume."),
    ]
]


def _t(args: str, expected: str) -> TestCase:
    return TestCase(args_json=args, expected_json=expected)


TWO_SUM = CodingProblem(
    title="Pair With Target Sum",
    statement=(
        "Given an array of integers nums and an integer target, return the indices of the two "
        "numbers that add up to target, as [i, j] with i < j. Exactly one valid answer exists, "
        "and you may not use the same element twice."
    ),
    examples=["nums = [2, 7, 11, 15], target = 9 -> [0, 1] because 2 + 7 = 9"],
    constraints=["2 <= nums.length <= 10^5", "-10^9 <= nums[i], target <= 10^9"],
    signature=Signature(
        function_name="twoSum",
        params=[Param(name="nums", type="int[]"), Param(name="target", type="int")],
        return_type="int[]",
    ),
    visible_tests=[
        _t('{"nums": [2, 7, 11, 15], "target": 9}', "[0, 1]"),
        _t('{"nums": [3, 2, 4], "target": 6}', "[1, 2]"),
    ],
    hidden_tests=[
        _t('{"nums": [3, 3], "target": 6}', "[0, 1]"),
        _t('{"nums": [-1, -2, -3, -4, -5], "target": -8}', "[2, 4]"),
        _t('{"nums": [0, 4, 3, 0], "target": 0}', "[0, 3]"),
    ],
    reference_python=(
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        seen = {}\n"
        "        for i, n in enumerate(nums):\n"
        "            if target - n in seen:\n"
        "                return [seen[target - n], i]\n"
        "            seen[n] = i\n"
        "        return []\n"
    ),
    reference_java=(
        "class Solution {\n"
        "    public int[] twoSum(int[] nums, int target) {\n"
        "        Map<Integer, Integer> seen = new HashMap<>();\n"
        "        for (int i = 0; i < nums.length; i++) {\n"
        "            Integer j = seen.get(target - nums[i]);\n"
        "            if (j != null) return new int[]{j, i};\n"
        "            seen.put(nums[i], i);\n"
        "        }\n"
        "        return new int[0];\n"
        "    }\n"
        "}\n"
    ),
    expected_time_complexity="O(n)",
    expected_space_complexity="O(n)",
)

MERGE_INTERVALS = CodingProblem(
    title="Merge Overlapping Intervals",
    statement=(
        "Given a list of intervals where intervals[i] = [start, end], merge all overlapping "
        "intervals and return the merged intervals sorted by start. Intervals that touch "
        "(e.g. [1,4] and [4,5]) are considered overlapping."
    ),
    examples=["intervals = [[1,3],[2,6],[8,10],[15,18]] -> [[1,6],[8,10],[15,18]]"],
    constraints=["1 <= intervals.length <= 10^4", "0 <= start <= end <= 10^4"],
    signature=Signature(
        function_name="merge",
        params=[Param(name="intervals", type="int[][]")],
        return_type="int[][]",
    ),
    visible_tests=[
        _t('{"intervals": [[1,3],[2,6],[8,10],[15,18]]}', "[[1,6],[8,10],[15,18]]"),
        _t('{"intervals": [[1,4],[4,5]]}', "[[1,5]]"),
    ],
    hidden_tests=[
        _t('{"intervals": [[1,4]]}', "[[1,4]]"),
        _t('{"intervals": [[5,6],[1,2],[2,4]]}', "[[1,4],[5,6]]"),
        _t('{"intervals": [[1,10],[2,3],[4,5]]}', "[[1,10]]"),
    ],
    reference_python=(
        "class Solution:\n"
        "    def merge(self, intervals):\n"
        "        out = []\n"
        "        for s, e in sorted(intervals):\n"
        "            if out and s <= out[-1][1]:\n"
        "                out[-1][1] = max(out[-1][1], e)\n"
        "            else:\n"
        "                out.append([s, e])\n"
        "        return out\n"
    ),
    reference_java=(
        "class Solution {\n"
        "    public int[][] merge(int[][] intervals) {\n"
        "        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));\n"
        "        List<int[]> out = new ArrayList<>();\n"
        "        for (int[] cur : intervals) {\n"
        "            if (!out.isEmpty() && cur[0] <= out.get(out.size() - 1)[1]) {\n"
        "                int[] last = out.get(out.size() - 1);\n"
        "                last[1] = Math.max(last[1], cur[1]);\n"
        "            } else {\n"
        "                out.add(new int[]{cur[0], cur[1]});\n"
        "            }\n"
        "        }\n"
        "        return out.toArray(new int[0][]);\n"
        "    }\n"
        "}\n"
    ),
    expected_time_complexity="O(n log n)",
    expected_space_complexity="O(n)",
)

FALLBACK_PROBLEMS: list[CodingProblem] = [TWO_SUM, MERGE_INTERVALS]
