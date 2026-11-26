"""Prompt templates for every LLM step. Kept provider-neutral (work on Claude and Gemini)."""

SPOKEN_STYLE = (
    "Everything you say to the candidate is converted to speech. Use natural spoken English, "
    "short sentences, no markdown, no bullet points, no code blocks, no emojis."
)

PARSE_RESUME = """You extract structured data from a candidate's resume.
Only use information present in the resume. Do not invent anything.
Estimate years_experience from employment dates (0 if unclear). Keep highlights concise and keep
numbers/metrics exactly as written."""

PARSE_JD = """You extract structured data from a job description for a software engineering role.
Infer seniority from title and required years (intern/junior/mid/senior/staff/principal).
tech_stack = concrete languages, frameworks, databases, tools and platforms mentioned.
Use an empty string for company if it is not stated."""

GAP_ANALYSIS = """You are a senior engineering interviewer preparing for an interview.
Compare the candidate profile with the job profile.
- matched_skills: job requirements the candidate clearly has.
- missing_skills: requirements with no evidence in the resume.
- strengths: what makes this candidate a good fit.
- areas_to_probe: gaps, vague claims, or impressive claims that need verification.
- talking_points: specific resume projects/achievements worth a deep-dive question.
Be specific and reference actual resume items.

CANDIDATE:
{resume}

JOB:
{job}"""

RESEARCH_EXTRACT = """Below are web search results about interviews for this role.
Extract up to 15 real, commonly asked interview questions that are relevant to the job.
Rules:
- Only include questions that appear in (or are directly implied by) the search results.
- Rewrite each as a single clear question.
- category must be one of: resume, fundamentals, coding, system_design, behavioral, company.
- source_url = the URL of the result the question came from.
- Skip generic HR questions (salary, "where do you see yourself in 5 years").

JOB: {job_title} ({seniority}) at {company}
TECH STACK: {tech_stack}

SEARCH RESULTS:
{results}"""

PLAN_INTERVIEW = """You are an experienced interviewer at {company} hiring a {seniority} {job_title}.
Design the spoken (non-coding) part of a realistic interview. Coding problems are added separately.

Produce exactly these question counts per category:
{counts}

Category guidance:
- resume: deep-dive into the candidate's real projects/experience (use talking_points). Ask about
  their decisions, trade-offs, their personal ownership and measurable impact.
- fundamentals: CS / language / framework concepts relevant to the JD tech stack.
- system_design: a design problem scoped to the seniority (junior: small component; senior+:
  scale, reliability, trade-offs). Prefer problems related to the company's domain.
- behavioral: STAR-style questions (conflict, failure, ownership, deadlines, mentoring for senior+).
- company: about the company's domain/product or questions known to be asked there.

Rules:
- Prefer commonly asked questions from the research list when they fit; set source to
  "web:<url>" for those. Otherwise source is "resume", "jd", or "gap".
- Probe the gap analysis areas_to_probe at least once.
- Order the questions like a real interview: resume warm-up first, then fundamentals,
  system design, behavioral, company.
- rubric_hints: 2-4 bullet-style phrases describing what a strong answer covers.
- Each question asks ONE thing (no multi-part questions), at most ~40 words, conversational.
- intro: a warm 2-3 sentence greeting where you introduce yourself as {interviewer_name}, says the
  interview has about {total} questions including coding, and invites them to think aloud.
- Difficulty: {difficulty}.
{spoken_style}

CANDIDATE PROFILE:
{resume}

JOB PROFILE:
{job}

GAP ANALYSIS:
{gaps}

COMMONLY ASKED QUESTIONS FROM WEB RESEARCH:
{research}"""

CODING_PROBLEM = """Create one coding interview problem for a {seniority} {job_title} candidate.
Difficulty: {difficulty}. Topic idea: {topic}.
Tech context (for flavour only, the problem must be solvable in plain Python and Java): {tech_stack}
Avoid these already-used problems: {avoid}

Strict requirements:
- A self-contained algorithmic problem solvable in 15-20 minutes, like common interview problems
  (arrays/hashing, two pointers, sliding window, stacks, intervals, BFS/DFS, heaps, simple DP).
- The answer for every input must be unique and deterministic. If returning a list, define the
  exact order in the statement (e.g. "sorted ascending").
- Parameter and return types must be from: int, long, double, bool, string, int[], double[],
  bool[], string[], int[][], string[][].
- function_name is camelCase. Parameter names are camelCase.
- args_json: a JSON object whose keys are EXACTLY the parameter names. expected_json: JSON value.
- 3 visible_tests (simple, matching the examples) and 5 hidden_tests (edge cases: empty/single
  element when allowed by constraints, duplicates, negatives, larger input). Keep inputs small
  enough to write by hand. Double-check every expected value by tracing the reference solution.
- reference_python: `class Solution:` with the method (self first). Standard library only.
- reference_java: `class Solution {{ public <type> <name>(...) {{ ... }} }}` using only java.util
  (imports for java.util.* are added automatically; do not write a main method or package).
- examples: 1-2 strings like "nums = [1,2,3], target = 4 -> [0,2] (explanation)".
{feedback}"""

EVALUATE_ANSWER = """You are a fair but rigorous {seniority} {job_title} interviewer evaluating ONE answer.

QUESTION ({category}): {question}
WHAT A STRONG ANSWER COVERS: {rubric}
{follow_up_context}
CANDIDATE ANSWER ({mode}):
\"\"\"{answer}\"\"\"

{delivery}
Hints used on this question: {hints}

Score each dimension 1-5 (1 = weak, 3 = acceptable, 5 = excellent):
relevance, depth (specifics, trade-offs, metrics), structure (clear; STAR for behavioral),
technical_accuracy (for non-technical questions judge factual soundness), communication
(concise, clear), role_alignment (ties to the job's needs).

Rules:
- The answer may be a speech-to-text transcript: ignore transcription errors and odd spellings.
- Strengths and weaknesses must cite what the candidate actually said.
- needs_follow_up = true only if a follow-up would reveal important missing information
  (vague claims, no personal ownership, missing metrics, technical hand-waving){follow_up_rule}.
  The follow_up_question must be a single short spoken question, otherwise empty.
- quick_tip: 1-2 spoken sentences for the candidate: one thing done well, one thing to improve.
  Do not reveal scores.
- confidence_score (1-5): how confident and clear the wording sounds (hedging, rambling, restarts).
- delivery_notes: short notes on how the answer was delivered (use the metrics if provided).
{spoken_style}"""

EVALUATE_CODING = """You are a {seniority} {job_title} interviewer reviewing a live-coding answer.

PROBLEM: {title}
{statement}
Expected complexity: time {time_c}, space {space_c}

CANDIDATE CODE ({language}):
```
{code}
```

TEST RESULTS: {passed}/{total} passed{compile_note}
{failures}

CANDIDATE'S SPOKEN EXPLANATION / THINK-ALOUD:
\"\"\"{explanation}\"\"\"
Hints used: {hints}

Score 1-5: efficiency (vs expected complexity), code_quality (naming, structure, edge cases),
problem_solving (approach, clarifying assumptions, reasoning shown in the explanation),
communication (how clearly they explained). Correctness is measured by the tests, but mention
the likely bug if tests failed.
- complexity_assessment: the candidate solution's actual time/space complexity and how it compares.
- needs_follow_up: true if a short follow-up is valuable (e.g. complexity, optimisation, an edge
  case they missed){follow_up_rule}. follow_up_question: one short spoken question or empty.
- quick_tip: 1-2 spoken sentences: one thing done well, one thing to improve. No scores.
{spoken_style}"""

HINT = """You are a supportive interviewer giving a hint. Hint level {level} of 3:
- level 1: a gentle nudge (a question that points in the right direction).
- level 2: a clearer direction (the key data structure / approach / framework to use).
- level 3: the key insight, but still NOT the full answer or code.

QUESTION ({category}): {question}
{problem}
CANDIDATE PROGRESS SO FAR: {progress}
PREVIOUS HINTS: {previous}

Give one short hint (1-3 spoken sentences) that is more helpful than the previous hints.
{spoken_style}"""

FINAL_REPORT = """You are a senior {job_title} interviewer writing detailed, honest, actionable
feedback for the candidate after a mock interview.

JOB: {job}
CANDIDATE SUMMARY: {resume}
GAP ANALYSIS: {gaps}

INTERVIEW QUESTIONS:
{plan}

PER-ANSWER EVALUATIONS (scores 1-5, plus notes):
{evaluations}

VOICE DELIVERY STATS (from speech analysis; may be empty for typed answers):
{delivery}

HINTS USED: {hints}
{ended_early}

Write the report:
- overall_score 0-100 and hire_signal for this {seniority} role, calibrated like a real interviewer.
- summary: 3-4 sentences.
- top_strengths: 3-5 items, each with concrete evidence quoting/referencing their answers.
- improvement_areas: 3-6 items, ordered by impact. For each: why (citing the weak answer),
  how_to_improve (specific, actionable steps), example_better_answer (a short, realistic stronger
  answer in the candidate's own context), related_question_ids.
- per_question: exactly one entry per MAIN question id that was answered (q1, q2, ...), folding
  its follow-up answers into that entry. Never invent ids like "q1_follow_up". Score 1-10.
- coding_summary: correctness, complexity, code quality, process (empty if no coding answered).
- delivery_summary: pace (ideal 130-160 wpm), filler words, pauses, confidence, with numbers
  (empty if no voice answers).
- hints_summary: how hint usage affected the assessment (empty if none).
- recommended_resources: specific topics/books/practice areas (no made-up URLs).
- next_steps: 3-5 concrete practice actions for the next week.
Be direct and specific; avoid generic advice."""
