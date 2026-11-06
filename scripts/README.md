# Benchmark & Utility Scripts Guide

This directory (along with `backend/benchmark_runner.py`) contains the tooling required to generate the evaluation datasets, verify node readiness, execute automated multi-agent benchmarks across different model families (`gemma`, `qwen`, `mixed`), and analyze empirical results.

---

## 1. `scripts/prepare_benchmark.py`
Generates the two standardized JSON benchmark datasets used for evaluation:
* `official_30_coding_tasks.json` — 30 Python programming and security tasks (equipped with unit tests, CWE vulnerability targets, and deterministic 15% adversarial failure injection payloads).
* `official_50_text_tasks.json` — 50 domain-knowledge and factual verification tasks (equipped with reference answers and deterministic 15% hallucination/prompt-injection payloads).

```bash
python scripts/prepare_benchmark.py
```
## 2. `scripts/validate_coding_benchmark.py`
Verifies the integrity of `official_30_coding_tasks.json` before running cluster-wide benchmarks. It checks JSON schema completeness, confirms that deterministic 15% failure injection flags are properly distributed, and validates the syntax of embedded Python unit tests.
```bash
python scripts/validate_coding_benchmark.py
```
## 3. `scripts/test_generate.py`
A lightweight diagnostic script used to verify connectivity, model loading, and token generation speed on individual nodes (PC1–PC4) or through the central API Gateway (PC0) before starting a full benchmark run.
```bash
python scripts/test_generate.py
```
## 4. `backend/benchmark_runner.py`
Executes the automated evaluation suites against the API Gateway (`http://localhost:8000`). For each task, it records:
  * End-to-end latency and pernode execution traces (`execution_trace`),
  * Prompt and completion costs,
  * Unit test execution results in an isolated Python sandbox (5.0s timeout) and static CWE regex scans (for coding tasks),
  * Judge GBNF rubric scores, initial vs final verdicts, self-correction retry counts, and resilience against Indirect Prompt Injection (`insistent_suggestion`, `misdirection_context`, `reverse_psychology`).
### Usage Examples
* **Run full benchmark suite (Coding + Text) using default families from `.env`:**
  ```bash
  python backend/benchmark_runner.py
  ```
* **Run benchmarks for a specific model family (`gemma`, `qwen`, or `mixed`):**
  * `gemma`:
    ```bash
    python backend/benchmark_runner.py --families gemma
    ```
  * `qwen`:
    ```bash
    python backend/benchmark_runner.py --families qwen
    ```
  * `mixed`:
    ```bash
    python backend/benchmark_runner.py --families mixed
    ```
* **Compare all three model families sequentially in a single automated run:**
  ```bash
  python backend/benchmark_runner.py --families gemma qwen mixed
  ```
* **Run only the coding suite or only the text suite:**
  * `gemma` and `coding`:
    ```bash
    python backend/benchmark_runner.py --suite coding --families gemma
    ```
  * `gemma` and `text`:
    ```bash
    python backend/benchmark_runner.py --suite text --families gemma
    ```
  * `qwen` and `coding`:
    ```bash
    python backend/benchmark_runner.py --suite coding --families qwen
    ```
  * `qwen` and `text`:
    ```bash
    python backend/benchmark_runner.py --suite text --families qwen
    ```
* **Resume an interrupted benchmark run (skips already completed (`run_name`, `task_id`) pairs in the output JSON):**
  ```bash
  python backend/benchmark_runner.py --families gemma qwen mixed --resume
  ```
* **Specify custom output filenames or a remote Gateway URL:**
  ```bash
  python backend/benchmark_runner.py \
    --gateway-url [http://192.168.1.10:8000](http://192.168.1.10:8000) \
    --suite coding \
    --families mixed \
    --coding-output results_30_coding_mixed.json
  ```
## 5. `scripts/analyze_results.py`
Parses the generated output files (`results_30_coding.json` and `results_50_text.json`) and computes aggregated metrics across configurations (`Fast`, `Parallel`, `Sequential`) and model families:
* **Coding Metrics:** Unit test pass rate (%), static CWE vulnerability rate, Judge security & performance rubric averages, and sandbox execution time.
* **Pro/Text Metrics:** Factual consistency scores, detected flaws count, and hallucination rejection rates.
* **Resilience and Overhead Metrics:** Indirect Prompt Injection catch rate (`judge_caught_injection`), self-correction recovery rate vs Judge Takeover rate, average token consumtion, wall-clock atency, and parallel stage time saved (straggler node analysis).
  ```bash
  python scripts/analyze_results.py
  ```
