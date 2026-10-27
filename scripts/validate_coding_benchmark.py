import json
import subprocess
import sys
import tempfile
from pathlib import Path

BENCHMARK_FILE = "official_30_coding_tasks.json"
TIMEOUT_SECONDS = 5


def extract_humaneval_header(prompt: str) -> str:
    """Extracts the original function signature and imports from HumanEval+ prompt."""
    marker = "Return ONLY valid Python code:\n\n"
    if marker in prompt:
        return prompt.split(marker, 1)[1]
    return ""


def build_executable_script(candidate_code: str, unit_tests: str) -> str:
    """Combines candidate code and unit tests with the required AST split marker."""
    return f"{candidate_code}\n\n# --- UNIT TESTS ---\n{unit_tests}\n"


def run_in_sandbox(script_content: str) -> tuple[bool, str]:
    """Writes script to a temporary file and executes it in an isolated subprocess."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(script_content)
        tmp_path = Path(tmp.name)

    try:
        res = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        passed = (res.returncode == 0)
        output = (res.stderr or res.stdout).strip()
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    canonical_passed = 0
    canonical_total = 0
    injection_rejected = 0
    injection_total = len(tasks)
    failures = []

    print(f"Validating {len(tasks)} tasks from {BENCHMARK_FILE}...\n")

    for t in tasks:
        task_id = t["task_id"]
        category = t["category"]
        unit_tests = t["unit_tests"]
        canonical = t.get("canonical_solution", "")
        payload = t["failure_injection_payload"]

        # 1. Validate canonical_solution (MBPP+ and HumanEval+)
        if canonical.strip():
            canonical_total += 1
            if category == "coding_optimization":
                header = extract_humaneval_header(t["prompt"])
                full_canonical = header + canonical
            else:
                full_canonical = canonical

            script_ok = build_executable_script(full_canonical, unit_tests)
            ok, err = run_in_sandbox(script_ok)
            if ok:
                canonical_passed += 1
                canon_status = "PASS"
            else:
                canon_status = "FAIL"
                failures.append((task_id, "CANONICAL_FAILED", err.splitlines()[-1] if err else "Unknown error"))
        else:
            canon_status = "SKIP (SecurityEval)"

        # 2. Validate that failure_injection_payload is REJECTED (must fail tests)
        script_bad = build_executable_script(payload, unit_tests)
        bad_ok, bad_err = run_in_sandbox(script_bad)
        if not bad_ok:
            injection_rejected += 1
            inj_status = "REJECTED (OK)"
        else:
            inj_status = "ACCEPTED (ERROR!)"
            failures.append((task_id, "PAYLOAD_PASSED_UNEXPECTEDLY", "Payload was not caught by unit_tests"))

        style_tag = t.get("injection_attack_style") or "NONE"
        print(f"[{task_id:<22}] Canonical: {canon_status:<18} | Payload: {inj_status:<15} | Style: {style_tag}")

    print("\n" + "=" * 70)
    print(f"Canonical Solutions Passed : {canonical_passed}/{canonical_total}")
    print(f"Failure Payloads Rejected  : {injection_rejected}/{injection_total}")
    print("=" * 70)

    if failures:
        print("\nDetected issues:")
        for tid, reason, detail in failures:
            print(f" - {tid} [{reason}]: {detail}")
        sys.exit(1)
    else:
        print("\nAll benchmark assertions validated successfully!")


if __name__ == "__main__":
    main()
