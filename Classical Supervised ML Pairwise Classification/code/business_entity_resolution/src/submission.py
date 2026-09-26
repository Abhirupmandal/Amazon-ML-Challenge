"""
Submission verification and validator execution module.
Invokes the official utils/validate_submission.py script to ensure zero formatting rejections.
"""

import subprocess
import sys
from pathlib import Path
from typing import Tuple

try:
    from src.config import OUTPUT_MATCHING, OUTPUT_CANDIDATES, TEST_DIR, UTILS_DIR
except ImportError:
    from .config import OUTPUT_MATCHING, OUTPUT_CANDIDATES, TEST_DIR, UTILS_DIR




def run_official_validator(
    matching_path: Path = OUTPUT_MATCHING,
    candidate_path: Path = OUTPUT_CANDIDATES,
    test_dir: Path = TEST_DIR,
    check_ids: bool = False
) -> Tuple[int, str]:
    """
    Execute utils/validate_submission.py against generated submission artifacts.
    Returns (exit_code, output_text).
    """
    validator_script = UTILS_DIR / "validate_submission.py"
    if not validator_script.exists():
        raise FileNotFoundError(f"Validator script not found at {validator_script}")

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]
    if check_ids:
        cmd.append("--check-ids")

    print(f"\nRunning official submission validator: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace"
    )

    stdout_str = result.stdout or ""
    stderr_str = result.stderr or ""
    combined_output = (stdout_str + "\n" + stderr_str).strip()
    print(combined_output)

    if result.returncode == 0:
        print("\n>>> OFFICIAL VALIDATION RESULT: PASS (All submission criteria met!) <<<\n")
    else:
        print("\n>>> OFFICIAL VALIDATION RESULT: FAIL (See issues above) <<<\n")

    return result.returncode, combined_output
