"""Submit a ZIP to the AIMO Interp Codabench competition. Run it yourself.

Your password is read with a hidden prompt and sent only to Codabench; it is
never stored, logged, or passed through anyone else.

    python scripts/codabench_submit.py dist/solution-self-perturb-small.zip

Flow (Codabench REST API): token auth -> create dataset -> PUT bytes to the
presigned URL -> mark dataset complete -> create submission -> poll until the
submission finishes, then print its scores.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import requests

BASE = "https://www.codabench.org"
COMPETITION_PHASE = 30038  # "Competition phase"
SMALL_TASK = 36241  # Small Model Track (as used for submissions on 2026-09-28)
TERMINAL = {"Finished", "Failed", "Cancelled"}


def fail(step: str, response: requests.Response) -> None:
    print(f"\n{step} failed: HTTP {response.status_code}\n{response.text[:1500]}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("--username", default="rmtaware")
    parser.add_argument("--phase", type=int, default=COMPETITION_PHASE)
    parser.add_argument("--task", type=int, default=SMALL_TASK)
    parser.add_argument("--poll-minutes", type=int, default=90)
    arguments = parser.parse_args()

    payload = arguments.zip.read_bytes()
    with ZipFile(arguments.zip) as bundle:
        names = bundle.namelist()
    has_marker = any(name.casefold() == "small.txt" for name in names)
    if arguments.task == SMALL_TASK and not has_marker:
        sys.exit("refusing: small-track task but the ZIP has no small.txt at its root")
    if "solution.py" not in names:
        sys.exit("refusing: solution.py is not at the ZIP root")

    print(f"file   : {arguments.zip}  ({len(payload):,} bytes)")
    print(f"sha256 : {hashlib.sha256(payload).hexdigest()}")
    print(f"target : phase {arguments.phase}, task {arguments.task}"
          f" ({'Small Model Track' if arguments.task == SMALL_TASK else 'custom'})")
    print(f"user   : {arguments.username}")
    if input("Submit this? Uses one of your daily slots. Type yes: ").strip().lower() != "yes":
        sys.exit("aborted, nothing submitted")

    password = getpass.getpass(f"Codabench password for {arguments.username} (hidden): ")
    response = requests.post(
        f"{BASE}/api/api-token-auth/",
        data={"username": arguments.username, "password": password},
        timeout=60,
    )
    del password
    if not response.ok:
        fail("login", response)
    headers = {"Authorization": f"Token {response.json()['token']}"}

    response = requests.post(
        f"{BASE}/api/datasets/",
        data={
            "type": "submission",
            "request_sassy_file_name": arguments.zip.name,
            "file_name": arguments.zip.name,
            "file_size": len(payload),
        },
        headers=headers,
        timeout=60,
    )
    if not response.ok:
        fail("create dataset", response)
    dataset = response.json()

    response = requests.put(
        dataset["sassy_url"], data=payload, headers={"Content-Type": "application/zip"}, timeout=600
    )
    if not response.ok:
        fail("upload", response)

    response = requests.put(f"{BASE}/api/datasets/completed/{dataset['key']}/", headers=headers, timeout=60)
    if not response.ok:
        fail("mark upload complete", response)

    response = requests.post(
        f"{BASE}/api/submissions/",
        json={"phase": arguments.phase, "tasks": [arguments.task], "data": dataset["key"]},
        headers=headers,
        timeout=60,
    )
    if not response.ok:
        fail("create submission", response)
    submission = response.json()
    submission_id = submission.get("id")
    print(f"\nsubmitted: id {submission_id}. Polling for the result (Ctrl+C is safe; it keeps running).")

    deadline = time.time() + arguments.poll_minutes * 60
    status = submission.get("status")
    while time.time() < deadline:
        response = requests.get(f"{BASE}/api/submissions/{submission_id}/", headers=headers, timeout=60)
        if response.ok:
            detail = response.json()
            if detail.get("status") != status:
                status = detail.get("status")
                print(f"  {time.strftime('%H:%M:%S')}  {status}")
            if status in TERMINAL:
                scores = {key: detail[key] for key in ("scores", "score", "leaderboard") if key in detail}
                print(json.dumps(scores or detail, indent=2, default=str))
                return
        time.sleep(30)
    print("still running; check the Codabench submissions page later")


if __name__ == "__main__":
    main()
