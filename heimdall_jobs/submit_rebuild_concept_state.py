"""Submit the concept-director rebuild job (baseline + 12 concepts)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


REPO_DIR_DEFAULT = "/home/athuser/gnome_home/concept_search"
HEIMDALL_API_DEFAULT = "http://node1.datasci.ath:7000"
ENV_NAME_DEFAULT = "kimi"   # node2 env that has httpx + numpy + concept_search (editable)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", default=os.environ.get("REPO_DIR", REPO_DIR_DEFAULT))
    p.add_argument("--env-name", default=os.environ.get("ENV_NAME", ENV_NAME_DEFAULT))
    p.add_argument("--api", default=os.environ.get("HEIMDALL_API", HEIMDALL_API_DEFAULT))
    p.add_argument("--name", default="rebuild_concept_state")
    p.add_argument("--node", default="node2")
    p.add_argument("--estimated-minutes", type=int, default=45)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_command(args: argparse.Namespace) -> str:
    return (
        "source ~/miniconda/etc/profile.d/conda.sh && "
        f"conda activate {args.env_name} && "
        f"cd {args.repo_dir} && "
        "PYTHONPATH=src PYTHONUNBUFFERED=1 "
        "python -u scripts/rebuild_concept_state.py"
    )


def submit(args: argparse.Namespace) -> None:
    spec = {
        "job_type": "custom",
        "name": args.name,
        "command": build_command(args),
        "gpus": 0,
        "node": args.node,
        "working_dir": args.repo_dir,
        "estimated_minutes": args.estimated_minutes,
        "tags": ["concept_search", "rebuild"],
    }
    body = {"spec": spec, "submitted_by": "concept_search"}

    if args.dry_run:
        print(json.dumps(body, indent=2))
        return

    req = urllib.request.Request(
        f"{args.api}/api/v1/jobs",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code} {e.reason}", file=sys.stderr)
        print(e.read().decode(errors="replace"), file=sys.stderr)
        sys.exit(1)

    job = payload.get("job", {})
    print(f"submitted: id={job.get('id')} status={job.get('status')}")
    for w in payload.get("warnings", []) or []:
        print(f"warning: {w}", file=sys.stderr)


if __name__ == "__main__":
    submit(parse_args())
