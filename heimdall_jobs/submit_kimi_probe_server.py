"""Start a Kimi-K2.5 probe server (TP=8) on node2 via Heimdall.

Persistent vllm-server job — occupies all 8 GPUs on node2 until cancelled.
Loads the SAE-decoder probe set from `probes/k25/` on node2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


REPO_DIR_DEFAULT = "/home/athuser/assistant-axis-exp"
HEIMDALL_API_DEFAULT = "http://node1.datasci.ath:7000"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", default=os.environ.get("REPO_DIR", REPO_DIR_DEFAULT),
                   help="Path to assistant-axis-exp on the target node.")
    p.add_argument("--api", default=os.environ.get("HEIMDALL_API", HEIMDALL_API_DEFAULT))
    p.add_argument("--name", default="kimi_probe_server")
    p.add_argument("--node", default="node2")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--probe-set-names", default="sae_steer",
                   help="Comma-separated probe-set names to load.")
    p.add_argument("--probes-dir", default="probes/k25",
                   help="Relative to --repo-dir.")
    p.add_argument("--env-name", default="kimi",
                   help="Conda env on target node that has vllm + the right "
                        "torch / model deps.")
    p.add_argument("--estimated-minutes", type=int, default=600)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_command(args: argparse.Namespace) -> str:
    py = (
        f"PYTHONUNBUFFERED=1 python -u scripts/probe_server_vllm.py "
        f"--model kimi --probes {args.probe_set_names} "
        f"--probes-dir {args.probes_dir} --port {args.port} --host 0.0.0.0"
    )
    return (
        "source ~/miniconda/etc/profile.d/conda.sh && "
        f"conda activate {args.env_name} && "
        f"cd {args.repo_dir} && "
        + py
    )


def main() -> None:
    args = parse_args()
    spec = {
        "job_type": "vllm-server",
        "name": args.name,
        "command": build_command(args),
        "gpus": 8,
        "node": args.node,
        "working_dir": args.repo_dir,
        "estimated_minutes": args.estimated_minutes,
        "persistent": True,
        "preemptable": False,
        "always_on": False,
        "tags": ["concept_search", "kimi", "probe_server"],
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
    main()
