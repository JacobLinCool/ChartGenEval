#!/usr/bin/env python3
"""Create the one-time, immutable pre-run freeze snapshot."""

from __future__ import annotations

import argparse

from contract import (
    build_freeze_payload,
    load_contract,
    resolve_repo_path,
    write_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config, _contract, _config_path, _contract_path = load_contract(args.config)
    if config["phase"] != "confirmatory":
        raise SystemExit("freeze.py accepts only the confirmatory config")
    if resolve_repo_path(args.out) != resolve_repo_path(config["freeze_path"]):
        raise SystemExit("--out must equal the unique freeze_path locked in confirmatory.json")
    payload = build_freeze_payload(args.config)
    write_json_exclusive(args.out, payload)
    print(f"frozen code bundle {payload['source_snapshot']['code_bundle_sha256']} -> {args.out}")


if __name__ == "__main__":
    main()
