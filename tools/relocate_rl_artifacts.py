#!/usr/bin/env python3
"""Move completed RL model directories to a data disk, preserving logical paths.

Default is a read-only inventory. --apply copies to a temporary directory,
verifies every SHA256, then replaces the source with a symlink. The original
copy is removed only after verification and successful symlink creation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

DIRECTORIES = ("dpo/merged", "grpo/merged", "dpo/adapter", "grpo/adapter",
               "smoke_dpo_adapter", "smoke_grpo_adapter")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(directory):
    paths = sorted(directory.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError(f"nested symlinks require manual inspection: {directory}")
    return {str(p.relative_to(directory)): {"bytes": p.stat().st_size,
                                           "mtime_ns": p.stat().st_mtime_ns}
            for p in paths if p.is_file()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", default="logs/rl_minicpm5_20260928")
    ap.add_argument("--target-root", default="/models/hch/Agent_MFF_Early_Warning/rl_runs/20260928")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    source, target = Path(args.source_root).absolute(), Path(args.target_root).absolute()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("source and target must be separate directory trees")
    plan = []
    for relative in DIRECTORIES:
        src, dst = source / relative, target / relative
        if src.is_symlink():
            if src.resolve() != dst.resolve() or not dst.is_dir():
                raise ValueError(f"unexpected or broken symlink: {src}")
            continue
        if not src.is_dir() or dst.exists():
            raise ValueError(f"missing source or existing destination: {src} -> {dst}")
        files = inventory(src)
        plan.append({"source": str(src), "destination": str(dst),
                     "bytes": sum(v["bytes"] for v in files.values()), "files": files})
    print(json.dumps({"apply": args.apply, "bytes": sum(p["bytes"] for p in plan),
                      "directories": [{k: v for k, v in p.items() if k != "files"} for p in plan]}, indent=2), flush=True)
    if not args.apply:
        return
    if not plan:
        print("All artifact directories are already linked; existing receipt retained.")
        return
    receipt = source / "relocation_manifest.json"
    if receipt.exists() and plan:
        raise FileExistsError(f"existing receipt; inspect before retrying: {receipt}")
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "moving", "directories": []}
    for item in plan:
        src, dst = Path(item["source"]), Path(item["destination"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        temporary = dst.with_name(dst.name + ".copying-" + token)
        backup = src.with_name(src.name + ".relocating-" + token)
        shutil.copytree(src, temporary)
        if inventory(src) != item["files"]:
            raise RuntimeError(f"source changed during copy; original retained: {src}")
        hashes = {}
        for name in item["files"]:
            a, b = digest(src / name), digest(temporary / name)
            if a != b:
                raise RuntimeError(f"checksum mismatch; original retained: {src / name}")
            hashes[name] = a
        if inventory(src) != item["files"]:
            raise RuntimeError(f"source changed during verification; original retained: {src}")
        temporary.rename(dst)
        src.rename(backup)
        try:
            src.symlink_to(dst, target_is_directory=True)
        except BaseException:
            backup.rename(src)
            raise
        shutil.rmtree(backup)
        result["directories"].append({**item, "sha256": hashes, "verified": True})
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(f"verified and linked: {src} -> {dst}", flush=True)
    result["status"] = "completed"
    result["bytes_moved"] = sum(item["bytes"] for item in result["directories"])
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
