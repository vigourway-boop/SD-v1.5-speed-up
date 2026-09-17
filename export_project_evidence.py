"""Create a reviewable local evidence bundle without credentials or model weights."""

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def export(destination, experiment_roots):
    selected = set()
    for pattern in ("*.py", "*.md", "requirements*.txt", "profiles/*.json", "docs/*.md"):
        selected.update(p for p in ROOT.glob(pattern) if p.is_file())
    # Explicitly exclude the historical credential helper and deployment secrets.
    for name in ("run_project.cmd", "run_standard_sd.cmd", "run_evaluated_speedup.cmd", "run_pynq_speedup.cmd", "deploy_pynq_server.cmd"):
        selected.add(ROOT / name)
    hardware = ROOT / "pynq_cosine_overlay"
    selected.add(hardware / "BUILD_STATUS.txt")
    for pattern in ("hls/src/*", "hls/*.tcl", "vivado/*.tcl"):
        selected.update(p for p in hardware.glob(pattern) if p.is_file())
    for name in ("cosine_overlay.bit", "cosine_overlay.hwh", "cosine_skip_hls_ip_export.zip",
                 "pynq_cosine_server.py", "start_pynq_cosine.sh", "install_pynq_service.sh"):
        path = hardware / "artifacts" / name
        if path.is_file():
            selected.add(path)
    for experiment in experiment_roots:
        experiment = experiment.resolve()
        experiment.relative_to(ROOT / "experiments")
        if not (experiment / "completion.json").exists():
            raise ValueError(f"Cannot bundle unfinished experiment: {experiment}")
        for path in experiment.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".json", ".csv", ".md"} or path.name in {
                "comparison_gallery.jpg", "worst_cases.jpg", "all_cases.jpg",
                "practical_tradeoff.png", "speed_quality.png", "feature_curves.png", "source_snapshot.zip"}:
                selected.add(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            archive.writestr(relative, content)
            entries.append({"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        manifest = {"created": datetime.now().isoformat(), "files": entries,
                    "scope": "Sources, docs, current FPGA artifacts and selected experiment evidence. No model weights, private keys, SSH helper, individual raw PNG sets or git history. Original experiments remain on disk."}
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Bundle integrity check failed")
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".sha256").write_text(f"{checksum}  {destination.name}\n", encoding="ascii")
    print(f"Bundle: {destination}\nFiles: {len(entries)}\nSHA256: {checksum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    export(args.output, args.experiments)
