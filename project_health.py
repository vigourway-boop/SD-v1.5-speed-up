"""Read-only local dependencies, model cache and PYNQ protocol checks."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPENDENCIES = {"torch": "torch", "diffusers": "diffusers", "transformers": "transformers",
                "accelerate": "accelerate", "safetensors": "safetensors",
                "numpy": "numpy", "Pillow": "PIL", "scikit-image": "skimage", "lpips": "lpips"}


def inspect_cache():
    from huggingface_hub import snapshot_download
    from config import MODEL_ID
    path = Path(snapshot_download(MODEL_ID, local_files_only=True))
    missing = []
    for name in ("model_index.json", "scheduler/scheduler_config.json", "tokenizer/vocab.json",
                 "tokenizer/merges.txt", "tokenizer/tokenizer_config.json"):
        if not (path / name).is_file():
            missing.append(name)
    for module in ("unet", "vae", "text_encoder"):
        if not (path / module / "config.json").is_file():
            missing.append(f"{module}/config.json")
        if not list((path / module).glob("*.safetensors")):
            missing.append(f"{module}/*.safetensors")
    return {"path": str(path), "missing": missing,
            "note": "File presence only; no model load or weight checksum verification"}


def check_project(backend="auto", host="192.168.2.99", port=9000, profile=None):
    errors, warnings = [], []
    result = {"python": sys.executable, "backend_requested": backend, "versions": {},
              "errors": errors, "warnings": warnings}
    for distribution, module in DEPENDENCIES.items():
        if importlib.util.find_spec(module) is None:
            errors.append(f"缺少依赖 / Missing dependency: {distribution}")
        else:
            try:
                result["versions"][distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                result["versions"][distribution] = "unknown"
    if errors:
        return result
    import torch
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        result["gpu"] = torch.cuda.get_device_name()
    else:
        warnings.append("CUDA不可用，将使用CPU生成，耗时可能很长 / CUDA unavailable")
    try:
        result["sd_cache"] = inspect_cache()
        if result["sd_cache"]["missing"]:
            warnings.append("SD缓存不完整；生成时可能需要下载 / Incomplete local model cache")
    except Exception as exc:
        result["sd_cache"] = {"error": str(exc)}
        warnings.append("没有完整的本地SD缓存；首次生成需要联网 / Local model cache unavailable")
    if profile:
        from controller_profile import load_controller_profile
        try:
            result["profile"] = load_controller_profile(profile)
            result["profile_path"] = str(Path(profile).resolve())
        except (ValueError, OSError) as exc:
            errors.append(f"参数文件无效 / Invalid profile: {exc}")
    artifacts = ROOT / "pynq_cosine_overlay" / "artifacts"
    result["local_hardware_files"] = {}
    for name in ("cosine_overlay.bit", "cosine_overlay.hwh"):
        path = artifacts / name
        result["local_hardware_files"][name] = {
            "exists": path.is_file(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
    result["board"] = {"checked": backend != "pc", "host": host, "port": port}
    if backend != "pc":
        from pynq_cosine_client import PynqCosineClient
        client = PynqCosineClient(host, port, timeout=3)
        try:
            client.connect()
            result["board"].update(available=True, ping_ms=client.health_check(), protocol="CSK4")
        except (ConnectionError, OSError, RuntimeError) as exc:
            result["board"].update(available=False, error=str(exc))
            message = f"PYNQ服务不可达 / Board service unavailable: {host}:{port}"
            (errors if backend == "pynq" else warnings).append(message)
        finally:
            client.close()
    result["board_check_scope"] = "PING verifies service protocol, not loaded bitstream identity or FPGA arithmetic"
    return result


def print_health(result):
    print(f"Python: {result['python']}")
    print(f"GPU: {result.get('gpu', 'CPU / unavailable')}")
    board = result.get("board", {})
    print(f"PYNQ: {board.get('available', 'not checked')} ({board.get('host', '')})")
    for warning in result["warnings"]:
        print(f"提示 / Note: {warning}")
    for error in result["errors"]:
        print(f"错误 / Error: {error}")
    print("检查未通过 / Check failed" if result["errors"] else "启动检查通过 / Startup checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "pc", "pynq"), default="auto")
    parser.add_argument("--host", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_project(args.backend, args.host, args.port, args.profile)
    print_health(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
