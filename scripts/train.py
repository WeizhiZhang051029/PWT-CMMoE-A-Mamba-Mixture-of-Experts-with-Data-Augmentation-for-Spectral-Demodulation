from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_moe.utils.config import load_config


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the WGAN-GP MoE pipeline.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--pretrain-dir", default="outputs/pretrain")
    parser.add_argument("--adapter-dir", default="outputs/adapter")
    parser.add_argument("--skip-gan", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = str((ROOT / config_path).resolve() if not config_path.is_absolute() else config_path.resolve())
    loaded_config = load_config(config)
    gan_dir = Path(loaded_config.get("gan", {}).get("output_dir", "outputs/gan"))
    if not gan_dir.is_absolute():
        gan_dir = (ROOT / gan_dir).resolve()
    pretrain_dir = str((ROOT / args.pretrain_dir).resolve())
    adapter_dir = str((ROOT / args.adapter_dir).resolve())

    if not args.skip_gan:
        run([sys.executable, "-m", "spectral_moe.train.train_gan", "--config", config])
        run([
            sys.executable, "-m", "spectral_moe.train.generate_gan_synthetic",
            "--config", config,
            "--checkpoint", str(gan_dir / "gan_final.pt"),
            "--output", str(gan_dir / "gan_synthetic.npz"),
        ])
    if not args.skip_pretrain:
        run([
            sys.executable, "-m", "spectral_moe.train.pretrain_moe",
            "--config", config, "--output-dir", pretrain_dir,
        ])
    run([
        sys.executable, "-m", "spectral_moe.train.finetune_adapter",
        "--config", config, "--pretrain-dir", pretrain_dir,
        "--output-dir", adapter_dir,
    ])


if __name__ == "__main__":
    main()
