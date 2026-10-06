"""Offline demo, budget preview, capability probe and explanation comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import ImageDataset
from .runner import plan_run, run


def build_parser():
    parser = argparse.ArgumentParser(prog="medshearletx")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run three scores on a synthetic image with a mock VLM")
    demo.add_argument("--output", default="output/demo")
    demo.add_argument("--transform", choices=["shearlet", "identity"], default="shearlet")
    for name in ("run", "probe"):
        command = commands.add_parser(name, help="Explain images" if name == "run" else "Test score capabilities on images")
        command.add_argument("--config", required=True)
        inputs = command.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--images", help="Image folder")
        inputs.add_argument("--manifest", help="CSV with image_path, optional sample_id and label")
        command.add_argument("--output", required=True)
        command.add_argument("--limit", type=int, default=1)
        command.add_argument("--scores", nargs="+", choices=["probability", "log_margin", "agreement"])
        command.add_argument("--target", help="Fixed target label; defaults to original prediction")
        command.add_argument("--dry-run", action="store_true", help="Validate configuration and print budget without API calls")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            destination = Path(args.output)
            if destination.exists() and any(destination.iterdir()):
                raise ValueError("Output directory must be empty; choose a new run directory")
            destination.mkdir(parents=True, exist_ok=True)
            # Deliberately synthetic: no patient data or credentials needed.
            pixels = np.full((128, 128, 3), 32, dtype=np.uint8)
            pixels[32:96, 32:96] = 220
            pixels[48:80, 60:68] = 255
            image_path = destination / "synthetic.png"
            Image.fromarray(pixels).save(image_path)
            config = {
                "model": {"backend": "mock", "seed": 7},
                "task": {"labels": ["signal_present", "signal_absent"], "question": "Is the central bright signal present?"},
                "scores": ["probability", "log_margin", "agreement"],
                "sampling": {"repeats": 8, "temperature": 1.0},
                "transform": {"name": args.transform},
                "image_size": 128,
                "explainer": {"steps": 6, "grid_size": 4, "mask_init": 0.8, "seed": 42},
                "max_total_requests": 500,
            }
            if args.transform == "shearlet":
                config["transform"]["scales"] = 2
            dataset = ImageDataset.from_folder(destination)
            output = destination / "comparison"
            kwargs = {}
        else:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            dataset = ImageDataset.from_folder(args.images) if args.images else ImageDataset.from_manifest(args.manifest)
            kwargs = {"limit": args.limit, "scores": args.scores, "target": args.target,
                      "probe": args.command == "probe"}
            if args.dry_run:
                print(json.dumps(plan_run(config, dataset, args.limit, args.scores, args.command == "probe"), indent=2))
                return 0
            output = args.output
        results = run(config, dataset, output, progress=print, **kwargs)
        successes = sum(row["status"] == "ok" for row in results)
        print(f"Completed {successes}/{len(results)} comparisons. Results: {Path(output).resolve() / 'results.json'}")
        for row in results:
            if row["status"] != "ok":
                print(f"{row['mode']}: {row['error']}")
        return 0 if successes == len(results) else 2
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(f"Error: {exc}")
        return 2
