"""Upload MNIST demo metrics (JSON lines from --metrics_file) to SwanLab.

The demo writes one JSON object per line:
  {"type": "train_step", "epoch": E, "step": S, "loss": L}
  {"type": "epoch_end", "epoch": E, "train_loss": TL, "test_loss": XL, "test_accuracy": XA}

Usage:
  export SWANLAB_API_KEY=<your key>
  python3 scripts/swanlab_upload.py --metrics /tmp/runs/cnn_cuda_3ep.jsonl \
      --name cnn-cuda-3epoch --model cnn --device cuda --lr 0.05
"""

import argparse
import json
import os

import swanlab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", required=True, help="JSON lines file written by the mnist demo"
    )
    parser.add_argument("--project", default="infinitrain-mnist-cnn")
    parser.add_argument("--name", required=True, help="experiment name")
    parser.add_argument("--model", default="", help="recorded in the experiment config")
    parser.add_argument(
        "--device", default="", help="recorded in the experiment config"
    )
    parser.add_argument("--lr", default="", help="recorded in the experiment config")
    args = parser.parse_args()

    if not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("SWANLAB_API_KEY is not set")

    rows = []
    with open(args.metrics) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    train_rows = [r for r in rows if r.get("type") == "train_step"]
    epoch_rows = [r for r in rows if r.get("type") == "epoch_end"]
    if not train_rows and not epoch_rows:
        raise SystemExit("no metrics found")

    steps_per_epoch = (
        max((r["step"] for r in train_rows), default=0) + 1 if train_rows else 0
    )

    run = swanlab.init(
        project=args.project,
        experiment_name=args.name,
        api_key=os.environ["SWANLAB_API_KEY"],
        config={
            "model": args.model,
            "device": args.device,
            "learning_rate": args.lr,
            "epochs": len(epoch_rows),
            "batch_size": 64,
            "dataset": "MNIST",
        },
    )

    for r in train_rows:
        swanlab.log({"train/loss": r["loss"]}, step=r["step"])
    for r in epoch_rows:
        step = (r["epoch"] + 1) * steps_per_epoch - 1
        metrics = {
            "train/epoch_loss": r["train_loss"],
            "test/loss": r["test_loss"],
            "test/accuracy": r["test_accuracy"],
        }
        swanlab.log(metrics, step=step)

    swanlab.finish()
    print(f"uploaded {len(train_rows)} train points and {len(epoch_rows)} epoch points")
    try:
        print(f"run url: {run.url}")
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
