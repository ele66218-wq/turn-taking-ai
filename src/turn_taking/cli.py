"""Command-line entry point: `turn-taking <subcommand> ...`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config, load_config
from .labels import label_summary, read_labels


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to a YAML config (default: config/default.yaml)"
    )


def cmd_synth(args: argparse.Namespace) -> int:
    from .synthetic import generate_dataset

    config = load_config(args.config)
    print(
        "[synth] generating a SYNTHETIC dataset for pipeline smoke-testing.\n"
        "         This is NOT real speech; train a usable model on real recordings."
    )
    rows = generate_dataset(args.out, config.audio.sample_rate, n_per_label=args.n_per_label, seed=args.seed)
    print(f"[synth] wrote {len(rows)} clips + labels.csv under {args.out}")
    print(f"[synth] label counts: {label_summary(rows)}")
    return 0


def cmd_dataset_info(args: argparse.Namespace) -> int:
    rows = read_labels(args.labels)
    trainable = [row for row in rows if row.is_trainable]
    print(f"[dataset] {args.labels}: {len(rows)} rows, {len(trainable)} trainable")
    print(f"[dataset] label counts: {label_summary(rows)}")
    speakers = sorted({row.speaker for row in rows})
    print(f"[dataset] speakers/groups: {speakers}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .dataset import build_dataset
    from .train import save_model, train_test_evaluate

    config = load_config(args.config)
    dataset = build_dataset(args.labels, args.audio_root, config, skip_missing=args.skip_missing)
    print(f"[train] loaded {len(dataset)} clips, {dataset.X.shape[1]} features")

    result = train_test_evaluate(dataset, config)
    print("[train] held-out metrics:")
    print(result.metrics.format())
    if result.cv_metrics:
        import numpy as np

        f1s = [m.f1 for m in result.cv_metrics]
        print(
            f"[train] grouped {len(result.cv_metrics)}-fold CV: "
            f"F1 mean={float(np.mean(f1s)):.3f} std={float(np.std(f1s)):.3f}"
        )
    else:
        print("[train] grouped CV skipped (fewer than 2 speaker groups in labels.csv)")

    save_model(result, args.model_out)
    print(f"[train] saved model to {args.model_out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .audio_io import load_wav
    from .infer import TurnTakingPredictor

    predictor = TurnTakingPredictor(args.model)
    signal = load_wav(args.wav, predictor.config.audio.sample_rate)
    probability = predictor.predict_proba(signal)
    print(f"[predict] {args.wav}: P(user wants the floor) = {probability:.3f}")
    return 0


def cmd_realtime(args: argparse.Namespace) -> int:
    from .infer import TurnTakingPredictor
    from .realtime import run_from_mic, run_simulation, summarize_config

    predictor = TurnTakingPredictor(args.model)
    print(f"[realtime] {summarize_config(predictor.config)}")
    if args.simulate is not None:
        run_simulation(predictor, args.simulate)
    else:
        try:
            run_from_mic(predictor)
        except RuntimeError as exc:
            print(f"[realtime] error: {exc}", file=sys.stderr)
            return 1
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    config: Config = load_config(args.config)
    import json

    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turn-taking",
        description="Turn-taking AI: estimate when a spoken-dialogue system should stop talking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synth", help="Generate a synthetic dataset for smoke-testing.")
    synth.add_argument("--out", type=Path, default=Path("data/raw"))
    synth.add_argument("--n-per-label", type=int, default=60)
    synth.add_argument("--seed", type=int, default=0)
    _add_common_args(synth)
    synth.set_defaults(func=cmd_synth)

    info = subparsers.add_parser("dataset-info", help="Print label counts and speaker groups.")
    info.add_argument("--labels", type=Path, default=Path("data/raw/labels.csv"))
    info.set_defaults(func=cmd_dataset_info)

    train = subparsers.add_parser("train", help="Train and save the baseline model.")
    train.add_argument("--labels", type=Path, default=Path("data/raw/labels.csv"))
    train.add_argument("--audio-root", type=Path, default=Path("data/raw"))
    train.add_argument("--model-out", type=Path, default=Path("models/baseline.joblib"))
    train.add_argument(
        "--skip-missing", action="store_true", help="Skip clips whose WAV file is missing."
    )
    _add_common_args(train)
    train.set_defaults(func=cmd_train)

    predict = subparsers.add_parser("predict", help="Run one WAV file through a saved model.")
    predict.add_argument("wav", type=Path)
    predict.add_argument("--model", type=Path, default=Path("models/baseline.joblib"))
    predict.set_defaults(func=cmd_predict)

    realtime = subparsers.add_parser("realtime", help="Run the live control loop (mic or simulated).")
    realtime.add_argument("--model", type=Path, default=Path("models/baseline.joblib"))
    realtime.add_argument(
        "--simulate", type=Path, default=None, help="Replay a WAV file instead of using the microphone."
    )
    realtime.set_defaults(func=cmd_realtime)

    show_config = subparsers.add_parser("config-show", help="Print the resolved configuration as JSON.")
    _add_common_args(show_config)
    show_config.set_defaults(func=cmd_config_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
