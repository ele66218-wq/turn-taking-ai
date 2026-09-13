#!/usr/bin/env python3
"""Export a trained RandomForest pipeline (scaler + classifier) to plain JSON.

The browser demo (docs/) has no scikit-learn, numpy, or joblib available --
it runs in vanilla JavaScript on GitHub Pages. This script flattens the
fitted StandardScaler and RandomForestClassifier into a JSON structure that
`docs/js/model.js` can walk directly (feature scaling, then per-tree
traversal, then averaging leaf class-probabilities across trees).

Usage
-----
    python scripts/export_model_json.py --model models/baseline.joblib \
        --out docs/model.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turn_taking.train import load_model  # noqa: E402


def _export_tree(tree) -> dict:
    """Flatten one sklearn DecisionTreeClassifier's internal arrays to lists.

    sklearn stores each node's children/feature/threshold/value in parallel
    numpy arrays indexed by node id; -1 (TREE_LEAF) marks a leaf's children.
    ``value`` holds unnormalised class counts per node -- we normalise here so
    the JS side only ever deals with probabilities.
    """
    t = tree.tree_
    values = t.value.reshape(t.node_count, -1)
    proba = values / values.sum(axis=1, keepdims=True)
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        # proba[:, 1] is P(class == 1) at every node; leaves are what matter,
        # but storing all nodes keeps this generically correct.
        "proba1": proba[:, 1].tolist(),
    }


def _export_random_forest(classifier) -> dict:
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError(f"classifier classes {classes} do not include the positive class 1")
    return {
        "type": "random_forest",
        "n_estimators": len(classifier.estimators_),
        "trees": [_export_tree(tree) for tree in classifier.estimators_],
    }


def _export_logistic_regression(classifier) -> dict:
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError(f"classifier classes {classes} do not include the positive class 1")
    positive_index = classes.index(1)
    coef = classifier.coef_[0] if classifier.coef_.shape[0] == 1 else classifier.coef_[positive_index]
    intercept = float(
        classifier.intercept_[0]
        if classifier.intercept_.shape[0] == 1
        else classifier.intercept_[positive_index]
    )
    return {"type": "logistic_regression", "coef": coef.tolist(), "intercept": intercept}


def export_model(model_path: Path, out_path: Path) -> None:
    loaded = load_model(model_path)
    pipeline = loaded.pipeline
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    classifier_type = type(classifier).__name__
    if classifier_type == "RandomForestClassifier":
        classifier_payload = _export_random_forest(classifier)
    elif classifier_type == "LogisticRegression":
        classifier_payload = _export_logistic_regression(classifier)
    else:
        raise ValueError(f"unsupported classifier type for export: {classifier_type}")

    payload = {
        "feature_names": loaded.feature_names,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "classifier": classifier_payload,
        "config": loaded.config.to_dict(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"[export] wrote {out_path} ({size_kb:.1f} KiB, {len(loaded.feature_names)} features)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/baseline.joblib"))
    parser.add_argument("--out", type=Path, default=Path("docs/model.json"))
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"error: model file not found: {args.model}", file=sys.stderr)
        return 1
    export_model(args.model, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
