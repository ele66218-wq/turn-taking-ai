// model.js -- load model.json (exported by scripts/export_model_json.py) and run inference.
"use strict";

const Model = (() => {
  async function load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
    const model = await res.json();
    if (!model.classifier || !model.scaler || !model.feature_names) {
      throw new Error("model.json is missing required fields");
    }
    return model;
  }

  function scale(model, featureVector) {
    const { mean, scale: scaleArr } = model.scaler;
    const out = new Float64Array(featureVector.length);
    for (let i = 0; i < featureVector.length; i++) {
      const s = scaleArr[i] === 0 ? 1e-10 : scaleArr[i];
      out[i] = (featureVector[i] - mean[i]) / s;
    }
    return out;
  }

  function treeProbaPositive(tree, x) {
    let node = 0;
    // sklearn convention: X[feature] <= threshold -> go left, else right. -1 marks a leaf.
    while (tree.children_left[node] !== -1) {
      node = x[tree.feature[node]] <= tree.threshold[node] ? tree.children_left[node] : tree.children_right[node];
    }
    return tree.proba1[node];
  }

  function randomForestProba(classifier, scaled) {
    let sum = 0;
    for (const tree of classifier.trees) sum += treeProbaPositive(tree, scaled);
    return sum / classifier.trees.length;
  }

  function logisticRegressionProba(classifier, scaled) {
    let z = classifier.intercept;
    for (let i = 0; i < scaled.length; i++) z += classifier.coef[i] * scaled[i];
    return 1 / (1 + Math.exp(-z));
  }

  /** P(user wants the floor) in [0, 1] for one raw (unscaled) feature vector. */
  function predictProba(model, featureVector) {
    const scaled = scale(model, featureVector);
    if (model.classifier.type === "random_forest") return randomForestProba(model.classifier, scaled);
    if (model.classifier.type === "logistic_regression") return logisticRegressionProba(model.classifier, scaled);
    throw new Error(`unsupported classifier type: ${model.classifier.type}`);
  }

  return { load, predictProba };
})();
