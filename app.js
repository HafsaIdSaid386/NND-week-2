/* Titanic TF.js — Shallow Binary Classifier
 * Runs entirely in the browser. No server required.
 * Reuse note: You can swap the schema by adjusting FEATURE/ TARGET names
 * and the preprocessing pipeline below.
 */

// ====== GLOBAL STATE ======
let rawTrain = [];        // array of objects from train.csv
let rawTest = [];         // array of objects from test.csv
let Xtrain = null;        // tf.Tensor for training features
let ytrain = null;        // tf.Tensor for training labels
let Xval = null;          // tf.Tensor for validation features
let yval = null;          // tf.Tensor for validation labels
let model = null;         // tf.Model
let valProbs = null;      // Float32Array of validation probabilities
let valLabels = null;     // Int32Array of validation labels
let lastThreshold = 0.5;  // current decision threshold

// Feature config (Kaggle Titanic)
const TARGET = "Survived"; // 0/1
const IDENT = "PassengerId"; // exclude from modeling
const BASE_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"];
// Engineered toggles come from UI: FamilySize, IsAlone

// Utility selectors
const $ = (id) => document.getElementById(id);

// ====== FILE LOADING & INSPECTION ======
async function readLocalCsv(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (res) => resolve(res.data),
      error: (err) => reject(err),
    });
  });
}

function topRows(arr, n = 8) {
  const keys = Object.keys(arr[0] || {});
  const rows = arr.slice(0, n).map((r) => keys.map((k) => r[k]));
  return { headers: keys, rows };
}

function renderTable(divId, data) {
  const el = $(divId);
  if (!data || data.headers.length === 0) {
    el.innerHTML = "<em>No data</em>";
    return;
  }
  const head = `<thead><tr>${data.headers.map(h => `<th>${h}</th>`).join("")}</tr></thead>`;
  const body = `<tbody>${data.rows.map(r => `<tr>${r.map(x => `<td>${x ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>`;
  el.innerHTML = `<table>${head}${body}</table>`;
}

function missingPercentages(rows) {
  if (!rows.length) return {};
  const keys = Object.keys(rows[0]);
  const total = rows.length;
  const miss = {};
  for (const k of keys) {
    let c = 0;
    for (const r of rows) if (r[k] === null || r[k] === undefined || r[k] === "") c++;
    miss[k] = +(100 * c / total).toFixed(2);
  }
  return miss;
}

function showDatasetInfo() {
  const lines = [];
  if (rawTrain.length) {
    const miss = missingPercentages(rawTrain);
    lines.push(`Train rows: ${rawTrain.length}`);
    lines.push(`Columns: ${Object.keys(rawTrain[0]).length}`);
    lines.push(`Missing % (train):`);
    for (const [k, v] of Object.entries(miss)) lines.push(`  - ${k}: ${v}%`);
  } else {
    lines.push("Train: (not loaded)");
  }
  if (rawTest.length) {
    const missT = missingPercentages(rawTest);
    lines.push("");
    lines.push(`Test rows: ${rawTest.length}`);
    lines.push(`Missing % (test):`);
    for (const [k, v] of Object.entries(missT)) lines.push(`  - ${k}: ${v}%`);
  } else {
    lines.push("");
    lines.push("Test: (not loaded)");
  }
  $("datasetInfo").textContent = lines.join("\n");
}

function barDataFromCounts(countsMap, titleKey = "group", valueKey = "value") {
  // tfjs-vis bar chart expects array of objects with x/y or key/value (we’ll use {index, value})
  const series = Object.entries(countsMap).map(([k, v]) => ({ index: k, value: v }));
  return series;
}

function chartSurvivalBySex() {
  // build counts on train
  const counts = { female: { died: 0, survived: 0 }, male: { died: 0, survived: 0 } };
  for (const r of rawTrain) {
    if (r[SEX()] == null || r[TARGET] == null) continue;
    const s = ("" + r[SEX()]).toLowerCase().trim();
    const label = +r[TARGET] === 1 ? "survived" : "died";
    if (!counts[s]) counts[s] = { died: 0, survived: 0 };
    counts[s][label]++;
  }
  const container = { name: "Survival by Sex", tab: "EDA" };
  const data = [
    { name: "Died", values: barDataFromCounts(Object.fromEntries(Object.entries(counts).map(([k,v]) => [k, v.died]))) },
    { name: "Survived", values: barDataFromCounts(Object.fromEntries(Object.entries(counts).map(([k,v]) => [k, v.survived]))) },
  ];
  tfvis.render.barchart(container, data, { xLabel: "Sex", yLabel: "Count", height: 200 }, $("barSex"));
}

function chartSurvivalByPclass() {
  const counts = { "1": { died: 0, survived: 0 }, "2": { died: 0, survived: 0 }, "3": { died: 0, survived: 0 } };
  for (const r of rawTrain) {
    if (r["Pclass"] == null || r[TARGET] == null) continue;
    const cls = String(r["Pclass"]);
    const label = +r[TARGET] === 1 ? "survived" : "died";
    if (!counts[cls]) counts[cls] = { died: 0, survived: 0 };
    counts[cls][label]++;
  }
  const container = { name: "Survival by Pclass", tab: "EDA" };
  const data = [
    { name: "Died", values: barDataFromCounts(Object.fromEntries(Object.entries(counts).map(([k,v]) => [k, v.died]))) },
    { name: "Survived", values: barDataFromCounts(Object.fromEntries(Object.entries(counts).map(([k,v]) => [k, v.survived]))) },
  ];
  tfvis.render.barchart(container, data, { xLabel: "Pclass", yLabel: "Count", height: 200 }, $("barPclass"));
}

$("btnLoad").addEventListener("click", async () => {
  try {
    const trainFile = $("trainFile").files[0];
    if (!trainFile) { alert("Please choose train.csv"); return; }
    rawTrain = await readLocalCsv(trainFile);

    const testFile = $("testFile").files[0];
    if (testFile) {
      rawTest = await readLocalCsv(testFile);
    } else {
      rawTest = [];
    }

    // Preview
    renderTable("trainPreview", topRows(rawTrain, 8));
    showDatasetInfo();
    chartSurvivalBySex();
    chartSurvivalByPclass();
  } catch (e) {
    console.error(e);
    alert("Error loading CSV(s): " + e.message);
  }
});

$("btnReset").addEventListener("click", () => {
  rawTrain = [];
  rawTest = [];
  Xtrain?.dispose(); ytrain?.dispose(); Xval?.dispose(); yval?.dispose();
  model?.dispose();
  Xtrain = ytrain = Xval = yval = model = null;
  valProbs = valLabels = null;
  $("trainPreview").innerHTML = "";
  $("datasetInfo").innerHTML = "";
  $("barSex").innerHTML = "";
  $("barPclass").innerHTML = "";
  $("preOut").innerHTML = "";
  $("modelSummary").innerHTML = "";
  $("lossContainer").innerHTML = "";
  $("accContainer").innerHTML = "";
  $("rocContainer").innerHTML = "";
  $("aucNote").innerHTML = "";
  $("cmContainer").innerHTML = "";
  $("prfContainer").innerHTML = "";
  $("predictNote").innerHTML = "";
  $("trainStatus").textContent = "";
  $("thrSlider").value = 0.5;
  $("thrVal").textContent = "0.50";
});

// ====== PREPROCESSING ======

// Column helpers (allow reuse if you swap schema)
const SEX = () => "Sex";
const EMB = () => "Embarked";
const PCLASS = () => "Pclass";
const AGE = () => "Age";
const FARE = () => "Fare";
const SIBSP = () => "SibSp";
const PARCH = () => "Parch";

// Stats helpers
function median(arr) {
  const clean = arr.filter(v => v != null && !Number.isNaN(v)).sort((a,b) => a-b);
  if (!clean.length) return null;
  const mid = Math.floor(clean.length / 2);
  return clean.length % 2 ? clean[mid] : (clean[mid - 1] + clean[mid]) / 2;
}
function mode(arr) {
  const counts = new Map();
  for (const v of arr) if (v != null && v !== "") counts.set(v, (counts.get(v) || 0) + 1);
  let best = null, bestC = -1;
  for (const [k,c] of counts) if (c > bestC) { best = k; bestC = c; }
  return best ?? null;
}
function standardize(vec) {
  const clean = vec.filter(v => v != null && !Number.isNaN(+v)).map(Number);
  const mean = clean.reduce((a,b)=>a+b,0) / (clean.length || 1);
  const sd = Math.sqrt(clean.reduce((s,v)=>s+Math.pow(v-mean,2),0) / (clean.length || 1)) || 1;
  return { mean, sd };
}

function oneHot(values) {
  const uniq = Array.from(new Set(values.map(v => v == null ? "NA" : String(v))));
  const index = new Map(uniq.map((u,i) => [u, i]));
  return { categories: uniq, index };
}

let featureSpec = null; // will hold encoders & scalers
let featureNames = [];  // expanded after one-hot

function buildFeatureSpec(rows) {
  // compute imputations & encoders on TRAIN ONLY
  const ageMed = median(rows.map(r => r[AGE()]));
  const embMode = mode(rows.map(r => r[EMB()]));
  // engineered
  const useFamily = $("toggleFamily").checked;
  const useIsAlone = $("toggleIsAlone").checked;

  // one-hot categories (fit)
  const sexOH = oneHot(rows.map(r => r[SEX()]));
  const pclassOH = oneHot(rows.map(r => r[PCLASS()]));
  const embOH = oneHot(rows.map(r => r[EMB()]));

  // scalers (fit)
  const ageStd = standardize(rows.map(r => r[AGE()] ?? ageMed));
  const fareStd = standardize(rows.map(r => r[FARE()] ?? 0));

  return {
    imputations: { ageMed, embMode },
    onehot: { sexOH, pclassOH, embOH },
    scalers: { ageStd, fareStd },
    engineered: { useFamily, useIsAlone },
  };
}

function expandFeatureNames(spec) {
  const names = [];
  // standardized numeric
  names.push("Age_std", "Fare_std");
  // SibSp, Parch (raw numeric)
  names.push("SibSp", "Parch");
  // one-hot
  for (const c of spec.onehot.sexOH.categories) names.push(`Sex=${c}`);
  for (const c of spec.onehot.pclassOH.categories) names.push(`Pclass=${c}`);
  for (const c of spec.onehot.embOH.categories) names.push(`Embarked=${c}`);
  // engineered
  if (spec.engineered.useFamily) names.push("FamilySize");
  if (spec.engineered.useIsAlone) names.push("IsAlone");
  return names;
}

function rowToVector(r, spec) {
  const v = [];

  // imputations
  const age = (r[AGE()] == null || r[AGE()] === "") ? spec.imputations.ageMed : r[AGE()];
  const emb = (r[EMB()] == null || r[EMB()] === "") ? spec.imputations.embMode : r[EMB()];

  // standardized numerics
  const aStd = (Number(age) - spec.scalers.ageStd.mean) / spec.scalers.ageStd.sd;
  const fStd = (Number(r[FARE()] ?? 0) - spec.scalers.fareStd.mean) / spec.scalers.fareStd.sd;
  v.push(aStd, fStd);

  // raw counts
  v.push(Number(r[SIBSP()] ?? 0));
  v.push(Number(r[PARCH()] ?? 0));

  // one-hot Sex/Pclass/Embarked
  const sexKey = r[SEX()] == null ? "NA" : String(r[SEX()]);
  const pclassKey = r[PCLASS()] == null ? "NA" : String(r[PCLASS()]);
  const embKey = emb == null ? "NA" : String(emb);

  // helper to push one-hot vector
  function pushOneHot(key, oh) {
    for (let i = 0; i < oh.categories.length; i++) {
      v.push(i === (oh.index.get(key) ?? -1) ? 1 : 0);
    }
  }
  pushOneHot(sexKey, spec.onehot.sexOH);
  pushOneHot(pclassKey, spec.onehot.pclassOH);
  pushOneHot(embKey, spec.onehot.embOH);

  // engineered
  if (spec.engineered.useFamily) {
    const family = Number(r[SIBSP()] ?? 0) + Number(r[PARCH()] ?? 0) + 1;
    v.push(family);
  }
  if (spec.engineered.useIsAlone) {
    const family = Number(r[SIBSP()] ?? 0) + Number(r[PARCH()] ?? 0) + 1;
    v.push(family === 1 ? 1 : 0);
  }

  return v;
}

function stratifiedSplit(rows, testRatio = 0.2) {
  // group by TARGET (0/1), shuffle, then split
  const g0 = [], g1 = [];
  for (const r of rows) {
    const y = r[TARGET];
    if (y == null) continue;
    (+y === 1 ? g1 : g0).push(r);
  }
  const shuffle = (a) => a.sort(() => Math.random() - 0.5);
  shuffle(g0); shuffle(g1);

  const n0 = Math.floor(g0.length * (1 - testRatio));
  const n1 = Math.floor(g1.length * (1 - testRatio));

  const train = g0.slice(0, n0).concat(g1.slice(0, n1));
  const val = g0.slice(n0).concat(g1.slice(n1));
  shuffle(train); shuffle(val);
  return { train, val };
}

$("btnPreprocess").addEventListener("click", () => {
  try {
    if (!rawTrain.length) { alert("Load train.csv first."); return; }

    // Fit spec on full train then split (or split then fit-on-train only; both are OK for homework)
    const { train, val } = stratifiedSplit(rawTrain, 0.2);
    featureSpec = buildFeatureSpec(train);
    featureNames = expandFeatureNames(featureSpec);

    // Build tensors
    const X_tr = train.map(r => rowToVector(r, featureSpec));
    const y_tr = train.map(r => Number(r[TARGET]));
    const X_va = val.map(r => rowToVector(r, featureSpec));
    const y_va = val.map(r => Number(r[TARGET]));

    Xtrain?.dispose(); ytrain?.dispose(); Xval?.dispose(); yval?.dispose();
    Xtrain = tf.tensor2d(X_tr);
    ytrain = tf.tensor2d(y_tr, [y_tr.length, 1]);
    Xval = tf.tensor2d(X_va);
    yval = tf.tensor2d(y_va, [y_va.length, 1]);

    // Print shapes & feature list
    const lines = [];
    lines.push(`Features (${featureNames.length}): ${featureNames.join(", ")}`);
    lines.push(`Xtrain shape: ${Xtrain.shape}`);
    lines.push(`ytrain shape: ${ytrain.shape}`);
    lines.push(`Xval shape:   ${Xval.shape}`);
    lines.push(`yval shape:   ${yval.shape}`);
    $("preOut").textContent = lines.join("\n");
  } catch (e) {
    console.error(e);
    alert("Preprocessing error: " + e.message);
  }
});

// ====== MODEL ======
function buildModel(inputDim) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [inputDim] }));
  m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  m.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return m;
}

$("btnBuild").addEventListener("click", () => {
  if (!Xtrain) { alert("Run Preprocessing first."); return; }
  model?.dispose();
  model = buildModel(Xtrain.shape[1]);
  $("modelSummary").textContent = "Model built. Click 'Show Summary' to print details.";
});

$("btnSummary").addEventListener("click", () => {
  if (!model) { alert("Build the model first."); return; }
  const summary = [];
  model.summary( (line) => summary.push(line) );
  $("modelSummary").textContent = summary.join("\n");
});

// ====== TRAINING ======
$("btnTrain").addEventListener("click", async () => {
  try {
    if (!model || !Xtrain) { alert("Build model & preprocess first."); return; }
    $("trainStatus").textContent = "Training…";

    const fitCallbacks = tfvis.show.fitCallbacks(
      { name: "Training", tab: "Training" },
      ["loss", "val_loss", "acc", "val_acc"],
      { height: 200 },
      $("lossContainer")
    );
    // We’ll split callbacks across two containers for clarity:
    const accCallbacks = {
      onEpochEnd: (epoch, logs) => {
        // Draw accuracy chart in accContainer
        // tfjs-vis fitCallbacks draws to one surface; we mimic by rendering text updates here:
        const txt = `epoch ${epoch+1} — acc: ${logs.acc?.toFixed(4)} | val_acc: ${logs.val_acc?.toFixed(4)}`;
        $("accContainer").textContent += ( $("accContainer").textContent ? "\n" : "" ) + txt;
      }
    };

    // Early stopping
    const earlyStop = tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 5, restoreBestWeights: true });

    const history = await model.fit(Xtrain, ytrain, {
      epochs: 50,
      batchSize: 32,
      validationData: [Xval, yval],
      callbacks: [fitCallbacks, accCallbacks, earlyStop],
      shuffle: true,
    });

    $("trainStatus").textContent = `Done. Best val_loss: ${Math.min(...history.history.val_loss).toFixed(4)}`;

    // Cache validation probs/labels for metrics section
    const probs = model.predict(Xval);
    valProbs = (await probs.data()).slice(); // Float32Array
    probs.dispose();
    const yv = await yval.data();
    valLabels = Int32Array.from(yv);

    renderRocAndAuc();
    updateThresholdUI(lastThreshold);
  } catch (e) {
    console.error(e);
    alert("Training error: " + e.message);
  }
});

// ====== METRICS (ROC, AUC, CM, PRF) ======
function rocPoints(labels, probs, steps = 101) {
  // Compute ROC by sweeping thresholds from 0->1
  const pts = [];
  for (let i = 0; i < steps; i++) {
    const thr = i / (steps - 1);
    const { tp, fp, tn, fn } = confusion(labels, probs, thr);
    const tpr = tp / (tp + fn || 1);
    const fpr = fp / (fp + tn || 1);
    pts.push({ tpr, fpr, thr });
  }
  // sort by fpr
  pts.sort((a,b) => a.fpr - b.fpr);
  return pts;
}

function aucFromRoc(pts) {
  // trapezoidal integration
  let auc = 0;
  for (let i = 1; i < pts.length; i++) {
    const x1 = pts[i-1].fpr, y1 = pts[i-1].tpr;
    const x2 = pts[i].fpr,   y2 = pts[i].tpr;
    auc += (x2 - x1) * (y1 + y2) / 2;
  }
  return +auc.toFixed(4);
}

function confusion(labels, probs, thr) {
  let tp=0, fp=0, tn=0, fn=0;
  for (let i=0;i<labels.length;i++) {
    const pred = probs[i] >= thr ? 1 : 0;
    const y = labels[i];
    if (pred === 1 && y === 1) tp++;
    else if (pred === 1 && y === 0) fp++;
    else if (pred === 0 && y === 0) tn++;
    else fn++;
  }
  return { tp, fp, tn, fn };
}

function precisionRecallF1(tp, fp, tn, fn) {
  const precision = tp / (tp + fp || 1);
  const recall = tp / (tp + fn || 1);
  const f1 = 2 * precision * recall / (precision + recall || 1);
  return {
    precision: +precision.toFixed(4),
    recall: +recall.toFixed(4),
    f1: +f1.toFixed(4)
  };
}

function renderRocAndAuc() {
  if (!valProbs || !valLabels) return;
  const pts = rocPoints(valLabels, valProbs, 201);
  const series = pts.map(p => ({ x: p.fpr, y: p.tpr }));
  const container = { name: "ROC", tab: "Metrics" };
  tfvis.render.scatterplot(container, { values: series }, {
    xLabel: "False Positive Rate",
    yLabel: "True Positive Rate",
    height: 220
  }, $("rocContainer"));
  const auc = aucFromRoc(pts);
  $("aucNote").textContent = `AUC = ${auc}`;
}

function updateThresholdUI(thr) {
  $("thrVal").textContent = thr.toFixed(2);
  if (!valProbs || !valLabels) {
    $("cmContainer").textContent = "Train first to view metrics.";
    $("prfContainer").textContent = "";
    return;
  }
  const { tp, fp, tn, fn } = confusion(valLabels, valProbs, thr);
  const { precision, recall, f1 } = precisionRecallF1(tp, fp, tn, fn);

  $("cmContainer").innerHTML =
    `Confusion Matrix @ thr=${thr.toFixed(2)}\n\n` +
    `TP: ${tp}    FP: ${fp}\n` +
    `FN: ${fn}    TN: ${tn}`;

  $("prfContainer").innerHTML =
    `Precision: ${precision}    Recall: ${recall}    F1: ${f1}`;
}

$("thrSlider").addEventListener("input", (e) => {
  lastThreshold = Number(e.target.value);
  updateThresholdUI(lastThreshold);
});

// ====== PREDICTION & EXPORT ======
function ensureReadyForPredict() {
  if (!model) { alert("Build & train the model first."); return false; }
  if (!rawTest.length) { alert("Load test.csv first."); return false; }
  if (!featureSpec) { alert("Run Preprocessing first (it defines encoders)."); return false; }
  return true;
}

$("btnPredict").addEventListener("click", async () => {
  try {
    if (!ensureReadyForPredict()) return;
    // Make X for test using the TRAIN-FITTED spec
    const Xtest = tf.tensor2d(rawTest.map(r => rowToVector(r, featureSpec)));
    const p = model.predict(Xtest);
    const probs = Array.from(await p.data());
    p.dispose(); Xtest.dispose();

    // Cache note and preview
    $("predictNote").textContent = `Predicted ${probs.length} probabilities for test.csv. Use export buttons below.`;

    // Store on window for export buttons (simple pattern)
    window.__titanic_probs__ = probs;
  } catch (e) {
    console.error(e);
    alert("Prediction error: " + e.message);
  }
});

function downloadCsv(filename, rows, header) {
  const esc = (v) => {
    if (v == null) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s;
  };
  const csv = [header.join(",")]
    .concat(rows.map(r => header.map(h => esc(r[h])).join(",")))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

$("btnExportSub").addEventListener("click", () => {
  try {
    if (!rawTest.length) { alert("Load test.csv first."); return; }
    const probs = window.__titanic_probs__;
    if (!probs) { alert("Run Predict first."); return; }
    const thr = lastThreshold;
    const rows = rawTest.map((r, i) => ({
      PassengerId: r[IDENT],
      Survived: probs[i] >= thr ? 1 : 0
    }));
    downloadCsv("submission.csv", rows, ["PassengerId", "Survived"]);
  } catch (e) {
    console.error(e);
    alert("Export error: " + e.message);
  }
});

$("btnExportProbs").addEventListener("click", () => {
  try {
    const probs = window.__titanic_probs__;
    if (!probs || !rawTest.length) { alert("Run Predict first."); return; }
    const rows = rawTest.map((r, i) => ({
      PassengerId: r[IDENT],
      Probability: +probs[i].toFixed(6)
    }));
    downloadCsv("probabilities.csv", rows, ["PassengerId", "Probability"]);
  } catch (e) {
    console.error(e);
    alert("Export error: " + e.message);
  }
});

$("btnSaveModel").addEventListener("click", async () => {
  try {
    if (!model) { alert("No model to save. Train first."); return; }
    await model.save("downloads://titanic-tfjs");
  } catch (e) {
    console.error(e);
    alert("Save error: " + e.message);
  }
});
