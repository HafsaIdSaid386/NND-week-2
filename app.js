// Titanic TF.js — final version (visor training + robust preprocessing)

let rawTrain = [], rawTest = [];
let Xtrain=null, ytrain=null, Xval=null, yval=null, model=null;
let valProbs=null, valLabels=null, lastThreshold=0.5;

const TARGET="Survived", IDENT="PassengerId";
const BASE_FEATURES=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"];

const $ = id => document.getElementById(id);

// ---------- CSV LOADING ----------
async function readLocalCsv(file){
  return new Promise((res,rej)=>{
    Papa.parse(file,{
      header:true, skipEmptyLines:true, dynamicTyping:true,
      complete:(r)=>res(r.data), error:rej
    });
  });
}
function topRows(arr,n=8){
  const keys=Object.keys(arr[0]||{}); 
  return {headers:keys, rows:arr.slice(0,n).map(r=>keys.map(k=>r[k]))};
}
function renderTable(divId,data){
  const el=$(divId);
  if(!data.headers?.length){ el.innerHTML="<em>No data</em>"; return; }
  const head=`<thead><tr>${data.headers.map(h=>`<th>${h}</th>`).join("")}</tr></thead>`;
  const body=`<tbody>${data.rows.map(r=>`<tr>${r.map(x=>`<td>${x??""}</td>`).join("")}</tr>`).join("")}</tbody>`;
  el.innerHTML=`<table>${head}${body}</table>`;
}
function missingPercentages(rows){
  if(!rows.length) return {};
  const keys=Object.keys(rows[0]), total=rows.length, miss={};
  for(const k of keys){ let c=0; for(const r of rows){ if(r[k]==null||r[k]==="") c++; } miss[k]=+(100*c/total).toFixed(2); }
  return miss;
}
function showDatasetInfo(){
  const out=[];
  if(rawTrain.length){
    out.push(`Train rows: ${rawTrain.length}`);
    const miss=missingPercentages(rawTrain); out.push(`Columns: ${Object.keys(rawTrain[0]).length}`); out.push("Missing % (train):");
    for(const [k,v] of Object.entries(miss)) out.push(`  - ${k}: ${v}%`);
  } else out.push("Train: (not loaded)");
  if(rawTest.length){
    out.push(""); out.push(`Test rows: ${rawTest.length}`); const missT=missingPercentages(rawTest); out.push("Missing % (test):");
    for(const [k,v] of Object.entries(missT)) out.push(`  - ${k}: ${v}%`);
  } else { out.push(""); out.push("Test: (not loaded)"); }
  $("datasetInfo").textContent=out.join("\n");
}

// ---------- Simple EDA charts ----------
const SEX = ()=>"Sex", EMB=()=>"Embarked", PCLASS=()=>"Pclass", AGE=()=>"Age", FARE=()=>"Fare", SIBSP=()=>"SibSp", PARCH=()=>"Parch";
function barDataFromCounts(map){ return Object.entries(map).map(([k,v])=>({index:k, value:v})); }
function chartSurvivalBySex(){
  const counts={female:{died:0,survived:0}, male:{died:0,survived:0}};
  for(const r of rawTrain){
    if(r[SEX()]==null || r[TARGET]==null) continue;
    const s=(String(r[SEX()]).toLowerCase().trim());
    const label=(+r[TARGET]===1)?"survived":"died";
    if(!counts[s]) counts[s]={died:0,survived:0}; counts[s][label]++;
  }
  const data=[
    {name:"Died", values:barDataFromCounts(Object.fromEntries(Object.entries(counts).map(([k,v])=>[k,v.died])) )},
    {name:"Survived", values:barDataFromCounts(Object.fromEntries(Object.entries(counts).map(([k,v])=>[k,v.survived])) )}
  ];
  tfvis.render.barchart({name:"Survival by Sex", tab:"EDA"}, data, {xLabel:"Sex", yLabel:"Count", height:200}, $("barSex"));
}
function chartSurvivalByPclass(){
  const counts={"1":{died:0,survived:0},"2":{died:0,survived:0},"3":{died:0,survived:0}};
  for(const r of rawTrain){
    if(r[PCLASS()]==null || r[TARGET]==null) continue;
    const c=String(r[PCLASS()]); const label=(+r[TARGET]===1)?"survived":"died"; if(!counts[c]) counts[c]={died:0,survived:0}; counts[c][label]++;
  }
  const data=[
    {name:"Died", values:barDataFromCounts(Object.fromEntries(Object.entries(counts).map(([k,v])=>[k,v.died])) )},
    {name:"Survived", values:barDataFromCounts(Object.fromEntries(Object.entries(counts).map(([k,v])=>[k,v.survived])) )}
  ];
  tfvis.render.barchart({name:"Survival by Pclass", tab:"EDA"}, data, {xLabel:"Pclass", yLabel:"Count", height:200}, $("barPclass"));
}

// ---------- UI: load/reset ----------
$("btnLoad").addEventListener("click", async ()=>{
  try{
    const tr=$("trainFile").files[0]; if(!tr){ alert("Please choose train.csv"); return; }
    rawTrain=await readLocalCsv(tr);
    const te=$("testFile").files[0]; rawTest=te? await readLocalCsv(te) : [];
    renderTable("trainPreview", topRows(rawTrain,8));
    showDatasetInfo(); chartSurvivalBySex(); chartSurvivalByPclass();
  }catch(e){ console.error(e); alert("Error loading CSV(s): "+e.message); }
});
$("btnReset").addEventListener("click", ()=>{
  rawTrain=[]; rawTest=[]; Xtrain?.dispose(); ytrain?.dispose(); Xval?.dispose(); yval?.dispose(); model?.dispose();
  Xtrain=ytrain=Xval=yval=model=null; valProbs=valLabels=null; lastThreshold=0.5;
  for(const id of ["trainPreview","datasetInfo","barSex","barPclass","preOut","modelSummary","lossContainer","accContainer","rocContainer","aucNote","cmContainer","prfContainer","predictNote"]) $(id).innerHTML="";
  $("thrSlider").value=0.5; $("thrVal").textContent="0.50"; $("trainStatus").textContent="";
});

// ---------- Preprocessing ----------
function median(arr){ const c=arr.filter(v=>v!=null && !Number.isNaN(v)).sort((a,b)=>a-b); if(!c.length) return null; const m=Math.floor(c.length/2); return c.length%2? c[m] : (c[m-1]+c[m])/2; }
function mode(arr){ const m=new Map(); for(const v of arr) if(v!=null && v!=="") m.set(v,(m.get(v)||0)+1); let best=null,cnt=-1; for(const [k,c] of m) if(c>cnt){best=k;cnt=c} return best??null; }
function standardize(vec){ const clean=vec.filter(v=>v!=null && !Number.isNaN(+v)).map(Number); const mean=clean.reduce((a,b)=>a+b,0)/(clean.length||1); const sd=Math.sqrt(clean.reduce((s,v)=>s+Math.pow(v-mean,2),0)/(clean.length||1))||1; return {mean,sd}; }
function oneHot(values){ const cats=Array.from(new Set(values.map(v=>v==null?"NA":String(v)))); const index=new Map(cats.map((c,i)=>[c,i])); return {categories:cats,index}; }

let featureSpec=null, featureNames=[];
function buildFeatureSpec(rows){
  let ageMed=median(rows.map(r=>r[AGE()])); if(ageMed==null||Number.isNaN(ageMed)) ageMed=28;
  let embMode=mode(rows.map(r=>r[EMB()])); if(embMode==null||embMode==="") embMode="S";
  const useFamily=$("toggleFamily").checked, useIsAlone=$("toggleIsAlone").checked;

  const sexOH=oneHot(rows.map(r=>r[SEX()])); const pclassOH=oneHot(rows.map(r=>r[PCLASS()])); const embOH=oneHot(rows.map(r=>r[EMB()]));
  const ageStd=standardize(rows.map(r=>r[AGE()]??ageMed)); const fareStd=standardize(rows.map(r=>r[FARE()]??0));

  return { imputations:{ageMed,embMode}, onehot:{sexOH,pclassOH,embOH}, scalers:{ageStd,fareStd}, engineered:{useFamily,useIsAlone} };
}
function expandFeatureNames(spec){
  const n=["Age_std","Fare_std","SibSp","Parch"];
  for(const c of spec.onehot.sexOH.categories) n.push(`Sex=${c}`);
  for(const c of spec.onehot.pclassOH.categories) n.push(`Pclass=${c}`);
  for(const c of spec.onehot.embOH.categories) n.push(`Embarked=${c}`);
  if(spec.engineered.useFamily) n.push("FamilySize");
  if(spec.engineered.useIsAlone) n.push("IsAlone");
  return n;
}
function rowToVector(r,s){
  const v=[];
  const age=(r[AGE()]==null||r[AGE()===""])? s.imputations.ageMed : r[AGE()];
  const emb=(r[EMB()]==null||r[EMB()===""])? s.imputations.embMode : r[EMB()];
  const aStd=(Number(age)-s.scalers.ageStd.mean)/s.scalers.ageStd.sd;
  const fStd=(Number(r[FARE()]??0)-s.scalers.fareStd.mean)/s.scalers.fareStd.sd;
  v.push(aStd, fStd);
  v.push(Number(r[SIBSP()]??0)); v.push(Number(r[PARCH()]??0));
  function pushOH(key,oh){ const k=(key==null?"NA":String(key)); for(let i=0;i<oh.categories.length;i++) v.push(i===(oh.index.get(k)??-1)?1:0); }
  pushOH(r[SEX()], s.onehot.sexOH); pushOH(r[PCLASS()], s.onehot.pclassOH); pushOH(emb, s.onehot.embOH);
  if(s.engineered.useFamily){ const fam=Number(r[SIBSP()]??0)+Number(r[PARCH()]??0)+1; v.push(fam); }
  if(s.engineered.useIsAlone){ const fam=Number(r[SIBSP()]??0)+Number(r[PARCH()]??0)+1; v.push(fam===1?1:0); }
  return v;
}
function stratifiedSplit(rows,ratio=0.2){
  const g0=[], g1=[]; for(const r of rows){ const y=r[TARGET]; if(y==null) continue; (+y===1?g1:g0).push(r); }
  const sh=a=>a.sort(()=>Math.random()-0.5); sh(g0); sh(g1);
  const n0=Math.floor(g0.length*(1-ratio)), n1=Math.floor(g1.length*(1-ratio));
  const train=g0.slice(0,n0).concat(g1.slice(0,n1)); const val=g0.slice(n0).concat(g1.slice(n1)); sh(train); sh(val); return {train,val};
}
$("btnPreprocess").addEventListener("click", ()=>{
  try{
    if(!rawTrain.length){ alert("Load train.csv first."); return; }
    const {train,val}=stratifiedSplit(rawTrain,0.2);
    featureSpec=buildFeatureSpec(train); featureNames=expandFeatureNames(featureSpec);
    const X_tr=train.map(r=>rowToVector(r,featureSpec)), y_tr=train.map(r=>Number(r[TARGET]));
    const X_va=val.map(r=>rowToVector(r,featureSpec)), y_va=val.map(r=>Number(r[TARGET]));
    Xtrain?.dispose(); ytrain?.dispose(); Xval?.dispose(); yval?.dispose();
    Xtrain=tf.tensor2d(X_tr); ytrain=tf.tensor2d(y_tr,[y_tr.length,1]);
    Xval=tf.tensor2d(X_va);   yval=tf.tensor2d(y_va,[y_va.length,1]);
    $("preOut").textContent=[
      `Features (${featureNames.length}): ${featureNames.join(", ")}`,
      `Xtrain shape: ${Xtrain.shape}`, `ytrain shape: ${ytrain.shape}`,
      `Xval shape:   ${Xval.shape}`,   `yval shape:   ${yval.shape}`,
    ].join("\n");
  }catch(e){ console.error(e); alert("Preprocessing error: "+e.message); }
});

// ---------- Model ----------
function buildModel(inputDim){
  const m=tf.sequential();
  m.add(tf.layers.dense({units:16, activation:"relu", inputShape:[inputDim]}));
  m.add(tf.layers.dense({units:1, activation:"sigmoid"}));
  m.compile({optimizer:"adam", loss:"binaryCrossentropy", metrics:["accuracy"]});
  return m;
}
$("btnBuild").addEventListener("click", ()=>{
  if(!Xtrain){ alert("Run Preprocessing first."); return; }
  model?.dispose(); model=buildModel(Xtrain.shape[1]);
  $("modelSummary").textContent="Model built. Click 'Show Summary' to print details.";
});
$("btnSummary").addEventListener("click", ()=>{
  if(!model){ alert("Build the model first."); return; }
  const lines=[]; model.summary(l=>lines.push(l)); $("modelSummary").textContent=lines.join("\n");
});

// ---------- Training (tfjs-vis visor, reliable across builds) ----------
$("btnTrain").addEventListener("click", async ()=>{
  try{
    if(!model || !Xtrain){ alert("Build model & preprocess first."); return; }
    $("trainStatus").textContent="Training…";
    const visorCbs=tfvis.show.fitCallbacks(
      {name:"Training", tab:"Training"},
      ["loss","val_loss","acc","val_acc"],
      {height:240}
    );
    const earlyStop=tf.callbacks.earlyStopping({monitor:"val_loss", patience:5, restoreBestWeights:true});
    const hist=await model.fit(Xtrain, ytrain, {
      epochs:50, batchSize:32, validationData:[Xval,yval],
      callbacks:[visorCbs, earlyStop], shuffle:true
    });
    $("trainStatus").textContent=`Done. Best val_loss: ${Math.min(...hist.history.val_loss).toFixed(4)}`;

    const p=model.predict(Xval); valProbs=(await p.data()).slice(); p.dispose();
    valLabels=Int32Array.from(await yval.data());
    renderRocAndAuc(); updateThresholdUI(lastThreshold);
  }catch(e){ console.error(e); alert("Training error: "+e.message); }
});

// ---------- Metrics (ROC/AUC + threshold) ----------
function confusion(labels, probs, thr){
  let tp=0,fp=0,tn=0,fn=0;
  for(let i=0;i<labels.length;i++){
    const pred=probs[i]>=thr?1:0, y=labels[i];
    if(pred===1&&y===1) tp++; else if(pred===1&&y===0) fp++; else if(pred===0&&y===0) tn++; else fn++;
  }
  return {tp,fp,tn,fn};
}
function rocPoints(labels, probs, steps=201){
  const pts=[]; for(let i=0;i<steps;i++){ const thr=i/(steps-1); const {tp,fp,tn,fn}=confusion(labels,probs,thr);
    const tpr=tp/(tp+fn||1), fpr=fp/(fp+tn||1); pts.push({tpr,fpr}); }
  pts.sort((a,b)=>a.fpr-b.fpr); return pts;
}
function aucFromRoc(pts){ let auc=0; for(let i=1;i<pts.length;i++){ const x1=pts[i-1].fpr,y1=pts[i-1].tpr,x2=pts[i].fpr,y2=pts[i].tpr; auc+= (x2-x1)*(y1+y2)/2; } return +auc.toFixed(4); }
function renderRocAndAuc(){
  if(!valProbs||!valLabels) return;
  const pts=rocPoints(valLabels,valProbs,201);
  tfvis.render.scatterplot({name:"ROC",tab:"Metrics"}, {values:pts.map(p=>({x:p.fpr,y:p.tpr}))}, {xLabel:"FPR",yLabel:"TPR",height:220}, $("rocContainer"));
  $("aucNote").textContent=`AUC = ${aucFromRoc(pts)}`;
}
function precisionRecallF1(tp,fp,tn,fn){
  const precision=tp/(tp+fp||1), recall=tp/(tp+fn||1), f1=2*precision*recall/(precision+recall||1);
  return {precision:+precision.toFixed(4), recall:+recall.toFixed(4), f1:+f1.toFixed(4)};
}
function updateThresholdUI(thr){
  $("thrVal").textContent=thr.toFixed(2);
  if(!valProbs||!valLabels){ $("cmContainer").textContent="Train first to view metrics."; $("prfContainer").textContent=""; return; }
  const {tp,fp,tn,fn}=confusion(valLabels,valProbs,thr);
  const {precision,recall,f1}=precisionRecallF1(tp,fp,tn,fn);
  $("cmContainer").textContent=`Confusion Matrix @ thr=${thr.toFixed(2)}\n\nTP: ${tp}    FP: ${fp}\nFN: ${fn}    TN: ${tn}`;
  $("prfContainer").textContent=`Precision: ${precision}    Recall: ${recall}    F1: ${f1}`;
}
$("thrSlider").addEventListener("input",(e)=>{ lastThreshold=Number(e.target.value); updateThresholdUI(lastThreshold); });

// ---------- Predict & Export ----------
function ensureReadyPredict(){ if(!model){alert("Train the model first.");return false} if(!rawTest.length){alert("Load test.csv first.");return false} if(!featureSpec){alert("Run Preprocessing first.");return false} return true; }
$("btnPredict").addEventListener("click", async ()=>{
  try{
    if(!ensureReadyPredict()) return;
    const Xtest=tf.tensor2d(rawTest.map(r=>rowToVector(r,featureSpec)));
    const p=model.predict(Xtest); const probs=Array.from(await p.data()); p.dispose(); Xtest.dispose();
    $("predictNote").textContent=`Predicted ${probs.length} probabilities for test.csv.`; window.__titanic_probs__=probs;
  }catch(e){ console.error(e); alert("Prediction error: "+e.message); }
});
function downloadCsv(filename, rows, header){
  const esc=v=>{ if(v==null) return ""; const s=String(v); return /[",\n]/.test(s)?`"${s.replace(/"/g,'""')}"`:s; };
  const csv=[header.join(",")].concat(rows.map(r=>header.map(h=>esc(r[h])).join(","))).join("\n");
  const blob=new Blob([csv],{type:"text/csv;charset=utf-8"}); const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url;a.download=filename;a.click();URL.revokeObjectURL(url);
}
$("btnExportSub").addEventListener("click", ()=>{
  try{
    if(!rawTest.length){alert("Load test.csv first.");return;}
    const probs=window.__titanic_probs__; if(!probs){alert("Run Predict first.");return;}
    const thr=lastThreshold;
    const rows=rawTest.map((r,i)=>({PassengerId:r[IDENT], Survived: probs[i]>=thr?1:0}));
    downloadCsv("submission.csv", rows, ["PassengerId","Survived"]);
  }catch(e){ console.error(e); alert("Export error: "+e.message); }
});
$("btnExportProbs").addEventListener("click", ()=>{
  try{
    const probs=window.__titanic_probs__; if(!probs||!rawTest.length){alert("Run Predict first.");return;}
    const rows=rawTest.map((r,i)=>({PassengerId:r[IDENT], Probability:+probs[i].toFixed(6)}));
    downloadCsv("probabilities.csv", rows, ["PassengerId","Probability"]);
  }catch(e){ console.error(e); alert("Export error: "+e.message); }
});
$("btnSaveModel").addEventListener("click", async ()=>{
  try{ if(!model){alert("No model to save.");return;} await model.save("downloads://titanic-tfjs"); }
  catch(e){ console.error(e); alert("Save error: "+e.message); }
});
