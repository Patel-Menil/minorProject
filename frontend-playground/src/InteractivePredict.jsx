import { useEffect, useMemo, useState } from 'react';
import { api, apiPost, fmt, pct, riskColor } from './api.js';
import { ErrorBox, Strip } from './DemoViews.jsx';

// Interactive prediction with the trained mortality + sepsis checkpoint (results/minor_project_sepsis/).
// Field list, defaults, bounds, units, example and model facts all come from /api/sepsis/input-schema;
// predictions come only from POST /api/sepsis/predict-input. Nothing here is hardcoded or simulated.

const DISCLAIMER = 'Academic demonstration only. The manually entered scenario is hypothetical, and these predictions are not medical advice or a clinically validated diagnosis.';
const TARGET_LABELS = { hospital_expired: 'In-hospital mortality', sepsis_after_24h: 'Sepsis documented after 24 hours' };

const str = v => (v === null || v === undefined ? '' : String(v));

function baselineValues(schema) {
  return {
    tabular: Object.fromEntries(schema.tabular.map(s => [s.name, str(s.default)])),
    timeseries: Object.fromEntries(schema.timeseries.map(s => [s.name, Array(schema.bins.length).fill(str(s.default))])),
  };
}

// '' = not measured (labs only); returns { value, error }
function parseField(spec, text) {
  if (text.trim() === '') return spec.kind === 'lab' ? { value: null } : { error: 'required' };
  if (!/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(text.trim())) return { error: 'not a number' };
  const v = Number(text);
  if (!Number.isFinite(v)) return { error: 'must be finite' };
  if (spec.kind === 'binary' && v !== 0 && v !== 1) return { error: 'must be 0 or 1' };
  if (v < spec.min || v > spec.max) return { error: `outside training range ${spec.min}–${spec.max}` };
  return { value: v };
}

function NumberField({ spec, text, base, onChange, error, compact }) {
  const edited = text !== base;
  if (spec.kind === 'binary') {
    return (
      <label className={`field-check ${edited ? 'edited' : ''}`} title={spec.doc || spec.label}>
        <input type="checkbox" checked={text === '1'} onChange={e => onChange(e.target.checked ? '1' : '0')} />
        {!compact && <span>{spec.label}</span>}
      </label>
    );
  }
  return (
    <input className={`field-input ${edited ? 'edited' : ''} ${error ? 'invalid' : ''}`} value={text} inputMode="decimal"
      placeholder={spec.kind === 'lab' ? 'not measured' : ''} aria-label={spec.label} aria-invalid={!!error}
      title={`${spec.doc || spec.label}${spec.unit ? ` (${spec.unit})` : ''} · training range ${spec.min}–${spec.max}${error ? ` · ${error}` : ''}`}
      onChange={e => onChange(e.target.value)} />
  );
}

function TabularGrid({ specs, values, base, errors, setTab }) {
  return (
    <div className="field-grid">
      {specs.map(s => (
        <div key={s.name} className="field">
          {s.kind !== 'binary' && <label className="field-label">{s.label}{s.unit ? ` (${s.unit})` : ''}</label>}
          <NumberField spec={s} text={values.tabular[s.name]} base={base.tabular[s.name]} error={errors[s.name]}
            onChange={v => setTab(s.name, v)} />
          <div className="field-hint">{errors[s.name] ? <span className="err">{errors[s.name]}</span>
            : values.tabular[s.name] === base.tabular[s.name] ? 'baseline (training median)' : 'edited'}</div>
        </div>
      ))}
    </div>
  );
}

function SeriesTable({ specs, bins, values, base, errors, setTs }) {
  return (
    <div className="series-wrap">
      <table className="table compact series">
        <thead><tr><th>Feature</th>{bins.map(b => <th key={b}>{b}</th>)}</tr></thead>
        <tbody>{specs.map(s => (
          <tr key={s.name}>
            <td title={s.doc || ''}>{s.label}{s.unit ? <span className="muted"> ({s.unit})</span> : s.kind === 'lab' ? <span className="muted"> (unit not recorded)</span> : null}</td>
            {bins.map((b, i) => (
              <td key={b}><NumberField compact spec={s} text={values.timeseries[s.name][i]} base={base.timeseries[s.name][i]}
                error={errors[`${s.name}|${i}`]} onChange={v => setTs(s.name, i, v)} /></td>
            ))}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function TargetCard({ name, t }) {
  const label = TARGET_LABELS[name] || name;
  const vthr = t.validation_chosen_threshold;
  return (
    <div className="card result-card">
      <div className="section-title">{label}</div>
      <div className="big-prob" style={{ color: riskColor(t.probability) }}>{(t.probability * 100).toFixed(1)}%</div>
      <div className="prob-track"><div className="prob-fill" style={{ width: pct(t.probability), background: riskColor(t.probability) }} /></div>
      <table className="table compact"><tbody>
        <tr><td>Predicted class (threshold {t.threshold})</td><td><strong>{t.predicted_class ? 'positive' : 'negative'}</strong></td></tr>
        <tr><td>Raw model probability</td><td><code>{t.probability}</code></td></tr>
        {vthr !== null && vthr !== undefined && (
          <tr><td>Reference only: validation-chosen threshold {fmt(vthr)}</td><td>{t.probability >= vthr ? 'above' : 'below'}</td></tr>)}
      </tbody></table>
      <p className="muted">A probability from the model, not a diagnosis.</p>
    </div>
  );
}

function Pipeline({ arch }) {
  const steps = [
    ['Tabular input', `${arch.tabular_input_dim} features`], ['MLP encoder', arch.tabular_encoder],
    ['Time-series input', `${arch.timeseries_bins} bins × ${arch.timeseries_input_dim}`], ['LSTM encoder', arch.timeseries_encoder],
    ['MultimodalFusion', arch.fusion], ['k-NN patient graph', arch.graph], ['GraphSAGE', arch.gnn],
    ['Two prediction heads', arch.heads.join(' + ')],
  ];
  return <div className="flow">{steps.map(([t, s]) => <div key={t} className="flow-box"><div className="flow-title">{t}</div><div className="muted">{s}</div></div>)}</div>;
}

function Metrics({ model }) {
  const m = model.test_metrics, v = model.validation_chosen_threshold;
  const rows = [['mort', 'In-hospital mortality'], ['sepsis', 'Sepsis documented after 24 h']];
  return (
    <>
      <p className="muted">Measured on {model.test_patients.toLocaleString()} stays of {model.test_hospitals} held-out test hospitals.
        AUROC and AUPRC rank patients and do not depend on a threshold; precision, recall, F1 and specificity do.</p>
      <table className="table">
        <thead><tr><th>Task</th><th>Prevalence</th><th>AUROC</th><th>AUPRC</th><th>Brier</th><th>Precision @0.5</th><th>Recall @0.5</th><th>F1 @0.5</th><th>Specificity @0.5</th></tr></thead>
        <tbody>{rows.map(([k, l]) => (
          <tr key={k}><td>{l}</td><td>{pct(m[k].prevalence)}</td><td>{fmt(m[k].auroc)}</td><td>{fmt(m[k].auprc)}</td><td>{fmt(m[k].brier)}</td>
            <td>{fmt(m[k].precision)}</td><td>{fmt(m[k].recall)}</td><td>{fmt(m[k].f1)}</td><td>{fmt(m[k].specificity)}</td></tr>
        ))}</tbody>
      </table>
      {v && (
        <table className="table">
          <thead><tr><th>Reference: threshold chosen on validation hospitals (best F1)</th><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
          <tbody>{rows.map(([k, l]) => (
            <tr key={k}><td>{l}</td><td>{fmt(v[k].threshold)}</td><td>{fmt(v[k].test.precision)}</td><td>{fmt(v[k].test.recall)}</td><td>{fmt(v[k].test.f1)}</td></tr>
          ))}</tbody>
        </table>
      )}
      <p className="warn-text">At the 0.5 threshold the model predicts no sepsis cases in the test set; sepsis ranking is moderate (AUROC) but
        detection of this rare proxy label is weak (AUPRC, precision, recall).</p>
    </>
  );
}

export default function InteractivePredict() {
  const [schema, setSchema] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [values, setValues] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showPipeline, setShowPipeline] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const [result, setResult] = useState(null);
  const [predictError, setPredictError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [requests, setRequests] = useState(0);
  const [stale, setStale] = useState(false);

  useEffect(() => {
    api('/api/sepsis/input-schema').then(s => { setSchema(s); setValues(baselineValues(s)); })
      .catch(e => setLoadError(e.message));
  }, []);

  const base = useMemo(() => (schema ? baselineValues(schema) : null), [schema]);
  const { payload, errors } = useMemo(() => {
    if (!schema || !values) return { payload: null, errors: {} };
    const errs = {}, tab = {}, ts = {};
    for (const s of schema.tabular) {
      const r = parseField(s, values.tabular[s.name]);
      if (r.error) errs[s.name] = r.error; else tab[s.name] = r.value;
    }
    for (const s of schema.timeseries) {
      ts[s.name] = values.timeseries[s.name].map((text, i) => {
        const r = parseField(s, text);
        if (r.error) errs[`${s.name}|${i}`] = r.error;
        return r.error ? null : r.value;
      });
    }
    return { payload: { tabular: tab, timeseries: ts }, errors: errs };
  }, [schema, values]);

  if (loadError) {
    return (
      <div className="page">
        <ErrorBox message={loadError} />
        <div className="card"><h3>Interactive prediction not available</h3>
          <p className="muted">This page uses only the trained mortality + sepsis checkpoint in results/minor_project_sepsis/. It does not fall
            back to another model. Train it with <code>python scripts/final_model.py train --task mortality_sepsis</code> and restart the server.</p></div>
      </div>
    );
  }
  if (!schema || !values) return <div className="page muted">Loading model and form (first load verifies the checkpoint)…</div>;

  const edit = fn => { setValues(v => fn(structuredClone(v))); setStale(true); };
  const setTab = (n, v) => edit(x => { x.tabular[n] = v; return x; });
  const setTs = (n, i, v) => edit(x => { x.timeseries[n][i] = v; return x; });
  const loadExample = () => edit(() => {
    const x = baselineValues(schema);
    for (const [k, v] of Object.entries(schema.example.tabular)) x.tabular[k] = str(v);
    for (const [k, vals] of Object.entries(schema.example.timeseries)) x.timeseries[k] = vals.map(str);
    return x;
  });
  const reset = () => { setValues(baselineValues(schema)); setResult(null); setPredictError(null); setStale(false); };
  const nErrors = Object.keys(errors).length;
  const predict = async () => {
    setBusy(true); setPredictError(null);
    try {
      const r = await apiPost('/api/sepsis/predict-input', { inputs: payload, threshold: 0.5 });
      setResult(r); setRequests(n => n + 1); setStale(false);
    } catch (e) {
      setResult(null); setPredictError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const model = schema.model;
  const quickTab = schema.tabular.filter(s => s.quick), quickTs = schema.timeseries.filter(s => s.quick);
  const editedCount = schema.tabular.filter(s => values.tabular[s.name] !== base.tabular[s.name]).length
    + schema.timeseries.reduce((n, s) => n + values.timeseries[s.name].filter((t, i) => t !== base.timeseries[s.name][i]).length, 0);

  return (
    <div className="page">
      <div className="card band-historical">
        <h2>Interactive prediction: mortality + sepsis model</h2>
        <p>Enter a <strong>hypothetical</strong> ICU patient's first 24 hours. The values are processed by the federated GNN trained on real eICU
          data: the tabular features go through an MLP encoder, the six 4-hour bins through an LSTM, both embeddings are combined by
          MultimodalFusion, the fused patient joins a k-NN graph with {schema.reference.context_patients} training-split patients, and GraphSAGE feeds
          two prediction heads.</p>
        <div className="status-row">
          <span className="status-pill ok">✓ {model.status}</span>
          <span className="tag">{model.checkpoint}</span><span className="tag">SHA-256 {model.sha256_prefix}…</span>
          <span className="tag">round {model.selected_round} of {model.rounds_completed}</span>
        </div>
        <p className="warn-text">{DISCLAIMER}</p>
      </div>

      <div className="card controls">
        <button onClick={loadExample}>Load example</button>
        <button onClick={reset}>Reset to reference values</button>
        <button className="primary" disabled={busy || nErrors > 0} onClick={predict}>{busy ? 'Predicting…' : 'Predict'}</button>
        <span className="divider" />
        <button onClick={() => setShowPipeline(s => !s)} aria-pressed={showPipeline}>Show model pipeline</button>
        <button onClick={() => setShowMetrics(s => !s)} aria-pressed={showMetrics}>Show evaluation metrics</button>
        <span className="muted">{editedCount} value(s) differ from the training-median baseline{nErrors ? ` · ${nErrors} invalid` : ''}</span>
      </div>

      {showPipeline && <div className="card"><div className="section-title">Model pipeline (from the checkpoint manifest)</div><Pipeline arch={model.architecture} /></div>}
      {showMetrics && <div className="card"><div className="section-title">Held-out test evaluation of this checkpoint</div><Metrics model={model} /></div>}

      <div className="grid-2 wide-left">
        <div className="card">
          <div className="section-title">Patient (tabular inputs, quick fields)</div>
          <TabularGrid specs={quickTab} values={values} base={base} errors={errors} setTab={setTab} />
          <div className="section-title">First 24 hours (time-series, quick fields)</div>
          <p className="muted">Labs: mean value in each 4-hour bin, in the unit recorded in eICU; leave empty for "not measured". Checkboxes: recorded in that bin.</p>
          <SeriesTable specs={quickTs} bins={schema.bins} values={values} base={base} errors={errors} setTs={setTs} />
          <p className="muted">Fields not shown here keep their baseline (training-split median) values; open the advanced section to edit them.</p>
        </div>

        <div className="result-col">
          {busy && <div className="card muted">Running the trained checkpoint…</div>}
          {predictError && <ErrorBox message={predictError} />}
          {result && (
            <>
              {stale && <div className="card warn-text">Inputs changed since this prediction. Press Predict to run the model again.</div>}
              <TargetCard name="hospital_expired" t={result.targets.hospital_expired} />
              <TargetCard name="sepsis_after_24h" t={result.targets.sepsis_after_24h} />
              <div className="card">
                <div className="section-title">What ran (request #{requests})</div>
                <ul className="plain">
                  <li>Checkpoint verified: {result.inference.checkpoint_verified ? 'yes' : 'no'} ({result.inference.checkpoint_sha256_prefix}…, round {result.checkpoint_round})</li>
                  <li>Inputs: tabular {result.inference.tabular_input_shape.join(' × ')}, time-series {result.inference.timeseries_input_shape.join(' × ')}</li>
                  <li>Fusion executed: {result.inference.fusion_executed ? 'yes' : 'no'}; GraphSAGE input = fused embedding: {result.inference.graphsage_input_is_fused ? 'yes' : 'no'}</li>
                  <li>Graph: {result.inference.graph_edges} edges, {result.inference.neighbours_of_input} into this patient ({result.inference.batch})</li>
                  <li>{result.inference.values_differing_from_baseline} submitted value(s) differ from the baseline</li>
                </ul>
                {[['MLP embedding', 'tab'], ['LSTM embedding', 'ts'], ['Fused embedding', 'fused'], ['GraphSAGE embedding', 'graph']].map(([l, k]) => (
                  <div key={k} className="flow-box"><div className="flow-title">{l} ({result.embeddings[k].length}-D)</div><Strip values={result.embeddings[k]} /></div>
                ))}
              </div>
            </>
          )}
          {!result && !busy && !predictError && <div className="card muted">Edit the inputs or load the example, then press Predict.</div>}
          <div className="card">
            <div className="section-title">Model information</div>
            <table className="table compact"><tbody>
              <tr><td>Model</td><td>FedAvg + GraphSAGE ({model.architecture.gnn})</td></tr>
              <tr><td>Input modalities</td><td>tabular ({model.architecture.tabular_input_dim}) + time-series ({model.architecture.timeseries_bins} × {model.architecture.timeseries_input_dim})</td></tr>
              <tr><td>Fusion</td><td>MultimodalFusion {model.architecture.fusion}</td></tr>
              <tr><td>Federated training</td><td>FedAvg, {model.rounds_completed} of {model.rounds_configured} rounds, {model.clients_per_round} clients/round, seed {model.seed}</td></tr>
              <tr><td>Selected checkpoint</td><td>round {model.selected_round} (validation loss {fmt(model.best_val_loss, 4)})</td></tr>
              <tr><td>Test cohort</td><td>{model.test_patients.toLocaleString()} stays, {model.test_hospitals} hospitals</td></tr>
              <tr><td>Mortality AUROC / AUPRC</td><td>{fmt(model.test_metrics.mort.auroc)} / {fmt(model.test_metrics.mort.auprc)}</td></tr>
              <tr><td>Sepsis AUROC / AUPRC</td><td>{fmt(model.test_metrics.sepsis.auroc)} / {fmt(model.test_metrics.sepsis.auprc)}</td></tr>
            </tbody></table>
            {model.label_definition && <p className="muted"><strong>Sepsis label:</strong> {model.label_definition.positive}; {model.label_definition.excluded}. {model.label_definition.limitation}.</p>}
          </div>
        </div>
      </div>

      <div className="card">
        <button onClick={() => setShowAdvanced(s => !s)} aria-expanded={showAdvanced}>{showAdvanced ? '▾' : '▸'} Advanced model inputs (all {schema.tabular.length} tabular + {schema.timeseries.length} time-series features)</button>
        {showAdvanced && (
          <>
            <p className="muted">Exact checkpoint order. Defaults are training-split medians; bounds are the 0.1st–99.9th percentile of training values
              (hover a field). Tabular values are z-scored with the statistics of a fixed reference training hospital, exactly as
              the training loader scales each hospital; labs are converted with the global lab statistics used to build the training data.</p>
            <div className="section-title">Tabular ({schema.tabular.length})</div>
            <TabularGrid specs={schema.tabular} values={values} base={base} errors={errors} setTab={setTab} />
            <div className="section-title">Time-series ({schema.timeseries.length} per bin)</div>
            <SeriesTable specs={schema.timeseries} bins={schema.bins} values={values} base={base} errors={errors} setTs={setTs} />
          </>
        )}
      </div>
      <p className="muted">{DISCLAIMER} The example is a hypothetical scenario, not a patient record.</p>
    </div>
  );
}
