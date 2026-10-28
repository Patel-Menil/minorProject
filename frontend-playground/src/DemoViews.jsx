import { useEffect, useState } from 'react';
import { api, fmt, pct } from './api.js';
import { ProbRow } from './GraphView.jsx';

function useApi(path) {
  const [state, setState] = useState({ data: null, error: null });
  useEffect(() => {
    let live = true;
    api(path).then(data => live && setState({ data, error: null }))
      .catch(err => live && setState({ data: null, error: err.message }));
    return () => { live = false; };
  }, [path]);
  return state;
}

export function ErrorBox({ message }) {
  return <div className="error-box"><strong>Not available.</strong> {message}</div>;
}

// ── Overview ────────────────────────────────────────────────────────────────
export function Overview() {
  const { data, error } = useApi('/api/overview');
  if (error) return <div className="page"><ErrorBox message={error} /></div>;
  if (!data) return <div className="page muted">Loading…</div>;
  const hosp = Object.entries(data.dataset.hospitals);
  const total = hosp.reduce((s, [, h]) => s + h.patients, 0);
  const cohort = hosp.reduce((s, [, h]) => s + h.stays_24h_or_longer, 0);
  return (
    <div className="page">
      <div className="card">
        <h2>{data.project}</h2>
        <p>{data.purpose}</p>
      </div>
      <div className="grid-2">
        <div className="card">
          <div className="section-title">Implemented pipeline</div>
          <ol className="steps">
            {data.pipeline.map(s => (
              <li key={s.step}><strong>{s.step}</strong> <span className="tag">{s.shape}</span><div className="muted">{s.module}</div></li>
            ))}
          </ol>
          <p className="muted">Targets: in-hospital mortality (<code>hospital_expired</code>) and ICU mortality
            (<code>icu_expired</code>), from the first 24 h of the ICU stay (six 4-hour bins), for stays of at least 24 h.</p>
        </div>
        <div className="card">
          <div className="section-title">Synthetic demo dataset</div>
          <div className="stats-grid">
            <Stat value={data.dataset.n_hospitals} label="Synthetic hospitals" />
            <Stat value={total} label="Synthetic ICU stays" />
            <Stat value={cohort} label="In 24 h cohort" />
            <Stat value={`${data.dataset.tabular_features} + ${data.dataset.timeseries_features}`} label="Tabular + series features" />
          </div>
          <div className="section-title">Hospital-level split</div>
          {['train', 'val', 'test'].map(k => (
            <div key={k} className="split-row"><span className={`role role-${k}`}>{k}</span>
              {data.split[k].map(h => <span key={h} className="chip">{h.replace('synth_hospital_', 'H')}</span>)}</div>
          ))}
          <div className="section-title">Demo checkpoint</div>
          <p className="muted">{data.model.run_name}, saved at round {data.model.round}, GNN {data.model.gnn_type}, k = {data.model.knn_k}, seed {data.model.seed}</p>
        </div>
      </div>
      <div className="card">
        <div className="section-title">Scope of the current implementation</div>
        <ul className="plain">{data.scope_limits.map(s => <li key={s}>{s}</li>)}</ul>
      </div>
    </div>
  );
}

function Stat({ value, label }) {
  return <div className="stat-box"><div className="value">{value}</div><div className="label">{label}</div></div>;
}

// ── Predict ─────────────────────────────────────────────────────────────────
function Strip({ values }) {
  const max = Math.max(...values.map(v => Math.abs(v)), 1e-6);
  return (
    <div className="strip" title={`${values.length} values`}>
      {values.map((v, i) => {
        const a = Math.min(Math.abs(v) / max, 1);
        return <span key={i} style={{ background: v >= 0 ? `rgba(59,130,246,${a})` : `rgba(239,68,68,${a})` }} />;
      })}
    </div>
  );
}

export function Predict() {
  const patients = useApi('/api/patients?split=test');
  const overview = useApi('/api/overview');
  const [selected, setSelected] = useState('');
  const [seed, setSeed] = useState(1);
  const [hospital, setHospital] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async (query) => {
    setBusy(true); setError(null);
    try {
      setResult(await api(`/api/predict?${query}`));
    } catch (err) {
      setResult(null); setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  if (patients.error) return <div className="page"><ErrorBox message={patients.error} /></div>;
  const list = patients.data?.patients || [];
  const hospitals = overview.data ? [...overview.data.split.test, ...overview.data.split.val, ...overview.data.split.train] : [];
  const pid = selected || list[0]?.patient_id || '';

  return (
    <div className="page">
      <div className="card controls">
        <label>Synthetic test patient
          <select value={pid} onChange={e => setSelected(e.target.value)}>
            {list.map(p => <option key={p.patient_id} value={p.patient_id}>{p.patient} · {p.hospital}</option>)}
          </select>
        </label>
        <button disabled={!pid || busy} onClick={() => run(`patient_id=${pid}`)}>Run model</button>
        <span className="divider" />
        <label>New synthetic patient, seed
          <input type="number" min="0" value={seed} onChange={e => setSeed(e.target.value)} />
        </label>
        <label>in hospital
          <select value={hospital} onChange={e => setHospital(e.target.value)}>
            <option value="">(first test hospital)</option>
            {hospitals.map(h => <option key={h} value={h}>{h}</option>)}
          </select>
        </label>
        <button disabled={busy} onClick={() => run(`new_patient=${seed}${hospital ? `&hospital=${hospital}` : ''}`)}>Generate + predict</button>
      </div>

      {error && <ErrorBox message={error} />}
      {busy && <p className="muted">Running the demo checkpoint…</p>}
      {result && <PredictionResult r={result} />}
      {!result && !error && !busy && <p className="muted">Choose a synthetic patient and run the model. Every number below is returned by the inference API from the saved demo checkpoint.</p>}
    </div>
  );
}

function PredictionResult({ r }) {
  const ts = r.inputs.timeseries;
  return (
    <>
      <div className="grid-2">
        <div className="card">
          <h3>{r.patient} <span className="tag">{r.hospital} · {r.hospital_split} split</span></h3>
          <p className="muted">{r.generated_on_request ? 'Generated on request by the synthetic generator.' : 'Synthetic patient from the demo dataset.'} ICU stay {r.inputs.unit_stay_hours} h.</p>
          <div className="section-title">Model output (demo checkpoint, round {r.model.round})</div>
          <ProbRow label="In-hospital mortality" p={r.p_in_hospital_mortality} />
          <ProbRow label="ICU mortality" p={r.p_icu_mortality} />
          <table className="table">
            <thead><tr><th>Task</th><th>Logit</th><th>Probability</th><th>Class at {r.threshold}</th><th>Synthetic label</th></tr></thead>
            <tbody>
              <tr><td>In-hospital mortality</td><td>{fmt(r.mortality_logit)}</td><td>{pct(r.p_in_hospital_mortality)}</td>
                <td>{r.pred_in_hospital_mortality ? 'positive' : 'negative'}</td><td>{r.synthetic_label ? (r.synthetic_label.hospital_expired ? 'died' : 'survived') : 'n/a (new patient)'}</td></tr>
              <tr><td>ICU mortality</td><td>{fmt(r.icu_logit)}</td><td>{pct(r.p_icu_mortality)}</td>
                <td>{r.pred_icu_mortality ? 'positive' : 'negative'}</td><td>{r.synthetic_label ? (r.synthetic_label.icu_expired ? 'died in ICU' : 'survived ICU') : 'n/a (new patient)'}</td></tr>
            </tbody>
          </table>
          <p className="warn-text">Demonstration model trained on synthetic data. These probabilities are not clinically meaningful.</p>
        </div>
        <div className="card">
          <div className="section-title">Tabular inputs (raw synthetic values, 15 of 67 shown)</div>
          <table className="table compact"><tbody>
            {Object.entries(r.inputs.tabular).map(([k, v]) => <tr key={k}><td>{k}</td><td>{fmt(v, 2)}</td></tr>)}
          </tbody></table>
        </div>
      </div>

      <div className="card">
        <div className="section-title">Temporal inputs: first 24 h in 4-hour bins (7 of 74 features shown)</div>
        <table className="table compact">
          <thead><tr><th>Feature</th>{ts.hours.map(h => <th key={h}>{h}</th>)}</tr></thead>
          <tbody>{Object.entries(ts.values).map(([k, vals]) => (
            <tr key={k}><td>{k}</td>{vals.map((v, i) => <td key={i}>{fmt(v, 1)}</td>)}</tr>
          ))}</tbody>
        </table>
        <p className="muted">{ts.hours.length} of 6 bins observed; missing bins are zero-padded and the LSTM reads only the observed length.</p>
      </div>

      <div className="card">
        <div className="section-title">Pipeline for this patient (actual tensors from the forward pass)</div>
        <div className="flow">
          <FlowBox title="Tabular input" sub={`${r.shapes.tabular_input[1]} features, z-scored per hospital`} />
          <FlowBox title="MLP encoder" sub={`${r.shapes.tabular_emb}-D`} values={r.embeddings.tabular} />
          <FlowBox title="Time-series input" sub={`${r.shapes.timeseries_input[1]} bins × ${r.shapes.timeseries_input[2]}`} />
          <FlowBox title="LSTM encoder" sub={`${r.shapes.timeseries_emb}-D`} values={r.embeddings.timeseries} />
          <FlowBox title="Fusion" sub={`${r.shapes.fused_emb}-D`} values={r.embeddings.fused} />
          <FlowBox title="k-NN graph + GraphSAGE" sub={`${r.shapes.graph_emb}-D, batch of ${r.batch_size}`} values={r.embeddings.graph} />
          <FlowBox title="Two heads" sub={`${pct(r.p_in_hospital_mortality)} / ${pct(r.p_icu_mortality)}`} />
        </div>
        <div className="section-title">Graph neighbours (messages received by this patient)</div>
        {r.neighbours.length ? (
          <div className="chips">{r.neighbours.map(n => <span key={n.row} className="chip">{n.patient} · cos {fmt(n.similarity, 2)}</span>)}</div>
        ) : <p className="muted">No graph edges (No-GNN model).</p>}
        <p className="muted">Edges are built from the fused embeddings of the {r.batch_size} patients in this patient's batch, all from the same hospital: no cross-hospital edges.</p>
      </div>
    </>
  );
}

function FlowBox({ title, sub, values }) {
  return (
    <div className="flow-box">
      <div className="flow-title">{title}</div>
      <div className="muted">{sub}</div>
      {values && <Strip values={values} />}
    </div>
  );
}

// ── Federation ──────────────────────────────────────────────────────────────
function LineChart({ series, height = 140 }) {
  const w = 520, pad = 28;
  const n = series[0].values.length;
  const all = series.flatMap(s => s.values);
  const lo = Math.min(...all), hi = Math.max(...all);
  const x = i => pad + (i * (w - 2 * pad)) / Math.max(n - 1, 1);
  const y = v => height - pad + ((v - lo) / (hi - lo || 1)) * (2 * pad - height);
  return (
    <svg viewBox={`0 0 ${w} ${height}`} className="chart" role="img" aria-label="Validation metrics per round">
      {series.map(s => (
        <g key={s.label}>
          <polyline fill="none" stroke={s.color} strokeWidth="2" points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(' ')} />
          {s.values.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r="3" fill={s.color}><title>{`${s.label}, round ${i + 1}: ${fmt(v, 4)}`}</title></circle>)}
        </g>
      ))}
      {Array.from({ length: n }, (_, i) => <text key={i} x={x(i)} y={height - 8} className="axis">{i + 1}</text>)}
    </svg>
  );
}

export function Federation() {
  const { data, error } = useApi('/api/federation');
  const [round, setRound] = useState(null);
  if (error) return <div className="page"><ErrorBox message={error} /></div>;
  if (!data) return <div className="page muted">Loading…</div>;
  const h = data.history;
  const sel = h[(round ?? h.length) - 1];
  return (
    <div className="page">
      <div className="card">
        <h3>FedAvg across synthetic hospitals</h3>
        <p className="muted">Each round the server samples {data.config.clients_per_round} of the {data.split.train.length} training hospitals.
          Each client trains a copy of the global model for {data.config.local_epochs} local epoch(s) on its own patients; only the weights
          return, and the server averages them weighted by the number of examples each client trained on
          (<code>fedavg_aggregate</code> in <code>scripts/train_federated.py</code>). The validation hospitals pick the checkpoint
          (round {data.selected_round}); the test hospitals are never used in training.</p>
      </div>
      <div className="grid-2">
        <div className="card">
          <div className="section-title">Validation hospitals, per round</div>
          <LineChart series={[
            { label: 'Val mortality AUROC', color: '#34d399', values: h.map(r => r.val_mort_auroc) },
            { label: 'Val ICU AUROC', color: '#60a5fa', values: h.map(r => r.val_icu_auroc) },
          ]} />
          <div className="legend-row"><span style={{ color: '#34d399' }}>● mortality AUROC</span><span style={{ color: '#60a5fa' }}>● ICU AUROC</span></div>
          <LineChart series={[{ label: 'Val loss', color: '#f59e0b', values: h.map(r => r.val_loss) }]} height={110} />
          <div className="legend-row"><span style={{ color: '#f59e0b' }}>● validation loss</span></div>
        </div>
        <div className="card">
          <div className="section-title">Round {sel.round} {sel.best ? '(new best checkpoint)' : ''}</div>
          <input type="range" min="1" max={h.length} value={sel.round} onChange={e => setRound(Number(e.target.value))} aria-label="Round" />
          <table className="table">
            <thead><tr><th>Client hospital</th><th>Status</th><th>Patients</th><th>Examples</th><th>Steps</th><th>FedAvg weight</th></tr></thead>
            <tbody>{sel.clients.map(c => (
              <tr key={c.hospital}><td>{c.hospital}</td><td>{c.status}</td><td>{c.patients}</td><td>{c.examples}</td><td>{c.steps}</td>
                <td><div className="weight"><span style={{ width: pct(c.fedavg_weight) }} />{pct(c.fedavg_weight)}</div></td></tr>
            ))}</tbody>
          </table>
          <p className="muted">Aggregated {sel.examples_trained} examples → global model. Val loss {fmt(sel.val_loss, 4)},
            val mortality AUROC {fmt(sel.val_mort_auroc)}, val ICU AUROC {fmt(sel.val_icu_auroc)}.</p>
        </div>
      </div>
    </div>
  );
}

// ── Evaluation ──────────────────────────────────────────────────────────────
const METRICS = [['auroc', 'AUROC'], ['auprc', 'AUPRC'], ['f1', 'F1 @0.5'], ['precision', 'Precision'], ['recall', 'Recall'], ['brier', 'Brier']];

export function Evaluation() {
  const { data, error } = useApi('/api/evaluation');
  if (error) return <div className="page"><ErrorBox message={error} /></div>;
  if (!data) return <div className="page muted">Loading…</div>;
  const runs = Object.entries(data.synthetic_demo.runs);
  const hist = data.historical_eicu.data;
  return (
    <div className="page">
      <div className="card band-synthetic">
        <h3>A. Synthetic demonstration results</h3>
        <p className="muted">{data.synthetic_demo.description} Labels are the synthetic generator's outcomes. These numbers show that the
          software pipeline learns; they say nothing about real patients.</p>
        {['mort', 'icu'].map(task => (
          <table key={task} className="table">
            <thead><tr><th>{task === 'mort' ? 'In-hospital mortality' : 'ICU mortality'}</th>{METRICS.map(([, l]) => <th key={l}>{l}</th>)}<th>Patients</th><th>Prevalence</th></tr></thead>
            <tbody>{runs.map(([name, r]) => (
              <tr key={name}><td>{name}</td>{METRICS.map(([k]) => <td key={k}>{fmt(r.test_metrics[`${task}_${k}`])}</td>)}
                <td>{r.test_metrics.n_patients}</td><td>{pct(r.test_metrics[`${task}_prevalence`])}</td></tr>
            ))}</tbody>
          </table>
        ))}
        <p className="muted">For reference, the untrained model scored validation mortality AUROC {fmt(runs[0][1].untrained_val_metrics.mort_auroc)}.
          Test hospitals: {runs[0][1].split.join(', ')}.</p>
      </div>
      <div className="card band-historical">
        <h3>B. Previously recorded eICU results (historical)</h3>
        <p className="muted">{data.historical_eicu.description}</p>
        {hist ? (
          <>
            <table className="table">
              <thead><tr><th>Model</th><th>Seeds</th><th>Mort AUROC</th><th>Mort AUPRC</th><th>ICU AUROC</th><th>ICU AUPRC</th></tr></thead>
              <tbody>{Object.entries(hist.summary).map(([name, s]) => (
                <tr key={name}><td>{name}</td><td>{s.n_seeds}</td>
                  {['mort_auroc', 'mort_auprc', 'icu_auroc', 'icu_auprc'].map(k => <td key={k}>{fmt(s[k].mean, 4)} ± {fmt(s[k].sd, 4)}</td>)}</tr>
              ))}</tbody>
            </table>
            <p className="muted">Source: {hist.source}. {hist.protocol?.evaluated_on}.</p>
          </>
        ) : <ErrorBox message="results/corrected/benchmark_results.json not found." />}
      </div>
    </div>
  );
}
