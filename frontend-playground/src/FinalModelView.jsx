import { useEffect, useState } from 'react';
import { api, fmt, pct } from './api.js';
import { ProbRow } from './GraphView.jsx';
import { ErrorBox, Strip } from './DemoViews.jsx';

// Real-data models (scripts/final_model.py):
//   kind="final"  -> mortality + ICU mortality, results/minor_project_final/, /api/final/*
//   kind="sepsis" -> mortality + sepsis documented after 24 h, results/minor_project_sepsis/, /api/sepsis/*
// There is no fallback to another model or to the synthetic demo.
const METRICS = [['auroc', 'AUROC'], ['auprc', 'AUPRC'], ['brier', 'Brier'], ['precision', 'Precision'],
  ['recall', 'Recall'], ['f1', 'F1'], ['specificity', 'Specificity']];

export default function FinalModelView({ kind = 'final' }) {
  const [status, setStatus] = useState({ data: null, error: null });
  const [index, setIndex] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api(`/api/${kind}/status`).then(data => setStatus({ data, error: null }))
      .catch(err => setStatus({ data: null, error: err.message }));
  }, [kind]);

  if (status.error) {
    return (
      <div className="page">
        <ErrorBox message={status.error} />
        <div className="card">
          <h3>{kind === 'sepsis' ? 'Mortality + sepsis model' : 'Final model'} not available</h3>
          <p className="muted">This tab shows only the model trained on the prepared eICU hospitals by
            <code>python scripts/final_model.py train{kind === 'sepsis' ? ' --task mortality_sepsis' : ''}</code>. It needs <code>processed/hospitals/hospital_&lt;id&gt;/</code> Parquet
            files (check with <code>python scripts/validate_data.py</code>). The synthetic demo in the other tabs is
            separate and is not used here.</p>
        </div>
      </div>
    );
  }
  if (!status.data) return <div className="page muted">Loading…</div>;

  const m = status.data.manifest;
  const primary = m.models[m.primary_model];
  const second = status.data.second_key, secondLabel = status.data.second_label;
  const label2 = secondLabel.charAt(0).toUpperCase() + secondLabel.slice(1);
  const target2 = (m.targets || ['hospital_expired', 'icu_expired'])[1];
  const run = async () => {
    setError(null);
    try { setResult(await api(`/api/${kind}/predict?test_index=${index}`)); } catch (err) { setResult(null); setError(err.message); }
  };

  return (
    <div className="page">
      <div className={`card ${status.data.synthetic ? 'band-synthetic' : 'band-historical'}`}>
        <h3>{m.title}</h3>
        <p><strong>Data source:</strong> {status.data.data_source}</p>
        {kind === 'sepsis' && (
          <p className="warn-text"><strong>Proxy label:</strong> sepsis = a sepsis or septic-shock diagnosis first documented after the first 24 h
            of the ICU stay (not a validated onset time). Stays with sepsis documented by 24 h are excluded. Academic prototype, not for clinical use.</p>)}
        <p className="muted">{m.task}. Hospital split {m.split.hospitals.train} / {m.split.hospitals.val} / {m.split.hospitals.test}{' '}
          (train / validation / test); checkpoint chosen by validation loss; test hospitals scored once, after training.</p>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="section-title">Checkpoint</div>
          <table className="table compact"><tbody>
            <tr><td>File</td><td className="break">{primary.checkpoint}</td></tr>
            <tr><td>SHA-256</td><td><code>{primary.sha256.slice(0, 16)}…</code></td></tr>
            <tr><td>Size</td><td>{(primary.size_bytes / 1024).toFixed(0)} KiB</td></tr>
            <tr><td>Selected round</td><td>{primary.training.selected_round} of {primary.training.rounds}
              {primary.training.completed ? ' (all rounds completed)' : ' (training incomplete)'}</td></tr>
            {primary.training.best_val_loss !== undefined && (
              <tr><td>Best validation loss</td><td>{fmt(primary.training.best_val_loss, 4)}</td></tr>)}
            <tr><td>Parameters</td><td>{primary.architecture.parameters.toLocaleString()}</td></tr>
            <tr><td>Seed</td><td>{m.config.seed}</td></tr>
          </tbody></table>
        </div>
        <div className="card">
          <div className="section-title">Architecture and federated training</div>
          <ul className="plain">
            <li>Tabular: {primary.architecture.tabular_encoder}</li>
            <li>Time-series: {primary.architecture.timeseries_input_dim} features × {primary.architecture.timeseries_bins} bins, {primary.architecture.timeseries_encoder}</li>
            <li>Fusion {primary.architecture.fusion}; graph: {primary.architecture.graph}</li>
            <li>GNN: {primary.architecture.gnn}; heads: {primary.architecture.heads.join(', ')}</li>
            <li>FedAvg: {m.config.rounds} rounds, {m.config.clients_per_round} clients per round, batch {m.config.batch_size}, lr {m.config.lr}</li>
            <li>Trained: {primary.training.optimizer_steps} optimizer steps, {primary.training.unique_patients_trained} unique patients
              from {primary.training.hospitals_sampled} sampled hospitals</li>
          </ul>
        </div>
      </div>

      <div className="card">
        <div className="section-title">Held-out test evaluation (newly measured on this checkpoint)</div>
        {['mort', second].map(task => (
          <table key={task} className="table">
            <thead><tr><th>{task === 'mort' ? 'In-hospital mortality' : label2}</th>
              {METRICS.map(([, l]) => <th key={l}>{l}</th>)}<th>Patients</th><th>Prevalence</th></tr></thead>
            <tbody>{Object.entries(m.models).map(([name, e]) => (
              <tr key={name}><td>{name}</td>{METRICS.map(([k]) => <td key={k}>{e.evaluation.metrics[task][k] === null ? 'undefined' : fmt(e.evaluation.metrics[task][k])}</td>)}
                <td>{e.evaluation.n_patients}</td><td>{pct(e.evaluation.metrics[task].prevalence)}</td></tr>
            ))}</tbody>
          </table>
        ))}
        {primary.evaluation.validation_chosen_threshold && (
          <table className="table">
            <thead><tr><th>Supplementary: threshold chosen on validation hospitals (best F1)</th><th>Threshold</th>
              <th>Precision</th><th>Recall</th><th>F1</th><th>Specificity</th></tr></thead>
            <tbody>{Object.entries(m.models).flatMap(([name, e]) => ['mort', second].map(task => {
              const v = e.evaluation.validation_chosen_threshold[task];
              return (<tr key={name + task}><td>{name} · {task === 'mort' ? 'in-hospital mortality' : secondLabel}</td><td>{fmt(v.threshold)}</td>
                {['precision', 'recall', 'f1', 'specificity'].map(k => <td key={k}>{fmt(v.test[k])}</td>)}</tr>);
            }))}</tbody>
          </table>
        )}
        <p className="muted">Threshold 0.5 for precision, recall, F1 and specificity. Historical results (results/corrected/) and synthetic demo
          results are shown in the Evaluation tab, not here.</p>
      </div>

      <div className="card controls">
        <label>Held-out test patient (0 – {status.data.test_patients - 1})
          <input type="number" min="0" max={status.data.test_patients - 1} value={index} onChange={e => setIndex(e.target.value)} />
        </label>
        <button onClick={run}>Run {kind === 'sepsis' ? 'mortality + sepsis' : 'final'} model</button>
        <span className="muted">Patients are shown by a masked index only; no eICU identifiers or raw values are displayed.</span>
      </div>
      {error && <ErrorBox message={error} />}
      {result && (
        <div className="grid-2">
          <div className="card">
            <h3>{result.patient} <span className="tag">{result.hospital} · test</span></h3>
            <ProbRow label="In-hospital mortality" p={result.p_in_hospital_mortality} />
            <ProbRow label={label2} p={result[`p_${result.second_key}`]} />
            <table className="table">
              <thead><tr><th>Task</th><th>Probability</th><th>Class at {result.threshold}</th><th>Recorded outcome</th></tr></thead>
              <tbody>
                <tr><td>In-hospital mortality</td><td>{pct(result.p_in_hospital_mortality)}</td><td>{result.pred_in_hospital_mortality ? 'positive' : 'negative'}</td>
                  <td>{result.recorded_outcome.hospital_expired ? 'died' : 'survived'}</td></tr>
                <tr><td>{label2}</td><td>{pct(result[`p_${result.second_key}`])}</td><td>{result[`pred_${result.second_key}`] ? 'positive' : 'negative'}</td>
                  <td>{result.recorded_outcome[target2] ? (kind === 'sepsis' ? 'sepsis documented' : 'died in ICU') : (kind === 'sepsis' ? 'no sepsis documented' : 'survived ICU')}</td></tr>
              </tbody>
            </table>
            <p className="warn-text">Research prototype. Not for clinical use.</p>
          </div>
          <div className="card">
            <div className="section-title">Pipeline for this patient</div>
            <p className="muted">Inputs: tabular {result.shapes.tabular_input.join(' × ')}, time-series {result.shapes.timeseries_input.join(' × ')}
              ({result.shapes.observed_bins} of 6 bins observed); {result.graph_neighbours} k-NN neighbours in its batch.</p>
            {[['MLP tabular embedding', 'tabular'], ['LSTM temporal embedding', 'timeseries'], ['Fused embedding', 'fused'], ['GraphSAGE embedding', 'graph']].map(([l, k]) => (
              <div key={k} className="flow-box"><div className="flow-title">{l}</div><div className="muted">{result.embeddings[k].length}-D</div><Strip values={result.embeddings[k]} /></div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
