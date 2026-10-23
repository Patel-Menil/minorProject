import { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Users } from 'lucide-react';
import './index.css';

// Real model output is generated locally (git-ignored); the synthetic sample is the fallback.
const DATA_FILES = ['/graph_data.json', '/graph_data_synthetic.json'];

async function loadGraph() {
  for (const url of DATA_FILES) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      const data = await res.json(); // dev server answers missing files with HTML -> throws
      if (data.meta && Array.isArray(data.nodes)) return data;
    } catch {
      // try the next file
    }
  }
  throw new Error('No graph data found. Run scripts/generate_playground_data.py.');
}

// Green -> amber -> red for a probability in [0, 1]
function riskColor(p) {
  const stops = [[16, 185, 129], [245, 158, 11], [239, 68, 68]];
  const t = Math.min(Math.max(p, 0), 1) * 2;
  const [a, b] = t < 1 ? [stops[0], stops[1]] : [stops[1], stops[2]];
  const f = t < 1 ? t : t - 1;
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * f)).join(',')})`;
}

const OUTCOME_COLOR = { died: '#ef4444', survived: '#64748b' };
const pct = p => `${(p * 100).toFixed(1)}%`;
const idOf = end => (typeof end === 'object' ? end.id : end); // force-graph replaces ids with node objects

function SourceBanner({ meta }) {
  if (meta.source === 'model') {
    return (
      <div className="source-banner source-model">
        <strong>Model predictions</strong> on held-out test hospital {meta.hospital} ({meta.patients} patients),
        checkpoint from round {meta.checkpoint_round}. Edges: {meta.graph}.
      </div>
    );
  }
  return (
    <div className="source-banner source-synthetic">
      <strong>Synthetic demonstration data.</strong> Random numbers: not real patients and not model output.
    </div>
  );
}

function App() {
  const [graphData, setGraphData] = useState(null);
  const [error, setError] = useState(null);
  const [colorBy, setColorBy] = useState('prediction');
  const [selectedNode, setSelectedNode] = useState(null);
  const graphRef = useRef(null);
  const areaRef = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    loadGraph().then(setGraphData).catch(err => setError(err.message));
  }, []);

  // Size the canvas to the graph area (it is not the full window: the sidebar takes 380px)
  useEffect(() => {
    if (!areaRef.current) return;
    const observer = new ResizeObserver(([entry]) =>
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height }));
    observer.observe(areaRef.current);
    return () => observer.disconnect();
  }, [graphData]);

  const handleNodeClick = (node) => {
    setSelectedNode(node);
    if (graphRef.current) {
      graphRef.current.centerAt(node.x, node.y, 1000);
      graphRef.current.zoom(6, 2000);
    }
  };

  const nodeColor = node =>
    colorBy === 'prediction'
      ? riskColor(node.pred_mortality)
      : node.true_hospital_expired ? OUTCOME_COLOR.died : OUTCOME_COLOR.survived;

  if (error) return <div className="load-error">{error}</div>;
  if (!graphData) return <div className="load-error">Loading graph…</div>;

  const neighbours = selectedNode
    ? graphData.links.filter(l => idOf(l.source) === selectedNode.id || idOf(l.target) === selectedNode.id).length
    : 0;
  // Synthetic sample: both the "predictions" and the "outcomes" are random numbers
  const isModel = graphData.meta.source === 'model';

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>Federated GNN</h1>
          <p>Intra-hospital patient similarity graph</p>
          <SourceBanner meta={graphData.meta} />
        </div>

        <div className="node-details">
          {selectedNode ? (
            <div className="patient-card">
              <h2 style={{ margin: '0 0 4px 0', fontSize: '1.25rem' }}>Patient {selectedNode.id}</h2>
              <p className="muted">Anonymous label, not an eICU identifier.</p>

              <div className="section-title">
                {isModel ? 'Model prediction' : 'Prediction (synthetic)'}
              </div>
              <ProbRow label="In-hospital mortality" p={selectedNode.pred_mortality} />
              <ProbRow label="ICU mortality" p={selectedNode.pred_icu_mortality} />

              <div className="section-title">{isModel ? 'Recorded outcome (ground truth)' : 'Outcome (synthetic)'}</div>
              <OutcomeRow label="In hospital" died={selectedNode.true_hospital_expired} />
              <OutcomeRow label="In ICU" died={selectedNode.true_icu_expired} />
              <p className="muted">
                {isModel
                  ? 'The model does not see this outcome when it makes the prediction above.'
                  : 'Random demonstration values: not a real patient and not a real outcome.'}
              </p>

              <div className="stats-grid">
                <div className="stat-box">
                  <div className="value">{neighbours}</div>
                  <div className="label">Linked Patients</div>
                </div>
                <div className="stat-box">
                  <div className="value">128D</div>
                  <div className="label">Fused Emb</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <Users size={48} />
              <h3 style={{ margin: '0 0 8px 0', fontWeight: 500, color: '#f8fafc' }}>Select a Patient</h3>
              <p style={{ margin: 0, fontSize: '0.875rem' }}>
                {isModel
                  ? "Click a node to compare the model's prediction with the recorded outcome."
                  : 'Click a node to see its synthetic demonstration values.'}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Graph Area */}
      <div className="graph-area" ref={areaRef}>
        <div className="graph-overlay">
          <div className="toggle" role="group" aria-label="Colour nodes by">
            <button aria-pressed={colorBy === 'prediction'} onClick={() => setColorBy('prediction')}>
              Predicted risk
            </button>
            <button aria-pressed={colorBy === 'outcome'} onClick={() => setColorBy('outcome')}>
              {isModel ? 'Recorded outcome' : 'Outcome (synthetic)'}
            </button>
          </div>
          {colorBy === 'prediction' ? (
            <div className="legend-item">
              <span>0%</span>
              <div className="gradient" />
              <span>100% predicted in-hospital mortality</span>
            </div>
          ) : (
            <>
              <div className="legend-item"><div className="dot" style={{ background: OUTCOME_COLOR.survived }} />Survived</div>
              <div className="legend-item"><div className="dot" style={{ background: OUTCOME_COLOR.died }} />Died in hospital</div>
            </>
          )}
        </div>

        <ForceGraph2D
          ref={graphRef}
          width={size.width}
          height={size.height}
          graphData={graphData}
          nodeLabel={n => `${n.id}: predicted ${pct(n.pred_mortality)}`}
          nodeColor={nodeColor}
          nodeRelSize={6}
          linkColor={() => 'rgba(255, 255, 255, 0.15)'}
          linkWidth={link => Math.max(0.5, (link.weight || 0) * 2)}
          onNodeClick={handleNodeClick}
          backgroundColor="#0f172a"
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          nodeCanvasObjectMode={() => 'after'}
          nodeCanvasObject={(node, ctx, globalScale) => {
            // Subtle ring for the selected node
            if (selectedNode && node.id === selectedNode.id) {
              ctx.beginPath();
              ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI, false);
              ctx.lineWidth = 2 / globalScale;
              ctx.strokeStyle = '#ffffff';
              ctx.stroke();
            }
          }}
        />
      </div>
    </div>
  );
}

function ProbRow({ label, p }) {
  return (
    <div className="prob-row">
      <div className="prob-label"><span>{label}</span><span>{pct(p)}</span></div>
      <div className="prob-track"><div className="prob-fill" style={{ width: pct(p), background: riskColor(p) }} /></div>
    </div>
  );
}

function OutcomeRow({ label, died }) {
  return (
    <div className="prob-label">
      <span>{label}</span>
      <span className={`status-badge ${died ? 'status-expired' : 'status-survived'}`}>{died ? 'Died' : 'Survived'}</span>
    </div>
  );
}

export default App;
