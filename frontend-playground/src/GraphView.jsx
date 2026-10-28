import { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Users } from 'lucide-react';
import { pct, riskColor } from './api.js';

// 1. demo model on synthetic data (API), 2. real model output generated locally (git-ignored),
// 3. the committed random sample. The banner always says which one is shown.
const DATA_FILES = ['/api/graph', '/graph_data.json', '/graph_data_synthetic.json'];

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
  throw new Error('No graph data found. Run: python scripts/run_demo.py all, then python scripts/run_demo.py serve');
}

const OUTCOME_COLOR = { died: '#ef4444', survived: '#64748b' };
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
  if (meta.source === 'demo_model') {
    return (
      <div className="source-banner source-synthetic">
        <strong>Demo model on synthetic data.</strong> Predictions of the demo checkpoint (round {meta.checkpoint_round}) for
        the {meta.patients} synthetic patients of held-out synthetic hospital {meta.hospital}. Edges: {meta.graph}.
        Not real patients; not clinically meaningful.
      </div>
    );
  }
  return (
    <div className="source-banner source-synthetic">
      <strong>Synthetic demonstration data.</strong> Random numbers: not real patients and not model output.
    </div>
  );
}

export default function GraphView() {
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
  // "model" and "demo_model" carry genuine model output; the random sample does not
  const isModel = graphData.meta.source === 'model' || graphData.meta.source === 'demo_model';
  const isSynthetic = graphData.meta.source !== 'model';

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>Patient graph</h1>
          <p>Intra-hospital patient similarity graph</p>
          <SourceBanner meta={graphData.meta} />
        </div>

        <div className="node-details">
          {selectedNode ? (
            <div className="patient-card">
              <h2 style={{ margin: '0 0 4px 0', fontSize: '1.25rem' }}>Patient {selectedNode.id}</h2>
              <p className="muted">{isSynthetic ? 'Synthetic patient label.' : 'Anonymous label, not an eICU identifier.'}</p>

              <div className="section-title">
                {isModel ? 'Model prediction' : 'Prediction (synthetic)'}
              </div>
              <ProbRow label="In-hospital mortality" p={selectedNode.pred_mortality} />
              <ProbRow label="ICU mortality" p={selectedNode.pred_icu_mortality} />

              <div className="section-title">
                {isSynthetic ? 'Outcome (synthetic label)' : 'Recorded outcome (ground truth)'}
              </div>
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
                  ? "Click a node to compare the model's prediction with the outcome label."
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
              {isSynthetic ? 'Outcome (synthetic)' : 'Recorded outcome'}
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

export function ProbRow({ label, p }) {
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
