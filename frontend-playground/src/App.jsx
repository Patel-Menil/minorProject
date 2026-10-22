import React, { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Activity, Users, ActivitySquare, HeartPulse } from 'lucide-react';
import './index.css';

function App() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const graphRef = useRef(null);

  useEffect(() => {
    fetch('/graph_data.json')
      .then(res => res.json())
      .then(data => {
        setGraphData(data);
      })
      .catch(err => console.error("Error loading graph data", err));
  }, []);

  const handleNodeClick = (node) => {
    setSelectedNode(node);
    
    // Pan to node and zoom in slightly
    if (graphRef.current) {
      graphRef.current.centerAt(node.x, node.y, 1000);
      graphRef.current.zoom(8, 2000);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>Federated GNN</h1>
          <p>eICU Hospital 102 (Intra-hospital Patient Graph)</p>
        </div>
        
        <div className="node-details">
          {selectedNode ? (
            <div className="patient-card">
              <div className={`status-badge ${selectedNode.expired ? 'status-expired' : 'status-survived'}`}>
                {selectedNode.expired ? 'Mortality Risk (High)' : 'Stable'}
              </div>
              
              <h2 style={{ margin: '0 0 16px 0', fontSize: '1.25rem' }}>{selectedNode.name}</h2>

              <div className="stats-grid">
                <div className="stat-box">
                  <div className="value">{graphData.links.filter(l => l.source.id === selectedNode.id || l.target.id === selectedNode.id).length}</div>
                  <div className="label">Similar Patients</div>
                </div>
                <div className="stat-box">
                  <div className="value">128D</div>
                  <div className="label">Fused Emb</div>
                </div>
              </div>
              
              <div style={{ marginTop: '24px' }}>
                <div className="section-title">Modality Encoders Active</div>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '12px', padding: '4px 8px', background: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa', borderRadius: '4px', border: '1px solid rgba(59,130,246,0.3)' }}>Tabular MLP</span>
                  <span style={{ fontSize: '12px', padding: '4px 8px', background: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa', borderRadius: '4px', border: '1px solid rgba(59,130,246,0.3)' }}>Time-Series LSTM</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <Users size={48} />
              <h3 style={{ margin: '0 0 8px 0', fontWeight: 500, color: '#f8fafc' }}>Select a Patient</h3>
              <p style={{ margin: 0, fontSize: '0.875rem' }}>Click on a node in the graph to view fused multi-modal data.</p>
            </div>
          )}
        </div>
      </div>

      {/* Graph Area */}
      <div className="graph-area">
        <div className="graph-overlay">
          <div className="legend-item">
            <div className="dot" style={{ background: '#10b981' }}></div>
            Stable
          </div>
          <div className="legend-item">
            <div className="dot" style={{ background: '#ef4444' }}></div>
            High Risk
          </div>
        </div>
        
        {graphData.nodes.length > 0 && (
          <ForceGraph2D
            ref={graphRef}
            graphData={graphData}
            nodeLabel="name"
            nodeColor="color"
            nodeRelSize={6}
            linkColor={() => 'rgba(255, 255, 255, 0.15)'}
            linkWidth={link => Math.max(0.5, (link.weight || 0) * 2)}
            onNodeClick={handleNodeClick}
            backgroundColor="#0f172a"
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
            nodeCanvasObjectMode={() => 'after'}
            nodeCanvasObject={(node, ctx, globalScale) => {
              // Add a subtle glowing ring for the selected node
              if (selectedNode && node.id === selectedNode.id) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI, false);
                ctx.lineWidth = 2 / globalScale;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();
              }
            }}
          />
        )}
      </div>
    </div>
  );
}

export default App;
