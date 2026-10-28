# Frontend demo (React + Vite)

Web UI for the synthetic viva demo and the patient-graph viewer. See the main [README](../README.md).

```powershell
npm ci            # once
npm run build     # -> dist/, served by: python scripts/run_demo.py serve  (http://127.0.0.1:8765)
npm run dev       # hot reload on http://localhost:5173; /api is proxied to the demo server on port 8765
npm run lint
```

- `src/App.jsx`: tabs and the permanent "synthetic data · not for clinical use" banner.
- `src/DemoViews.jsx`: Overview, Predict, Federation and Evaluation. All data comes from the local demo API, and errors are shown as they are.
- `src/GraphView.jsx`: patient-graph viewer. It loads `/api/graph` (demo model on synthetic data), then `public/graph_data.json` (real-model output, git-ignored), then `public/graph_data_synthetic.json` (random sample, labelled as such).
- `src/api.js`: fetch helper with no fallback values.
