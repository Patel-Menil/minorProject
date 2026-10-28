// Calls the local demo API (python scripts/run_demo.py serve). Errors are shown to the
// user as they are; there is deliberately no fallback to made-up predictions.
export async function api(path) {
  let res;
  try {
    res = await fetch(path);
  } catch {
    throw new Error('Demo API not reachable. Start it with: python scripts/run_demo.py serve');
  }
  let body;
  try {
    body = await res.json();
  } catch {
    throw new Error(`Demo API not reachable at ${path} (no JSON answer). Start it with: python scripts/run_demo.py serve`);
  }
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

export const pct = p => `${(p * 100).toFixed(1)}%`;
export const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(d));

// Green -> amber -> red for a probability in [0, 1]
export function riskColor(p) {
  const stops = [[16, 185, 129], [245, 158, 11], [239, 68, 68]];
  const t = Math.min(Math.max(p, 0), 1) * 2;
  const [a, b] = t < 1 ? [stops[0], stops[1]] : [stops[1], stops[2]];
  const f = t < 1 ? t : t - 1;
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * f)).join(',')})`;
}
