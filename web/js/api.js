// Thin wrapper over the two backend endpoints.

export async function fetchProblems() {
  const res = await fetch('/api/problems');
  if (!res.ok) throw new Error(`GET /api/problems -> ${res.status}`);
  const data = await res.json();
  return data.problems;
}

export async function solve(slug, values) {
  const res = await fetch('/api/solve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug, values }),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try { detail = (await res.json()).detail ?? detail; } catch { /* keep status */ }
    throw new Error(`solve failed: ${detail}`);
  }
  return res.json();
}
