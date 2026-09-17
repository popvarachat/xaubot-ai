const REQUIRED_KEYS = [
  'source', 'generated_at', 'host', 'git_head', 'state',
  'fresh_rows', 'fresh_setup_events', 'matured_setup_events',
  'per_block_matured', 'economic_outcomes', 'promotion',
];

function json(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store',
      ...extraHeaders,
    },
  });
}

function bearerToken(request) {
  const auth = request.headers.get('authorization') || '';
  return auth.startsWith('Bearer ') ? auth.slice(7) : '';
}

function isBlindStatus(body) {
  if (!body || typeof body !== 'object') return false;
  for (const key of REQUIRED_KEYS) if (!(key in body)) return false;
  if (body.source !== 'goldmicro-v5-prospective') return false;
  if (body.economic_outcomes !== 'HIDDEN / NOT EVALUATED') return false;
  if (body.promotion !== 'DISABLED') return false;
  if (!Array.isArray(body.per_block_matured) || body.per_block_matured.length !== 5) return false;
  for (const key of ['fresh_rows', 'fresh_setup_events', 'matured_setup_events']) {
    if (!Number.isInteger(body[key]) || body[key] < 0) return false;
  }
  for (const count of body.per_block_matured) {
    if (!Number.isInteger(count) || count < 0) return false;
  }
  return true;
}

function viewAuthorized(request) {
  return Boolean(request.headers.get('cf-access-authenticated-user-email'));
}

function staleMinutes(record, nowMs) {
  const generated = Date.parse(record?.status?.generated_at || '');
  if (!Number.isFinite(generated)) return null;
  return Math.max(0, Math.floor((nowMs - generated) / 60000));
}

function renderDashboard(record, staleAfterMinutes, nowMs) {
  const status = record?.status || null;
  const age = status ? staleMinutes(record, nowMs) : null;
  const health = !status ? 'NO_DATA' : (age !== null && age > staleAfterMinutes ? 'STALE' : 'HEALTHY');
  const blocks = status?.per_block_matured || [0, 0, 0, 0, 0];
  const esc = (v) => String(v ?? '-').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  return `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GOLDmicro V5 Status</title><style>
body{font-family:system-ui;margin:0;background:#0b1020;color:#eef2ff}.wrap{max-width:980px;margin:40px auto;padding:0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}.card{background:#151c33;border:1px solid #293452;border-radius:14px;padding:18px}
.k{font-size:12px;color:#9fb0d8;text-transform:uppercase}.v{font-size:24px;font-weight:700;margin-top:8px}.ok{color:#4ade80}.warn{color:#fbbf24}.muted{color:#9fb0d8;font-size:13px}
</style></head><body><div class="wrap"><h1>GOLDmicro V5 Prospective</h1>
<p class="muted">Blind status only. Economic outcomes and promotion remain disabled.</p>
<div class="grid">
<div class="card"><div class="k">Health</div><div class="v ${health === 'HEALTHY' ? 'ok' : 'warn'}">${esc(health)}</div></div>
<div class="card"><div class="k">State</div><div class="v">${esc(status?.state)}</div></div>
<div class="card"><div class="k">Fresh rows</div><div class="v">${esc(status?.fresh_rows)}</div></div>
<div class="card"><div class="k">Matured</div><div class="v">${esc(status?.matured_setup_events)}</div></div>
<div class="card"><div class="k">Blocks</div><div class="v">[${blocks.map(esc).join(', ')}]</div></div>
<div class="card"><div class="k">Age</div><div class="v">${age === null ? '-' : `${age} min`}</div></div>
</div><p class="muted">Last accepted: ${esc(record?.accepted_at)} | git ${esc(status?.git_head)}</p></div></body></html>`;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const staleAfterMinutes = Number.parseInt(env.STALE_AFTER_MINUTES || '360', 10);

    if (url.pathname === '/health') {
      return json({ ok: true, service: 'goldmicro-v5-status-staging', economics: 'HIDDEN', promotion: 'DISABLED' });
    }

    if (url.pathname === '/ingest' && request.method === 'POST') {
      if (!env.STATUS_INGEST_TOKEN || bearerToken(request) !== env.STATUS_INGEST_TOKEN) {
        return json({ accepted: false, error: 'unauthorized' }, 401);
      }
      let body;
      try { body = await request.json(); } catch { return json({ accepted: false, error: 'invalid_json' }, 400); }
      if (!isBlindStatus(body)) return json({ accepted: false, error: 'invalid_blind_status_contract' }, 400);
      const record = { accepted_at: new Date().toISOString(), status: body };
      await env.STATUS_KV.put('latest', JSON.stringify(record));
      return json({ accepted: true, accepted_at: record.accepted_at }, 202);
    }

    if (url.pathname === '/api/status' && request.method === 'GET') {
      if (!viewAuthorized(request)) return json({ error: 'cloudflare_access_required' }, 401);
      const record = await env.STATUS_KV.get('latest', 'json');
      const age = record ? staleMinutes(record, Date.now()) : null;
      const health = !record ? 'NO_DATA' : (age !== null && age > staleAfterMinutes ? 'STALE' : 'HEALTHY');
      return json({ health, age_minutes: age, stale_after_minutes: staleAfterMinutes, record });
    }

    if (url.pathname === '/dashboard' && request.method === 'GET') {
      if (!viewAuthorized(request)) return new Response('Cloudflare Access required', { status: 401 });
      const record = await env.STATUS_KV.get('latest', 'json');
      return new Response(renderDashboard(record, staleAfterMinutes, Date.now()), {
        headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' },
      });
    }

    return json({ error: 'not_found' }, 404);
  },
};
