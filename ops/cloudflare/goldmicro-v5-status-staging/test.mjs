import assert from 'node:assert/strict';
import worker from './src/index.js';

class MemoryKV {
  constructor() { this.map = new Map(); }
  async put(key, value) { this.map.set(key, value); }
  async get(key, type) {
    const value = this.map.get(key);
    if (value == null) return null;
    return type === 'json' ? JSON.parse(value) : value;
  }
}

const env = {
  STATUS_KV: new MemoryKV(),
  STATUS_INGEST_TOKEN: 'staging-test-token-abcdefghijklmnopqrstuvwxyz',
  STALE_AFTER_MINUTES: '360',
};

const payload = {
  source: 'goldmicro-v5-prospective',
  generated_at: new Date().toISOString(),
  host: 'test-host',
  git_head: 'deadbeef',
  state: 'ACCUMULATING_FRESH_EVIDENCE',
  fresh_rows: 86,
  fresh_setup_events: 2,
  matured_setup_events: 1,
  per_block_matured: [1, 0, 0, 0, 0],
  economic_outcomes: 'HIDDEN / NOT EVALUATED',
  promotion: 'DISABLED',
};

let res = await worker.fetch(new Request('https://status.example/health'), env);
assert.equal(res.status, 200);
let health = await res.json();
assert.equal(health.ok, false);
assert.equal(health.health, 'NO_DATA');
assert.equal(health.age_minutes, null);

res = await worker.fetch(new Request('https://status.example/ingest', { method: 'POST', body: JSON.stringify(payload) }), env);
assert.equal(res.status, 401);

res = await worker.fetch(new Request('https://status.example/ingest', {
  method: 'POST',
  headers: { authorization: `Bearer ${env.STATUS_INGEST_TOKEN}`, 'content-type': 'application/json' },
  body: JSON.stringify(payload),
}), env);
assert.equal(res.status, 202);

res = await worker.fetch(new Request('https://status.example/health'), env);
health = await res.json();
assert.equal(health.ok, true);
assert.equal(health.health, 'HEALTHY');
assert.equal(health.stale_after_minutes, 360);

res = await worker.fetch(new Request('https://status.example/api/status'), env);
assert.equal(res.status, 401);

res = await worker.fetch(new Request('https://status.example/api/status', {
  headers: { 'cf-access-authenticated-user-email': 'tester@example.com' },
}), env);
assert.equal(res.status, 200);
const status = await res.json();
assert.equal(status.health, 'HEALTHY');
assert.deepEqual(status.record.status.per_block_matured, [1, 0, 0, 0, 0]);
assert.equal(status.record.status.economic_outcomes, 'HIDDEN / NOT EVALUATED');

const bad = { ...payload, economic_outcomes: 'VISIBLE' };
res = await worker.fetch(new Request('https://status.example/ingest', {
  method: 'POST',
  headers: { authorization: `Bearer ${env.STATUS_INGEST_TOKEN}`, 'content-type': 'application/json' },
  body: JSON.stringify(bad),
}), env);
assert.equal(res.status, 400);

res = await worker.fetch(new Request('https://status.example/dashboard', {
  headers: { 'cf-access-authenticated-user-email': 'tester@example.com' },
}), env);
assert.equal(res.status, 200);
const html = await res.text();
assert.match(html, /ACCUMULATING_FRESH_EVIDENCE/);
assert.match(html, /\[1, 0, 0, 0, 0\]/);

const stalePayload = {
  ...payload,
  generated_at: new Date(Date.now() - 361 * 60_000).toISOString(),
};
res = await worker.fetch(new Request('https://status.example/ingest', {
  method: 'POST',
  headers: { authorization: `Bearer ${env.STATUS_INGEST_TOKEN}`, 'content-type': 'application/json' },
  body: JSON.stringify(stalePayload),
}), env);
assert.equal(res.status, 202);

res = await worker.fetch(new Request('https://status.example/health'), env);
health = await res.json();
assert.equal(health.ok, false);
assert.equal(health.health, 'STALE');
assert.ok(health.age_minutes >= 361);

console.log('GOLDmicro status worker tests passed');
