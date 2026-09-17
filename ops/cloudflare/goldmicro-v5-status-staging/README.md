# GOLDmicro V5 Status Worker — STAGING

Purpose: expose a read-only blind status surface for the prospective GOLDmicro V5 evidence collector without revealing economic outcomes or enabling promotion/trading.

## Endpoints

- `POST /ingest` — accepts the validated blind status contract only; requires `Authorization: Bearer <STATUS_INGEST_TOKEN>`.
- `GET /dashboardhealth` — generic health only; never exposes run data.
- `GET /dashboardapi/status` — latest blind status; requires Cloudflare Access identity header.
- `GET /dashboard` — minimal status dashboard; requires Cloudflare Access identity header.

The worker rejects payloads unless `economic_outcomes == "HIDDEN / NOT EVALUATED"` and `promotion == "DISABLED"`.

## Persistence

A dedicated STAGING KV namespace is bound as `STATUS_KV`. Only the latest blind status record is stored under key `latest`.

## Stale policy

Default `STALE_AFTER_MINUTES=360`. With a four-hour local collection cadence, this gives a two-hour grace window before status becomes `STALE`.

## Security / Human Gate

This directory is code/config preparation only. Deployment requires a separate Human Gate because it creates Cloudflare resources and a secret. At deploy time:

1. create a dedicated STAGING KV namespace;
2. copy `wrangler.toml.example` to a local uncommitted `wrangler.toml` and insert the KV namespace id;
3. set `STATUS_INGEST_TOKEN` through Wrangler secret storage;
4. configure Cloudflare Access for dashboard/status viewing;
5. deploy only to STAGING;
6. update the n8n STAGING flow to forward the already-validated blind status to `/ingest`.

Do not commit tokens, Access credentials, or generated Wrangler state. Do not route production trading traffic through this worker.
