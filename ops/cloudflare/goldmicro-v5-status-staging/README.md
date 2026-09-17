# GOLDmicro V5 Status Worker — STAGING

Purpose: expose a read-only blind status surface for the prospective GOLDmicro V5 evidence collector without revealing economic outcomes or enabling promotion/trading.

## Endpoints

- `POST /ingest` — accepts the validated blind status contract only; requires `Authorization: Bearer <STATUS_INGEST_TOKEN>`.
- `GET /health` — public non-sensitive health only; returns `HEALTHY`, `STALE`, `NO_DATA`, or `INVALID_TIMESTAMP`, plus age/stale threshold. It never exposes setup counts, state, economic outcomes, or promotion data.
- `GET /api/status` — latest blind status; requires Cloudflare Access identity.
- `GET /dashboard` — minimal status dashboard; requires Cloudflare Access identity.

The worker rejects payloads unless `economic_outcomes == "HIDDEN / NOT EVALUATED"` and `promotion == "DISABLED"`.

## Persistence

A dedicated STAGING KV namespace is bound as `STATUS_KV`. Only the latest blind status record is stored under key `latest`.

## Stale policy

Default `STALE_AFTER_MINUTES=360`. With a four-hour local collection cadence, this gives a two-hour grace window before status becomes `STALE`. The public `/health` endpoint is intentionally limited to non-sensitive liveness/staleness metadata so an external monitor can alert without Cloudflare Access credentials.

## Security / Human Gate

The STAGING deployment uses a dedicated KV namespace, a Wrangler-managed `STATUS_INGEST_TOKEN`, and Cloudflare Access scoped only to `/dashboard` and `/api/status`. `/ingest` remains protected by its bearer secret and `/health` remains public but non-sensitive for monitoring.

Do not commit tokens, Access credentials, generated Wrangler state, or the local `wrangler.toml`. Do not route production trading traffic through this worker. Economic outcomes remain hidden and promotion/trading remain disabled.
