# Public Streamlit Dashboard (No Bot Logic)

This folder is designed for a separate public repo and Streamlit Community Cloud deploy.
It only contains a read-only dashboard.

## What This Protects

- No execution logic
- No strategy code
- No API keys in code
- No private bot internals

## App Entry File

- `streamlit_app.py`

## Dependencies

- `requirements.txt`

## Data Source Modes

### 1) Recommended: Remote Database (live)

Set Streamlit secrets:

```toml
DATABASE_URL = "postgresql://readonly_user:password@host:5432/dbname"
MAX_TOTAL_RISK_DOLLARS = "60"
```

### 2) Local SQLite fallback (non-live in cloud)

Set Streamlit secrets:

```toml
SQLITE_PATH = "data/kalshi_quotes.sqlite"
MAX_TOTAL_RISK_DOLLARS = "60"
```

Note: On Streamlit Community Cloud, SQLite is only as fresh as files in the deployed repo.
For real-time updates, use `DATABASE_URL`.

## Deploy

1. Push this folder content to your new public repo root.
2. In Streamlit Community Cloud, deploy the repo with main file `streamlit_app.py`.
3. Add secrets in app settings.

## Resume-Safe Architecture

- Keep the trading bot and strategy in a private repo/server.
- Publish only aggregated data to a read-only DB for the dashboard.
- Public repo contains dashboard only.
