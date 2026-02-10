# 🎯 Kalshi Trading Bot - Dashboard Deployment

Your live trading dashboard is ready to share on your resume!

## Local Testing

Before deploying, test locally:

```powershell
# Install streamlit and plotly
pip install streamlit plotly pandas

# Run the dashboard
streamlit run src/dashboard.py
```

This opens a browser at `http://localhost:8501` showing:
- Portfolio P&L, risk, win rate, Sharpe ratio
- Equity curve and daily P&L charts
- Recent trades and series breakdown
- Forecast accuracy leaderboard
- Open positions

## Deploy to Streamlit Cloud (Free)

### Step 1: Push code to GitHub (optional but recommended)

If your repo is private, Streamlit Cloud can still access it—just authenticate once.

```powershell
# Initialize git (if not already done)
git init
git add .
git commit -m "Add trading dashboard"
git remote add origin https://github.com/YOUR_USERNAME/kalshi-bot.git
git push -u origin main
```

### Step 2: Create Streamlit Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign up" → authenticate with GitHub
3. Allow Streamlit to access your GitHub repos

### Step 3: Deploy

1. Click "New app"
2. Select:
   - **Repository**: `your-username/kalshi-bot`
   - **Branch**: `main`
   - **Main file path**: `src/dashboard.py`
3. Click "Deploy"

Streamlit Cloud will:
- Install dependencies from `requirments.txt`
- Run the dashboard
- Give you a public URL (e.g., `https://kalshi-bot-XXXXX.streamlit.app`)
4. **Share on your resume!**

### Step 4: Keep Database Updated

**Important:** The dashboard reads your local SQLite database. To keep it live:

**Option A (Auto-sync):**
- Store the database on a cloud service (AWS S3, Google Drive, etc.)
- Modify `dashboard.py` to fetch it before rendering

**Option B (Manual sync):**
- Periodically upload your `data/kalshi_quotes.sqlite` to a public cloud location
- Dashboard downloads it on load

**Option C (Database Read Replica):**
- Run a small script that syncs your local DB to a hosted PostgreSQL
- Dashboard connects to the hosted DB instead

---

## Alternative: Self-Hosted (Better for Live Data)

If you want truly live data syncing:

### Deploy to Railway, Render, or Heroku

1. Add a small Flask API on your machine that serves the latest trades/positions
2. Dashboard calls the API instead of reading local SQLite
3. Deploy Streamlit app to Railway/Render (free tier available)

This is more complex but keeps data fully live.

---

## Resume Copy-Paste

Add this to your resume:

```
Kalshi Weather Trading Bot Dashboard
• Built live Python trading bot using market APIs and weather forecasts
• Implemented real-time portfolio analytics: P&L tracking, Sharpe ratio, forecast accuracy
• Dashboard: [Live Dashboard URL] (Streamlit)
• Features: risk management, order amending, correlation hedging, volatility alerts
```

---

## Troubleshooting

**Database not found on deploy:**
- Streamlit Cloud doesn't have write access to your local files
- **Solution:** Use Option C above (sync to cloud DB)

**Dashboard shows "No trades yet":**
- The database might not be updated on the deployed version
- **Solution:** Manually load a copy of your database to a cloud storage the app can read

**Want more customization?**
- Add live P&L targets (e.g., "$10 goal today")
- Add position heatmaps by strike
- Add realized vs unrealized P&L breakdown
- Show bot logs / recent actions

---

**Test it now:** `streamlit run src/dashboard.py`
