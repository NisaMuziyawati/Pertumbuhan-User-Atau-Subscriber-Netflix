# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# ---------------------------
# Page config & style
# ---------------------------
st.set_page_config(page_title="Netflix Subscriber Growth (Logistic)",
                   page_icon="🎬", layout="wide")

st.markdown("""
<style>
/* dark background + cards */
body { background-color:#0b0b0b; color: #ddd; }
h1,h2,h3,p,label { color:#fff !important; }
.card {
  padding: 14px;
  border-radius: 10px;
  background: linear-gradient(180deg,#111,#0f0f0f);
  border: 1px solid rgba(255,255,255,0.04);
}
.small { color: #999; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.title("🎬 Netflix Subscriber Growth — Logistic Forecast (sampai 2030)")
st.markdown("Dashboard: historis ➜ model logistic ➜ prediksi sampai **2030** (K = **700 juta**)")

# ---------------------------
# Upload dataset
# ---------------------------
st.sidebar.header("Upload data")
uploaded = st.sidebar.file_uploader("Upload file users.csv (kolom subscription_start_date, is_active)", type=["csv"])

use_sample = False
if uploaded is None:
    st.sidebar.info("No file uploaded — menggunakan sample data contoh.")
    use_sample = True

# ---------------------------
# Load data (uploaded or sample)
# ---------------------------
if not use_sample:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Gagal membaca CSV: " + str(e))
        st.stop()
else:
    # sample yearly active subscribers (counts) in millions for demo (matches typical Netflix-scale)
    sample = {
        "Year": [2013,2014,2015,2016,2017,2018,2019,2020,2021,2022],
        "Subscribers_count": [33_000_000,48_000_000,70_000_000,89_000_000,110_000_000,139_000_000,167_000_000,204_000_000,221_000_000,230_000_000]
    }
    df = pd.DataFrame(sample)

# ---------------------------
# Preprocess: if raw users.csv -> aggregate to yearly active subscribers
# ---------------------------
if not use_sample:
    # ensure date parse
    df["subscription_start_date"] = pd.to_datetime(df["subscription_start_date"], errors="coerce")
    # filter active subscribers
    if "is_active" in df.columns:
        df_active = df[df["is_active"] == True].copy()
    else:
        df_active = df.copy()
    # if subscription_start_date exists -> group by year
    if "subscription_start_date" in df_active.columns and df_active["subscription_start_date"].notna().any():
        df_active["Year"] = df_active["subscription_start_date"].dt.year
        agg = df_active.groupby("Year").size().reset_index(name="Subscribers_count")
    else:
        st.error("Dataset tidak berisi kolom 'subscription_start_date' yang valid. Pastikan ada kolom tersebut.")
        st.stop()
else:
    agg = df.rename(columns={"Subscribers_count": "Subscribers_count"})

agg = agg.sort_values("Year").reset_index(drop=True)

# convert counts -> millions for plotting & modeling convenience
agg["Subscribers_million"] = agg["Subscribers_count"] / 1_000_000.0

# ---------------------------
# Show data & key metrics
# ---------------------------
st.subheader("Data historis (jumlah subscriber per tahun)")
col1, col2 = st.columns([2,1])
with col1:
    st.dataframe(agg[["Year","Subscribers_count","Subscribers_million"]], use_container_width=True)
with col2:
    total_active = int(agg["Subscribers_count"].max())
    avg_monthly_spend = None
    # attempt to show some quick cards (if monthly_spend exists in original df)
    try:
        if not use_sample and "monthly_spend" in df.columns:
            avg_monthly_spend = df["monthly_spend"].mean()
    except:
        avg_monthly_spend = None

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Latest year", agg["Year"].max())
    st.metric("Subscribers (max year)", f"{agg['Subscribers_million'].iloc[-1]:.2f} M")
    if avg_monthly_spend:
        st.write(f"<div class='small'>Avg monthly spend: ${avg_monthly_spend:.2f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

# ---------------------------
# Logistic model setup (user controls)
# ---------------------------
st.subheader("Model: Logistic Growth (fit parameter dari data)")
K_million = st.number_input("Kapasitas maksimum (juta user)", min_value=100.0, max_value=2000.0, value=700.0, step=10.0)
forecast_end = st.number_input("Prediksi sampai tahun", min_value=int(agg["Year"].max()+1), max_value=2040, value=2030, step=1)

# time arrays
years_hist = agg["Year"].values.astype(float)
y_hist = agg["Subscribers_million"].values.astype(float)

# logistic function
def logistic(t, r, t0):
    K = float(K_million)
    return K / (1.0 + np.exp(-r * (t - t0)))

# fit r and t0 while K fixed
fit_success = False
try:
    p0 = [0.2, np.mean(years_hist)]
    popt, pcov = curve_fit(logistic, years_hist, y_hist, p0=p0, maxfev=10000)
    r_fit, t0_fit = popt[0], popt[1]
    fit_success = True
except Exception:
    # fallback heuristic: approximate r by linearizing small sample
    r_fit = 0.2
    t0_fit = np.mean(years_hist)

# predictions
years_pred = np.arange(int(years_hist.min()), int(forecast_end) + 1)
pred_mean = logistic(years_pred, r_fit, t0_fit)

# build uncertainty envelope (simple ±10%)
upper = pred_mean * 1.10
lower = pred_mean * 0.90

# interpolation of pred to compute MSE against historical years
pred_at_hist = logistic(years_hist, r_fit, t0_fit)
mse = mean_squared_error(y_hist, pred_at_hist)

# Year reach 90% K
try:
    target = 0.9 * K_million
    # invert logistic: t = t0 - (1/r) * ln(K / target - 1)
    t_90 = t0_fit - (1.0 / r_fit) * np.log(K_million / target - 1.0)
    t_90 = float(np.round(t_90, 2))
except Exception:
    t_90 = None

# ---------------------------
# Plotly chart: line + points + shading
# ---------------------------
st.subheader("Grafik: Historis + Prediksi (Logistic)")

fig = go.Figure()

# shaded envelope (prediction uncertainty)
fig.add_traces([
    go.Scatter(x=years_pred, y=upper, mode='lines', line=dict(width=0), showlegend=False),
    go.Scatter(x=years_pred, y=lower, mode='lines', line=dict(width=0), fill='tonexty',
               fillcolor='rgba(255,0,0,0.12)', name='Uncertainty ±10%')
])

# predicted mean line
fig.add_trace(go.Scatter(x=years_pred, y=pred_mean, mode='lines', name='Prediksi (Logistic)', line=dict(color='crimson', width=3)))

# historical points
fig.add_trace(go.Scatter(x=years_hist, y=y_hist, mode='markers+lines', name='Data Historis', marker=dict(size=8, color='white')))

# layout
fig.update_layout(
    plot_bgcolor='#0b0b0b',
    paper_bgcolor='#0b0b0b',
    font=dict(color='#ddd'),
    title=f'Logistic Fit — K={K_million:.0f}M, r={r_fit:.3f}, t0={t0_fit:.2f} (MSE={mse:.4f})',
    xaxis_title='Tahun',
    yaxis_title='Subscribers (juta)',
    legend=dict(bgcolor='rgba(0,0,0,0.2)')
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Model Summary & Insights
# ---------------------------
st.subheader("Ringkasan Model & Insight")
st.markdown(f"- Estimasi parameter fit: **r = {r_fit:.4f}**, **t0 = {t0_fit:.2f}**")
st.markdown(f"- Mean Squared Error (hist vs model): **{mse:.4f} (juta^2)**")
st.markdown(f"- Kapasitas pasar (K): **{K_million:.0f} juta user**")

if t_90 is not None and np.isfinite(t_90):
    st.markdown(f"- Tahun perkiraan mencapai **90% dari K ({0.9*K_million:.0f} juta)**: sekitar **{t_90}**")
else:
    st.markdown("- Tahun pencapaian 90% K: tidak dapat diperkirakan dengan data saat ini")

st.markdown("---")
st.markdown("<div class='small'>Note: Envelope shading = simple ±10% untuk memberi gambaran ketidakpastian. Untuk interval kepercayaan formal perlu fitting statistik yang lebih dalam.</div>", unsafe_allow_html=True)

# ---------------------------
# Footer / export
# ---------------------------
st.write("")
st.markdown("<div style='text-align:center; color:gray;'>© 2025 Netflix Growth Dashboard — Prepared for TA-09</div>", unsafe_allow_html=True)
