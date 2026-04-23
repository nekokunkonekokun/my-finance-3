import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as mticker
import streamlit as st

# --- 1. Page Config ---
st.set_page_config(page_title="Sniper Mission Control", layout="wide")

# --- 2. Configuration ---
ticker_sym = "NIY=F"
interval = "30m"
period = "1mo"
ma_window = 25
std_window = 160

# --- 3. Data Acquisition ---
@st.cache_data(ttl=600)
def load_data():
    raw = yf.download(ticker_sym, period=period, interval=interval, auto_adjust=True)
    if raw.empty: return pd.DataFrame()
    
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 全カラム名を強制正規化
    df.columns = [str(c).lower().capitalize() for c in df.columns]
    df = df.dropna(subset=['Close']).reset_index()
    
    # 【最重要】名前が何であれ、0番目の列（日付）を "TS" に固定
    df.rename(columns={df.columns[0]: "TS"}, inplace=True)
    return df

df = load_data()

if df.empty or "Close" not in df.columns:
    st.error("Market Data fetch failed.")
    st.stop()

# --- 4. Calculation ---
df['MA25'] = df['Close'].rolling(window=ma_window).mean()
df['Bias'] = (df['Close'] - df['MA25']) / df['MA25'] * 100
df['B_Mean'] = df['Bias'].rolling(window=std_window).mean()
df['B_Std'] = df['Bias'].rolling(window=std_window).std()
df['Z'] = (df['Bias'] - df['B_Mean']) / df['B_Std']
df['T_Score'] = df['Z'] * 10 + 50
df['Prob'] = df['Z'].apply(lambda x: norm.cdf(x) * 100 if pd.notnull(x) else np.nan)
df['Accel'] = df['T_Score'].diff()

latest = df.iloc[-1]

# --- 5. Report Logic ---
def get_report(price, t, acc, prob):
    p_rep = f" [PRICE]   ¥{price:,.0f} (Active Hours)"
    d_rep = f" [REPORT]  Deviation: {t:.1f} | Prob: {prob:.1f}%"
    a_rep = f" [CONTACT] Accel: {acc:.2f} ({'UP ↑' if acc > 0 else 'DOWN ↓'})"
    if t >= 85 and acc < 0: c = "OVER 85! SNIPER SHORT READY!"
    elif acc >= 5.0: c = "★ MAC POWER! GO GO! ★"
    elif t >= 75 and acc < 0: c = "LEVEL 75. Fading. Short?"
    elif t > 70: c = "Overheated. Prepare Sniper."
    else: c = "Sideways Truth. Wait."
    return f"{p_rep}\n{d_rep}\n{a_rep}\n [CONSULT] {c}"

# --- 6. UI ---
st.title("Sniper Mission Control")
st.code(get_report(latest['Close'], latest['T_Score'], latest['Accel'], latest['Prob']))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2.2, 1, 1]})

# 土日を詰めるためIndex（連番）を使用
idx = df.index
ax1.plot(idx, df['Close'], color='black', linewidth=1); ax1.plot(idx, df['MA25'], color='orange', linewidth=2)
ax1.grid(True, alpha=0.3); ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

ax2.plot(idx, df['T_Score'], color='blue'); ax2.axhline(75, color='red'); ax2.axhline(50, color='gray', alpha=0.5)
ax2.fill_between(idx, 75, 100, color='red', alpha=0.1); ax2.grid(True, alpha=0.3)

ax3.bar(idx, df['Accel'], color=['red' if x > 0 else 'blue' for x in df['Accel']], alpha=0.7)
ax3.axhline(0, color='black', linewidth=1); ax3.grid(True, alpha=0.3)

# 【ここが修正点】名前ではなく "TS" カラムを確実に参照
t_idx = np.linspace(0, len(df)-1, 10, dtype=int)
ax3.set_xticks(t_idx)
ax3.set_xticklabels([pd.to_datetime(df["TS"].iloc[i]).strftime('%m/%d %H:%M') for i in t_idx], rotation=45)

st.pyplot(fig)
