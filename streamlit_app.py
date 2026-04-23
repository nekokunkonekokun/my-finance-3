import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as mticker
import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="Sniper Mission Control", layout="wide")

# --- Configuration ---
ticker_sym = "NIY=F"
interval = "30m"
period = "1mo"
ma_window = 25
std_window = 160

# --- Data Acquisition ---
@st.cache_data(ttl=600) # 10分間キャッシュして効率化
def load_data():
    data = yf.download(ticker_sym, period=period, interval=interval, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.copy().dropna(subset=['Close']).reset_index()

df = load_data()
latest = df.iloc[-1]

# --- Calculation ---
df['MA25'] = df['Close'].rolling(window=ma_window).mean()
df['Bias'] = (df['Close'] - df['MA25']) / df['MA25'] * 100
df['Bias_Mean'] = df['Bias'].rolling(window=std_window).mean()
df['Bias_Std'] = df['Bias'].rolling(window=std_window).std()
df['Z_Score'] = (df['Bias'] - df['Bias_Mean']) / df['Bias_Std']
df['T_Score'] = df['Z_Score'] * 10 + 50
df['Probability'] = df['Z_Score'].apply(lambda x: norm.cdf(x) * 100 if pd.notnull(x) else np.nan)
df['Acceleration_T'] = df['T_Score'].diff()

# --- Mission Control Report Logic ---
def get_mission_control_report(price, t, acc_t, prob):
    price_report = f" [PRICE]   ¥{price:,.0f} (15m Delay)"
    report = f" [REPORT]  Deviation: {t:.1f} | Prob: {prob:.1f}%"
    direction = "UP ↑" if acc_t > 0 else "DOWN ↓"
    contact = f" [CONTACT] Accel: {acc_t:.2f} ({direction})"
    
    if t >= 85 and acc_t < 0:
        consult = " [CONSULT] OVER 90! SNIPER SHORT READY!"
    elif acc_t >= 5.0:
        consult = " [CONSULT] ★ MAC POWER! GO GO! ★"
    elif t >= 75 and acc_t < 0:
        consult = " [CONSULT] LEVEL 75. Fading. Short?"
    elif t > 70:
        consult = " [CONSULT] Overheated. Prepare Sniper."
    else:
        consult = " [CONSULT] Sideways Truth. Wait."
    return f"{price_report}\n{report}\n{contact}\n{consult}"

# --- UI Layout ---
st.title(f"Nikkei 225 CME: Sniper Mission Control")

# Report Panel (Top)
panel_text = get_mission_control_report(latest['Close'], latest['T_Score'], latest['Acceleration_T'], latest['Probability'])
st.code(panel_text) # 指令室らしい等幅フォント表示

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                                     gridspec_kw={'height_ratios': [2.2, 1, 1]})

# ax1: Price
ax1.plot(df.index, df['Close'], marker='o', markersize=2, linewidth=1, color='black', label='CME Price')
ax1.plot(df.index, df['MA25'], color='orange', linewidth=2.5, label='25-Line')
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax1.legend(loc='upper left')

# ax2: T-Score
ax2.plot(df.index, df['T_Score'], color='blue', linewidth=1.5)
ax2.axhline(y=75, color='red', linestyle='-', linewidth=2)
ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
ax2.axhline(y=30, color='green', linestyle='--', linewidth=2)
ax2.fill_between(df.index, 75, 100, color='red', alpha=0.1)
ax2.set_ylabel("Deviation (T-Score)")
ax2.grid(True, alpha=0.3)

# ax3: Accel Power
colors = ['red' if x > 0 else 'blue' for x in df['Acceleration_T']]
ax3.bar(df.index, df['Acceleration_T'], color=colors, alpha=0.7)
ax3.axhline(y=0, color='black', linewidth=1.0)
ax3.set_ylabel("Power (Accel T)")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

