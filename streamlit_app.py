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

# --- 3. Data Acquisition (Robust Edition) ---
@st.cache_data(ttl=600)
def load_data():
    # データをダウンロード
    raw = yf.download(ticker_sym, period=period, interval=interval, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    
    # 階層構造(MultiIndex)を解消
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # カラム名を一旦すべて小文字にしてから先頭大文字に統一（KeyError対策の肝）
    df.columns = [str(c).lower().capitalize() for c in df.columns]
    
    # 価格データがない行を削除
    df = df.dropna(subset=['Close'])
    
    # インデックスを振り直し（ここで土日祝を詰める）
    # 元の日付インデックスは 'Datetime' か 'Date' という名前のカラムになる
    df = df.reset_index()
    
    # 1列目（日付）の名前を 'Timestamp' に固定
    df.rename(columns={df.columns[0]: 'Timestamp'}, inplace=True)
    
    return df

df = load_data()

# データが空の場合のガード
if df.empty or 'Close' not in df.columns:
    st.error("Market Data is unavailable right now.")
    st.stop()

# --- 4. Calculation ---
# Closeを使って計算。名前を統一したのでここでKeyErrorは起きません。
df['MA25'] = df['Close'].rolling(window=ma_window).mean()
df['Bias'] = (df['Close'] - df['MA25']) / df['MA25'] * 100
df['Bias_Mean'] = df['Bias'].rolling(window=std_window).mean()
df['Bias_Std'] = df['Bias'].rolling(window=std_window).std()

df['Z_Score'] = (df['Bias'] - df['Bias_Mean']) / df['Bias_Std']
df['T_Score'] = df['Z_Score'] * 10 + 50
df['Probability'] = df['Z_Score'].apply(lambda x: norm.cdf(x) * 100 if pd.notnull(x) else np.nan)
df['Acceleration_T'] = df['T_Score'].diff()

# 最新の1行を取得
latest = df.iloc[-1]

# --- 5. Report Logic ---
def get_mission_control_report(price, t, acc_t, prob):
    price_report = f" [PRICE]   ¥{price:,.0f} (Active Hours)"
    report = f" [REPORT]  Deviation: {t:.1f} | Prob: {prob:.1f}%"
    direction = "UP ↑" if acc_t > 0 else "DOWN ↓"
    contact = f" [CONTACT] Accel: {acc_t:.2f} ({direction})"
    
    if t >= 85 and acc_t < 0:
        consult = " [CONSULT] OVER 85! SNIPER SHORT READY!"
    elif acc_t >= 5.0:
        consult = " [CONSULT] ★ MAC POWER! GO GO! ★"
    elif t >= 75 and acc_t < 0:
        consult = " [CONSULT] LEVEL 75. Fading. Short?"
    elif t > 70:
        consult = " [CONSULT] Overheated. Prepare Sniper."
    else:
        consult = " [CONSULT] Sideways Truth. Wait."
    return f"{price_report}\n{report}\n{contact}\n{consult}"

# --- 6. UI ---
st.title("Nikkei 225 CME: Sniper Mission Control")
st.code(get_mission_control_report(latest['Close'], latest['T_Score'], latest['Acceleration_T'], latest['Probability']))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2.2, 1, 1]})

# X軸には df.index（連番）を使うことで、おっちゃんの望み通り「土日祝」を詰める
x_axis = df.index

# ax1: Price
ax1.plot(x_axis, df['Close'], marker='o', markersize=2, linewidth=1, color='black', label='Price')
ax1.plot(x_axis, df['MA25'], color='orange', linewidth=2.5, label='25-Line')
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# ax2: T-Score
ax2.plot(x_axis, df['T_Score'], color='blue', linewidth=1.5)
ax2.axhline(y=75, color='red', linestyle='-', linewidth=2)
ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
ax2.axhline(y=30, color='green', linestyle='--', linewidth=2)
ax2.fill_between(x_axis, 75, 100, color='red', alpha=0.1)
ax2.set_ylabel("T-Score")
ax2.grid(True, alpha=0.3)

# ax3: Accel
ax3.bar(x_axis, df['Acceleration_T'], color=['red' if x > 0 else 'blue' for x in df['Acceleration_T']], alpha=0.7)
ax3.axhline(y=0, color='black', linewidth=1.0)
ax3.set_ylabel("Power")
ax3.grid(True, alpha=0.3)

# X軸のラベル（時刻）を表示
tick_idx = np.linspace(0, len(df)-1, 10, dtype=int)
ax3.set_xticks(tick_idx)
# Timestampカラムから時刻を読み取って表示
ax3.set_xticklabels([pd.to_datetime(df['Timestamp'].iloc[i]).strftime('%m/%d %H:%M') for i in tick_idx], rotation=45)

st.pyplot(fig)
