import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

base_path = os.path.dirname(__file__)

# ════════════════════════════════════════════════════════════════════════════
# 1. LIVE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

LIVE_CONFIG = {
    # Exchange Settings
    'exchange_id': 'binance',  # binance, bybit, kraken...
    
    # Trading Settings
    'symbol': 'BTC/USDT',      # Lưu ý format của CCXT thường là Base/Quote
    'timeframe': '15m',
    'limit': 500,              # Cần đủ dữ liệu để tính indicators (EMA 200)
    
    # Model Paths (Đường dẫn đến file đã train)
    
    'model_path': './LSTM_Trading_Models_V4.2/models/BTC-USDT_best.keras',
    'scaler_path': './LSTM_Trading_Models_V4.2/models/scaler_BTC-USDT.pkl',
    
    # Strategy Parameters (Phải khớp với file Training)
    'sequence_length': 60,
    'min_confidence': 50,      
    'min_reward_risk': 1.5,
    'min_predicted_return': 0.1, # %
    'risk_per_trade': 0.02,    # 2% Balance
    'leverage': 5,             # Đòn bẩy muốn sử dụng
    'sl_buffer': 1.2,          # Hệ số nới SL (dựa trên MAE)
    'mae_return_avg': 0.15     # Giá trị MAE trung bình từ Backtest (Điền tay vào đây hoặc load từ log)
}

# ════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (MUST MATCH TRAINING EXACTLY)
# ════════════════════════════════════════════════════════════════════════════

def calculate_technical_indicators_live(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy y nguyên logic từ file training để đảm bảo nhất quán dữ liệu.
    Đã lược bỏ phần xử lý Target vì Live không có tương lai.
    """
    df = df.copy()
    
    # Đảm bảo index là datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Time features
    hour = df.index.hour
    df['Hour_Sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * hour / 24)
    
    day_of_week = df.index.dayofweek
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    df['Is_Weekend'] = (df.index.dayofweek >= 5).astype(float)
    df['Asian_Session'] = ((hour >= 0) & (hour < 8)).astype(float)
    df['European_Session'] = ((hour >= 8) & (hour < 16)).astype(float)
    df['US_Session'] = ((hour >= 16) & (hour <= 23)).astype(float)

    # Price features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility
    df['Volatility_10'] = df['Returns'].rolling(10).std()
    df['Volatility_30'] = df['Returns'].rolling(30).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Norm'] = (df['RSI'] - 50) / 50

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Norm'] = df['MACD'] / (df['Close'] + 1e-10)

    # BB
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma20 + (2 * std20)
    df['BB_Lower'] = sma20 - (2 * std20)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (sma20 + 1e-10)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / (df['Close'] + 1e-10)

    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-10)
    df['Volume_Trend'] = df['Volume'].rolling(10).apply(lambda x: 1 if x[-1] > x[0] else 0, raw=True)
    df['Log_Volume_Ratio'] = np.log1p(df['Volume_Ratio'])

    # Candle patterns
    df['Body_Size'] = np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'] + 1e-10)
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-10)

    # EMAs
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df['EMA_9_Dist'] = (df['Close'] - df['EMA_9']) / (df['Close'] + 1e-10)
    df['EMA_21_Dist'] = (df['Close'] - df['EMA_21']) / (df['Close'] + 1e-10)
    df['EMA_50_Dist'] = (df['Close'] - df['EMA_50']) / (df['Close'] + 1e-10)
    df['EMA_200_Dist'] = (df['Close'] - df['EMA_200']) / (df['Close'] + 1e-10)

    # Momentum
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_10'] = df['Close'].pct_change(10)
    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100

    # Support/Resistance
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'] + 1e-10)
    
    return df

def get_live_features(df):
    """Chọn lọc features y hệt như lúc training (Hàm select_features trong code cũ)"""
    # LƯU Ý: Danh sách này phải KHỚP CHÍNH XÁC với danh sách features sau khi đã 
    # loại bỏ correlation cao ở bước training.
    # Tốt nhất là bạn nên lưu list features ra file json khi train, nhưng ở đây tôi
    # sẽ list ra các features mặc định thường được chọn trong code training của bạn.
    
    selected_features = [
        'RSI_Norm', 'MACD_Norm', 'MACD_Hist',
        'BB_Width', 'BB_Position',
        'ATR_Pct','Log_Volume_Ratio', 'Volume_Trend',
        'Body_Size', 'Upper_Shadow', 'Lower_Shadow',
        'EMA_9_Dist', 'EMA_21_Dist', 'EMA_200_Dist',
        'Momentum_5', 'Momentum_10',
        'Price_Position', 'Volatility_10',
        # Time features
        'Hour_Sin', 'Hour_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos', 
        'Is_Weekend', 'Asian_Session', 'European_Session', 'US_Session'
    ]
    
    # Kiểm tra xem features có tồn tại trong df không
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        print(f"⚠️ Missing features: {missing}")
        # Nếu thiếu có thể do chưa đủ dữ liệu để tính (ví dụ EMA 200)
        return None
        
    return df[selected_features]

# ==========================================
# 3. GIAO DIỆN STREAMLIT & TRADINGVIEW
# ==========================================
st.set_page_config(page_title="AI Bitcoin Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1c2128; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    
    base_path = os.path.dirname(__file__)
    
    # Kết hợp với tên file để tạo đường dẫn tuyệt đối
    model = load_model(base_path, "BTC_USDT_best.pkl")
    scaler = joblib.load(base_path, "BTC_USDT_ensemble.pkl")
    # Kiểm tra tồn tại để báo lỗi rõ ràng trên Streamlit
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model tại: {model_path}")
        st.stop()
        
    model = joblib.load(model_path)

    return model, scaler

model, scaler = load_assets()

# Sidebar info
st.sidebar.title("🤖 AI Control Panel")
st.sidebar.info("Model: LSTM + Multi-Head Attention\nStatus: Live Monitoring")

# Chia cột giao diện
col_signal, col_chart = st.columns([1, 1.8])

with col_chart:
    st.markdown("### 📊 Market View")
    tv_html = """
    <div style="height:600px;">
        <div id="tv_chart" style="height:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({
            "autosize": true, "symbol": "BINANCE:BTCUSDT", "interval": "15",
            "timezone": "Asia/Ho_Chi_Minh", "theme": "dark", "style": "1",
            "locale": "vi", "container_id": "tv_chart"
        });
        </script>
    </div>
    """
    components.html(tv_html, height=620)

with col_signal:
    st.markdown("### 🧠 AI Prediction")
    signal_box = st.empty()
    price_metrics = st.empty()
    details_box = st.empty()

# ==========================================
# 4. LUỒNG CHẠY LIVE
# ==========================================
try:
    model, scaler = load_assets()
    exchange = ccxt.binance()

    while True:
        # 1. Lấy dữ liệu
        ohlcv = exchange.fetch_ohlcv(ST_CONFIG['symbol'], timeframe='15m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['ts'], unit='ms') + timedelta(hours=7)
        df.set_index('Date', inplace=True)

        # 2. Xử lý Feature
        df_indicators = calculate_indicators(df)
        feat_df = get_selected_features(df_indicators)
        
        # 3. Dự đoán
        if len(feat_df) >= ST_CONFIG['sequence_length']:
            last_seq = feat_df.tail(ST_CONFIG['sequence_length']).values
            last_seq_scaled = scaler.transform(last_seq)
            input_data = np.expand_dims(last_seq_scaled, axis=0)
            
            # Predict: [Max_Gain, Max_Loss, Net_Return]
            preds = model.predict(input_data, verbose=0)
            p_gain, p_loss, p_ret = preds[0][0][0], preds[1][0][0], preds[2][0][0]

            # 4. Hiển thị UI
            current_price = df['Close'].iloc[-1]
            
            # Logic Signal
            if p_ret > 0.2:
                color, label, icon = "#00ff88", "STRONG BUY", "🚀"
            elif p_ret > 0.05:
                color, label, icon = "#2ecc71", "BUY", "📈"
            elif p_ret < -0.2:
                color, label, icon = "#ff4b4b", "STRONG SELL", "💀"
            elif p_ret < -0.05:
                color, label, icon = "#e74c3c", "SELL", "📉"
            else:
                color, label, icon = "#8b949e", "WAITING", "⚖️"

            signal_box.markdown(f"""
                <div style="background-color:{color}22; border: 2px solid {color}; padding:25px; border-radius:15px; text-align:center;">
                    <h1 style="color:{color}; margin:0;">{icon} {label}</h1>
                    <h2 style="margin:5px 0;">${current_price:,.2f}</h2>
                    <p style="opacity:0.8;">Dự báo Net Return: {p_ret:+.3f}%</p>
                </div>
            """, unsafe_allow_html=True)

            with price_metrics.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Dự báo Max Gain", f"{p_gain:.2f}%")
                m2.metric("Dự báo Max Loss", f"{p_loss:.2f}%")
                m3.metric("R/R Ratio", f"{abs(p_gain/p_loss):.2f}" if p_loss !=0 else "N/A")

            details_box.write(f"⏱️ Cập nhật lúc: {datetime.now().strftime('%H:%M:%S')}")

        time.sleep(30) # Refresh mỗi 30 giây

except Exception as e:
    st.error(f"Lỗi vận hành: {e}")
