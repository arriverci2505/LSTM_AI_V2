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

# ════════════════════════════════════════════════════════════════════════════
# 1. LIVE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

LIVE_CONFIG = {
    # Exchange Settings
    'exchange_id': 'binance',  # binance, bybit, kraken...
    'api_key': 'YOUR_API_KEY',
    'api_secret': 'YOUR_API_SECRET',
    'sandbox_mode': True,      # True = Testnet, False = Real Money (CẨN TRỌNG!)
    
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

# ════════════════════════════════════════════════════════════════════════════
# 3. TRADING BOT CLASS
# ════════════════════════════════════════════════════════════════════════════

class AI_Trader:
    def __init__(self, config):
        self.config = config
        self.connect_exchange()
        self.load_brain()
        
    def connect_exchange(self):
        """Kết nối sàn giao dịch"""
        try:
            exchange_class = getattr(ccxt, self.config['exchange_id'])
            self.exchange = exchange_class({
                'apiKey': self.config['api_key'],
                'secret': self.config['api_secret'],
                'enableRateLimit': True,
                'options': {'defaultType': 'future'} # Giả sử đánh Futures
            })
            self.exchange.set_sandbox_mode(self.config['sandbox_mode'])
            print(f"✅ Connected to {self.config['exchange_id']} (Sandbox: {self.config['sandbox_mode']})")
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            exit()

    def load_brain(self):
        """Load Model và Scaler"""
        try:
            print(f"⏳ Loading model from {self.config['model_path']}...")
            self.model = load_model(self.config['model_path'])
            self.scaler = joblib.load(self.config['scaler_path'])
            print("✅ Brain loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model/scaler: {e}")
            print("Hãy chắc chắn bạn đã train model và đường dẫn file đúng.")
            exit()

    def fetch_data(self):
        """Lấy dữ liệu nến mới nhất"""
        try:
            # Lấy nhiều nến hơn limit một chút để trừ hao
            ohlcv = self.exchange.fetch_ohlcv(
                self.config['symbol'], 
                self.config['timeframe'], 
                limit=self.config['limit'] + 50
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"⚠️ Fetch Error: {e}")
            return None

    def predict(self, df):
        """Xử lý dữ liệu và đưa ra dự đoán"""
        # 1. Calculate Indicators
        df_processed = calculate_technical_indicators_live(df)
        df_processed.dropna(inplace=True)
        
        # 2. Select Features
        features_df = get_live_features(df_processed)
        if features_df is None or len(features_df) < self.config['sequence_length']:
            print("⚠️ Not enough data for prediction")
            return None, None
            
        # 3. Scale Data (Chỉ transform, không fit)
        # Lấy đúng sequence_length nến cuối cùng
        last_sequence = features_df.tail(self.config['sequence_length']).values
        scaled_sequence = self.scaler.transform(last_sequence)
        
        # Reshape for LSTM: (1, 60, n_features)
        input_data = np.expand_dims(scaled_sequence, axis=0)
        
        # 4. Predict
        # Model trả về [max_gain, max_loss, net_return]
        predictions = self.model.predict(input_data, verbose=0)
        
        p_gain = predictions[0][0][0] # Max Gain
        p_loss = predictions[1][0][0] # Max Loss
        p_ret = predictions[2][0][0]  # Net Return
        
        return p_gain, p_loss, p_ret

    def execute_logic(self, p_gain, p_loss, p_ret, current_price):
        """Logic vào lệnh dựa trên dự đoán"""
        
        # --- 1. Tính toán chỉ số ---
        # Logic tính Confidence giống hệt Backtest
        mae = self.config['mae_return_avg']
        z_score = abs(p_ret) / (mae + 1e-10)
        
        # Consistency Check
        consistency_penalty = 1.0
        if (p_ret > 0 and p_gain < 0) or (p_ret < 0 and p_loss > 0):
            consistency_penalty = 0.6
            
        confidence = ((2 / (1 + np.exp(-z_score * 1.5)) - 1) * 100) * consistency_penalty
        
        # --- 2. Xác định hướng ---
        if p_ret > 0:
            direction = 'LONG'
            base_tp = abs(p_gain)
            base_sl = abs(p_loss)
        else:
            direction = 'SHORT'
            base_tp = abs(p_loss)
            base_sl = abs(p_gain)
            
        reward_risk = base_tp / (base_sl + 1e-10)

        print(f"\n📊 ANALYSIS [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"   Price: {current_price} | Pred Return: {p_ret:.4f}%")
        print(f"   Direction: {direction} | Conf: {confidence:.2f}% | R/R: {reward_risk:.2f}")

        # --- 3. Điều kiện vào lệnh ---
        if (confidence >= self.config['min_confidence'] and 
            reward_risk >= self.config['min_reward_risk'] and
            abs(p_ret) >= self.config['min_predicted_return']):
            
            print(f"🚀 SIGNAL FOUND: {direction} BTC!")
            
            # Tính TP/SL
            mae_buffer = mae * self.config['sl_buffer']
            sl_pct = max((base_sl + mae_buffer) / 100, 0.01) # Min SL 1%
            tp_pct = max(base_tp / 100, 0.012)
            
            # Thực hiện lệnh (Ở đây chỉ in ra, bạn bỏ comment đoạn dưới để đánh thật)
            self.place_order(direction, current_price, sl_pct, tp_pct)
        else:
            print("💤 No valid signal. Waiting...")

    def place_order(self, direction, price, sl_pct, tp_pct):
        """Gửi lệnh lên sàn"""
        print(f"⚡ EXECUTING {direction}...")
        
        # Tính SL/TP Price
        if direction == 'LONG':
            side = 'buy'
            sl_price = price * (1 - sl_pct)
            tp_price = price * (1 + tp_pct)
        else:
            side = 'sell'
            sl_price = price * (1 + sl_pct)
            tp_price = price * (1 - tp_pct)
            
        print(f"   Entry: {price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
        
        # --- CODE THỰC THI (CCXT) ---
        # Lưu ý: Cần tính toán khối lượng (amount) dựa trên balance
        try:
            # 1. Lấy Balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            
            # 2. Tính Size (Risk 2%)
            risk_amt = usdt_balance * self.config['risk_per_trade']
            # Công thức size: Risk / (Entry - SL)
            # Simplified for leverage:
            position_value = risk_amt / sl_pct 
            amount = position_value / price
            
            # Check leverage limit
            max_val = usdt_balance * self.config['leverage']
            if position_value > max_val:
                amount = max_val / price
                print("⚠️ Adjusted size due to leverage limit")

            # 3. Đặt lệnh Market
            # order = self.exchange.create_market_order(
            #     symbol=self.config['symbol'],
            #     side=side,
            #     amount=amount
            # )
            # print(f"✅ Order Placed: {order['id']}")
            
            # 4. Đặt SL/TP (Tùy sàn hỗ trợ gộp lệnh hay phải đặt riêng)
            # ... Code đặt OCO order hoặc Stop Market order ...
            
            print(f"⚠️ Simulation Mode: Would buy {amount:.4f} BTC")
            
        except Exception as e:
            print(f"❌ Order Failed: {e}")

    def run(self):
        """Vòng lặp chính"""
        print("🤖 Bot Started. Waiting for next candle...")
        while True:
            try:
                # Đồng bộ thời gian để chạy ngay khi đóng nến
                now = datetime.now()
                # Ví dụ 15m: chạy vào phút 0, 15, 30, 45
                # Hoặc đơn giản là chạy mỗi 30 giây để check
                
                df = self.fetch_data()
                if df is not None:
                    current_price = df['Close'].iloc[-1]
                    
                    # Dự đoán
                    p_gain, p_loss, p_ret = self.predict(df)
                    
                    if p_gain is not None:
                        self.execute_logic(p_gain, p_loss, p_ret, current_price)
                
                # Sleep 60s để tránh spam API
                time.sleep(60) 
                
            except KeyboardInterrupt:
                print("🛑 Bot stopped by user.")
                break
            except Exception as e:
                print(f"⚠️ Error in loop: {e}")
                time.sleep(10)

# ════════════════════════════════════════════════════════════════════════════
# 4. MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = AI_Trader(LIVE_CONFIG)
    bot.run()