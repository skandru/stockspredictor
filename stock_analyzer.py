# stock_analyzer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import time

class StockAnalyzer:
    def __init__(self, lookback_period=60, prediction_period=180):
        self.lookback_period = lookback_period  # Number of days to look back
        self.prediction_period = prediction_period  # Number of days to predict ahead
        self.model = None
        self.scaler = StandardScaler()
        
    def get_technical_indicators(self, df):
        # Calculate various technical indicators
        
        # Moving averages
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2*df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2*df['Close'].rolling(window=20).std()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_StdDev'] = df['Volume'].rolling(window=20).std()
        
        # Additional momentum indicators
        # Rate of Change (ROC)
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
    
    def prepare_data(self, symbol, start_date, end_date):
        try:
            # Download historical data
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            print(f"Downloaded {len(df)} rows of historical data for {symbol}")
            
            if len(df) < 200:  # Need at least 200 days of data for meaningful analysis
                raise ValueError(f"Insufficient historical data for {symbol}: only {len(df)} days available")
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Add delay to avoid rate limiting
            time.sleep(1)  # 1 second delay between API calls
            
            # Calculate technical indicators
            df = self.get_technical_indicators(df)
            print(f"Calculated technical indicators for {symbol}")
            
            # Calculate future returns (target variable)
            df['Future_Return'] = df['Close'].shift(-self.prediction_period) / df['Close'] - 1
            
            # Create features
            feature_columns = ['Close', 'Volume', 'MA50', 'MA200', 'RSI', 'MACD', 
                             'Signal_Line', 'BB_middle', 'BB_upper', 'BB_lower', 
                             'Volume_MA', 'Volume_StdDev', 'ROC', 'ATR']
            
            # Check for missing values
            missing_values = df[feature_columns].isnull().sum()
            if missing_values.any():
                print(f"Missing values in {symbol} data: {missing_values[missing_values > 0]}")
            
            # Remove rows with NaN values
            df = df.dropna()
            print(f"After removing NaN values: {len(df)} rows remain for {symbol}")
            
            if len(df) < 100:  # Minimum required rows after cleaning
                raise ValueError(f"Insufficient clean data points for {symbol}: only {len(df)} valid rows")
            
            # Split into features and target
            X = df[feature_columns].values
            y = df['Future_Return'].values
            
            print(f"Final data shape for {symbol}: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            print(f"Error preparing data for {symbol}: {str(e)}")
            raise
    
    def train(self, X, y):
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train the model with updated parameters
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='rmse'  # Moved eval_metric here
            )
            
            # Train the model with simplified parameters
            self.model.fit(
                X_train_scaled, 
                y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate performance metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return mse, r2
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            raise
    
    def predict_returns(self, symbol, prediction_date):
        try:
            # Get recent data for prediction
            stock = yf.Ticker(symbol)
            df = stock.history(period='1y')
            
            # Calculate technical indicators
            df = self.get_technical_indicators(df)
            
            # Prepare features
            feature_columns = ['Close', 'Volume', 'MA50', 'MA200', 'RSI', 'MACD', 
                             'Signal_Line', 'BB_middle', 'BB_upper', 'BB_lower', 
                             'Volume_MA', 'Volume_StdDev', 'ROC', 'ATR']
            
            # Get the most recent data point
            recent_data = df[feature_columns].iloc[-1].values.reshape(1, -1)
            
            # Scale the features
            recent_data_scaled = self.scaler.transform(recent_data)
            
            # Make prediction
            predicted_return = self.model.predict(recent_data_scaled)[0]
            
            return predicted_return
            
        except Exception as e:
            print(f"Error making prediction for {symbol}: {str(e)}")
            raise