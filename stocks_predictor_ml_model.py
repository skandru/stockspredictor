''' Below code to validate at CLI , use stock_prediction_fastapi_backend.py '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import time
from datetime import datetime, timedelta

# Global date configurations
TRAINING_START_DATE = '2021-01-01'  # Start date for training data
TRAINING_END_DATE = '2024-12-01'    # End date for training data
PREDICTION_DATE = '2025-08-01'      # Date for making predictions

# You can quickly modify these dates by changing the values above


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
        
        return df
    
    def prepare_data(self, symbol, start_date, end_date):
        try:
            # Download historical data
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Add delay to avoid rate limiting
            time.sleep(1)  # 1 second delay between API calls
            
            # Calculate technical indicators
            df = self.get_technical_indicators(df)
            
            # Calculate future returns (target variable)
            df['Future_Return'] = df['Close'].shift(-self.prediction_period) / df['Close'] - 1
            
            # Create features
            feature_columns = ['Close', 'Volume', 'MA50', 'MA200', 'RSI', 'MACD', 
                             'Signal_Line', 'BB_middle', 'BB_upper', 'BB_lower', 'Volume_MA']
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Split into features and target
            X = df[feature_columns].values
            y = df['Future_Return'].values
            
            return X, y
            
        except Exception as e:
            print(f"Error preparing data for {symbol}: {str(e)}")
            raise
    
    def train(self, X, y):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train the model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return mse, r2
    
    def predict_returns(self, symbol, current_date):
        # Get recent data for prediction
        stock = yf.Ticker(symbol)
        df = stock.history(period='1y')
        
        # Calculate technical indicators
        df = self.get_technical_indicators(df)
        
        # Prepare features
        feature_columns = ['Close', 'Volume', 'MA50', 'MA200', 'RSI', 'MACD', 
                         'Signal_Line', 'BB_middle', 'BB_upper', 'BB_lower', 'Volume_MA']
        
        # Get the most recent data point
        recent_data = df[feature_columns].iloc[-1].values.reshape(1, -1)
        
        # Scale the features
        recent_data_scaled = self.scaler.transform(recent_data)
        
        # Make prediction
        predicted_return = self.model.predict(recent_data_scaled)[0]
        
        return predicted_return

def get_sp500_symbols():
    """Get first 100 S&P 500 constituents using Wikipedia"""
    try:
        # Read S&P 500 table from Wikipedia
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = tables[0]
        return sp500_table['Symbol'].tolist()[:100]  # Return first 100 symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {str(e)}")
        # Fallback to first 100 major S&P 500 components
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'XOM', 'UNH', 'JNJ', 
                'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'LLY', 'ABBV', 'PEP', 'KO', 'BAC', 
                'AVGO', 'COST', 'MCD', 'TMO', 'CSCO', 'ACN', 'ABT', 'WMT', 'CRM', 'LIN', 'DHR',
                'VZ', 'CMCSA', 'ADBE', 'NKE', 'NEE', 'TXN', 'PM', 'RTX', 'UNP', 'BMY', 'MS',
                'QCOM', 'HON', 'UPS', 'INTC', 'CVS', 'INTU', 'ORCL', 'WFC', 'T', 'AMD', 'CAT']

def analyze_sp500_stocks():
    sp500_symbols = get_sp500_symbols()
    print(f"Analyzing {len(sp500_symbols)} S&P 500 stocks...")
    
    analyzer = StockAnalyzer(lookback_period=60, prediction_period=180)
    results = []
    
    for symbol in sp500_symbols:
        try:
            print(f"Processing {symbol}...")
            # Prepare data
            X, y = analyzer.prepare_data(symbol, TRAINING_START_DATE, TRAINING_END_DATE)
            
            # Train model
            mse, r2 = analyzer.train(X, y)
            
            # Make prediction
            predicted_return = analyzer.predict_returns(symbol, PREDICTION_DATE)
            
            results.append({
                'Symbol': symbol,
                'Predicted_Return': predicted_return * 100,  # Convert to percentage
                'Model_R2': r2
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    # Sort stocks by predicted return
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Predicted_Return', ascending=False)
    
    # Display results
    pd.set_option('display.float_format', lambda x: '%.3f' % x)  # Format float numbers
    
    # Add current price and market cap
    current_prices = {}
    market_caps = {}
    for symbol in results_df['Symbol']:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            current_prices[symbol] = info.get('regularMarketPrice', None)
            market_caps[symbol] = info.get('marketCap', None)
        except:
            current_prices[symbol] = None
            market_caps[symbol] = None
    
    results_df['Current_Price'] = results_df['Symbol'].map(current_prices)
    results_df['Market_Cap_B'] = results_df['Symbol'].map(market_caps).apply(lambda x: x/1e9 if x else None)
    
    # Format the results
    print("\nTop Stock Predictions for Next 6-12 Months:")
    print("============================================")
    print("\nTop 20 Stocks by Predicted Return:")
    print(results_df.head(20).to_string(index=False, 
          columns=['Symbol', 'Predicted_Return', 'Current_Price', 'Market_Cap_B', 'Model_R2'],
          float_format=lambda x: '{:,.2f}'.format(x) if pd.notnull(x) else 'N/A'))
    
    return results_df

if __name__ == "__main__":
    print("Starting S&P 500 stock analysis...")
    results = analyze_sp500_stocks()