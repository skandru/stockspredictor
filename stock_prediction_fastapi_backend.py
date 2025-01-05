from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Optional
import pandas as pd
from stock_analyzer import StockAnalyzer  # Import from your existing code

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint for the API"""
    return {
        "message": "Welcome to the Stock Predictor API",
        "version": "1.0",
        "endpoints": {
            "/": "GET - This help message",
            "/api/stocks/predict": "POST - Predict stocks",
            "/api/stocks/symbols": "GET - Get stock symbols"
        }
    }

class PredictionRequest(BaseModel):
    stocks: List[str]
    prediction_date: str
    analyze_all_sp500: bool = False

# Rest of your existing code remains the same...
def get_sp500_symbols():
    """Get S&P 500 symbols with company names"""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        # Clean up symbols and create list of dicts with symbol and company name
        sp500_data = [
            {
                'symbol': symbol.replace('.', '-'),
                'companyName': company
            }
            for symbol, company in zip(df['Symbol'], df['Security'])
        ]
        return sp500_data
    except Exception as e:
        print(f"Error in get_sp500_symbols: {str(e)}")
        # Fallback to major S&P 500 components
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'XOM', 'UNH', 'JNJ',
                'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'LLY', 'ABBV', 'PEP', 'KO', 'BAC',
                'AVGO', 'COST', 'MCD', 'TMO', 'CSCO', 'ACN', 'ABT', 'WMT', 'CRM', 'LIN', 'DHR',
                'VZ', 'CMCSA', 'ADBE', 'NKE', 'NEE', 'TXN', 'PM', 'RTX', 'UNP', 'BMY', 'MS',
                'QCOM', 'HON', 'UPS', 'INTC', 'CVS', 'INTU', 'ORCL']

def get_nasdaq100_symbols():
    """Get Nasdaq 100 symbols"""
    try:
        # Fallback to hardcoded list of major Nasdaq 100 components
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO', 'ASML', 'PEP', 
                'COST', 'ADBE', 'CSCO', 'NFLX', 'TMUS', 'CMCSA', 'AMD', 'INTC', 'INTU', 'QCOM',
                'AMGN', 'HON', 'AMAT', 'ISRG', 'ADI', 'BKNG', 'SBUX', 'MDLZ', 'GILD', 'LRCX',
                'ADP', 'REGN', 'VRTX', 'PANW', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ABNB', 'FTNT',
                'KDP', 'ORLY', 'KHC', 'MNST', 'PYPL', 'CTAS', 'MCHP', 'ADSK', 'CHTR', 'MAR']
    except Exception as e:
        print(f"Error in get_nasdaq100_symbols: {str(e)}")
        return []  # Return empty list instead of raising exception

@app.get("/api/stocks/symbols")
async def get_stock_symbols():
    """Get both S&P 500 and Nasdaq 100 symbols"""
    try:
        sp500 = get_sp500_symbols()
        nasdaq100 = get_nasdaq100_symbols()
        
        return {
            "sp500": sp500,
            "nasdaq100": nasdaq100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stocks/predict")
async def predict_stocks(request: PredictionRequest):
    try:
        # Calculate training dates
        current_date = datetime.now()
        training_start_date = (current_date - timedelta(days=4*365)).strftime('%Y-%m-%d')
        training_end_date = (current_date - timedelta(days=3)).strftime('%Y-%m-%d')
        
        # Get stock list
        stocks_to_analyze = get_sp500_symbols() if request.analyze_all_sp500 else request.stocks
        print(f"Analyzing stocks: {stocks_to_analyze}")
        
        # Initialize analyzer
        analyzer = StockAnalyzer(lookback_period=60, prediction_period=180)
        results = []
        errors = []
        
        for symbol in stocks_to_analyze:
            try:
                print(f"Processing {symbol}...")
                # Prepare data and make prediction
                X, y = analyzer.prepare_data(symbol, training_start_date, training_end_date)
                print(f"{symbol}: Data prepared, shape: {X.shape}")
                
                mse, r2 = analyzer.train(X, y)
                print(f"{symbol}: Model trained, R2: {r2}")
                
                predicted_return = analyzer.predict_returns(symbol, request.prediction_date)
                print(f"{symbol}: Prediction made: {predicted_return}")
                
                # Get current stock info
                stock = yf.Ticker(symbol)
                info = stock.info
                
                results.append({
                    'symbol': symbol,
                    'predicted_return': float(predicted_return * 100),
                    'current_price': info.get('regularMarketPrice'),
                    'market_cap_B': info.get('marketCap', 0) / 1e9 if info.get('marketCap') else None,
                    'model_r2': float(r2),
                    'company_name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', '')
                })
                print(f"{symbol}: Successfully processed")
                
            except Exception as e:
                error_msg = f"Error processing {symbol}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                continue
        
        # Sort by predicted return and get top 50
        results.sort(key=lambda x: x['predicted_return'], reverse=True)
        top_50_results = results[:50]
        
        response = {
            "predictions": top_50_results,
            "metadata": {
                "training_start_date": training_start_date,
                "training_end_date": training_end_date,
                "prediction_date": request.prediction_date,
                "total_stocks_analyzed": len(results),
                "total_errors": len(errors),
                "errors": errors[:5] if errors else [],  # Include first 5 errors in metadata
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if not results and errors:
            print("No successful predictions, all attempts failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to process any stocks successfully",
                    "errors": errors[:5]  # First 5 errors
                }
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)