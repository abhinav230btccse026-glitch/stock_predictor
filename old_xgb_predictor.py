import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import xgboost as xgb
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')



def predict_with_confidence(model, scaler, features, symbol, label_mapping):
    """Generate prediction with detailed analysis"""
    try:
        latest_features = features.iloc[-1:].fillna(features.mean())
        latest_scaled = scaler.transform(latest_features)
        
        prediction_numeric = model.predict(latest_scaled)[0]
        probabilities = model.predict_proba(latest_scaled)[0]
        confidence = max(probabilities) * 100
        
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        prediction = reverse_mapping[prediction_numeric]
        
        latest_vals = latest_features.iloc[0]
        
        reasons = []
        if latest_vals['RSI'] < 30:
            reasons.append("RSI oversold (bullish)")
        elif latest_vals['RSI'] > 70:
            reasons.append("RSI overbought (bearish)")
        if latest_vals['MACD_Diff'] > 0:
            reasons.append("MACD bullish crossover")
        elif latest_vals['MACD_Diff'] < 0:
            reasons.append("MACD bearish crossover")
        if latest_vals['Stoch_K_Daily'] < 20:
            reasons.append("Daily Stochastic oversold")
        elif latest_vals['Stoch_K_Daily'] > 80:
            reasons.append("Daily Stochastic overbought")
        if latest_vals['Stoch_K_Hourly'] > 0:
            reasons.append("Hourly Stochastic stronger than daily")
        elif latest_vals['Stoch_K_Hourly'] < 0:
            reasons.append("Hourly Stochastic weaker than daily")
        if latest_vals['VWAP_Position'] > 2:
            reasons.append("Price above VWAP (bullish)")
        elif latest_vals['VWAP_Position'] < -2:
            reasons.append("Price below VWAP (bearish)")
        if latest_vals['BB_Position'] > 0.8:
            reasons.append("Near BB upper band (overbought)")
        elif latest_vals['BB_Position'] < 0.2:
            reasons.append("Near BB lower band (oversold)")
        if latest_vals['MA_Cross_Bullish']:
            reasons.append("MA bullish crossover")
        if latest_vals['Pattern_Score'] == 1:
            reasons.append("Bullish candlestick pattern")
        elif latest_vals['Pattern_Score'] == -1:
            reasons.append("Bearish candlestick pattern")
        
        position_size = calculate_position_size(latest_vals['ATR'])
        
        emoji = '📈' if prediction == 'BUY' else '📉' if prediction == 'SELL' else '👐'
        hold_period = "for 7 days" if prediction == 'BUY' else ""
        reason_text = "; ".join(reasons) if reasons else "Mixed signals"
        
        output = f"[{emoji} {prediction}] - {symbol} {hold_period} | Confidence: {confidence:.1f}% | {reason_text}"
        if prediction != 'HOLD':
            output += f" | Position Size: ₹{position_size:,.0f}"
        
        return output, confidence, prediction
        
    except Exception as e:
        return f"Error predicting {symbol}: {str(e)}", 0, 'HOLD'
