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

def calculate_position_size(atr_value, account_size=100000, risk_per_trade=0.02):
    """Calculate position size based on normalized ATR"""
    if pd.isna(atr_value) or atr_value == 0:
        return 0.01
    return min(account_size * risk_per_trade / (atr_value / 100), account_size * 0.1)