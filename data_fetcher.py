"""
data_fetcher.py
---------------
Yahoo Finance'ten hisse senedi verisi çeker ve teknik indikatörler ekler.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import os


def fetch_stock_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Yahoo Finance'ten OHLCV verisi çeker.
    
    Args:
        ticker:   Hisse kodu, örn. "AAPL", "THYAO.IS"
        start:    Başlangıç tarihi "YYYY-MM-DD"
        end:      Bitiş tarihi   "YYYY-MM-DD"
        interval: Zaman dilimi (1d, 1h, 15m ...)
    
    Returns:
        OHLCV + teknik indikatörlü DataFrame
    """
    print(f"[DataFetcher] {ticker} verisi indiriliyor ({start} → {end}) ...")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"'{ticker}' için veri bulunamadı. Ticker'ı kontrol et.")

    # Sütun isimlerini düzelt (MultiIndex gelebilir)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df = df.astype(float)

    print(f"[DataFetcher] {len(df)} satır veri alındı.")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fiyat verisine teknik indikatörler ekler.
    
    Eklenen indikatörler:
        Trend   : SMA20, SMA50, EMA20, MACD, MACD Signal
        Momentum: RSI14, Stochastic %K & %D
        Volatilite: Bollinger üst/alt bant, ATR
        Hacim   : OBV
        Normalize: Fiyatı % değişim olarak ifade eden 'returns'
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # --- Trend ---
    df["sma20"]        = SMAIndicator(close, window=20).sma_indicator()
    df["sma50"]        = SMAIndicator(close, window=50).sma_indicator()
    df["ema20"]        = EMAIndicator(close, window=20).ema_indicator()
    macd               = MACD(close)
    df["macd"]         = macd.macd()
    df["macd_signal"]  = macd.macd_signal()
    df["macd_diff"]    = macd.macd_diff()

    # --- Momentum ---
    df["rsi"]          = RSIIndicator(close, window=14).rsi()
    stoch              = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"]      = stoch.stoch()
    df["stoch_d"]      = stoch.stoch_signal()

    # --- Volatilite ---
    bb                 = BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]     = bb.bollinger_hband()
    df["bb_lower"]     = bb.bollinger_lband()
    df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / close
    df["atr"]          = AverageTrueRange(high, low, close, window=14).average_true_range()

    # --- Hacim ---
    df["obv"]          = OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    # --- Getiri ---
    df["returns"]      = close.pct_change()
    df["log_returns"]  = np.log(close / close.shift(1))

    # NaN içeren satırları temizle (indikatörler ilk N satırı boş bırakır)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)   # Date'i sütuna al (gym-anytrading için)

    print(f"[DataFetcher] Teknik indikatörler eklendi. Toplam özellik sayısı: {df.shape[1]}")
    return df


def prepare_train_test(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Veriyi zaman sırasına göre (shuffle yok!) train/test'e böler.
    """
    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df  = df.iloc[split:].reset_index(drop=True)
    print(f"[DataFetcher] Train: {len(train_df)} | Test: {len(test_df)} satır")
    return train_df, test_df


def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[DataFetcher] Veri kaydedildi: {path}")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"[DataFetcher] Veri yüklendi: {path} ({len(df)} satır)")
    return df
