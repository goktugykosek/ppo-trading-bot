"""
trading_env.py
--------------
gym-anytrading tabanlı özelleştirilmiş trading ortamı.

Aksiyon uzayı:
    0 → HOLD  (bekle)
    1 → BUY   (al)
    2 → SELL  (sat)

Gözlem uzayı:
    Son `window_size` adımdaki normalize edilmiş feature vektörü.

Ödül fonksiyonu:
    Gerçekleşen P&L (profit & loss) + pozisyon tutma cezası
    → Ajan hem kâr etmek hem de gereksiz bekleme yapmamak için optimize eder.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional


class TradingEnv(gym.Env):
    """
    Tek hisse senedi üzerinde long/flat işlem yapan RL ortamı.
    
    Parametreler:
        df          : Teknik indikatörlü DataFrame (data_fetcher'dan gelir)
        window_size : Ajanın gördüğü geçmiş adım sayısı
        initial_balance: Başlangıç sermayesi ($)
        commission  : İşlem komisyonu (oransal, örn. 0.001 = %0.1)
        reward_scaling: Ödülü ölçeklendirme faktörü
    """

    metadata = {"render_modes": ["human"]}

    # Gözlemde kullanılacak feature sütunları (Date ve OHLCV hariç)
    FEATURE_COLS = [
        "Close", "Volume",
        "sma20", "sma50", "ema20",
        "macd", "macd_signal", "macd_diff",
        "rsi", "stoch_k", "stoch_d",
        "bb_width", "atr",
        "obv", "returns", "log_returns"
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
        initial_balance: float = 10_000.0,
        commission: float = 0.001,
        reward_scaling: float = 1e-4,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.df              = df.copy()
        self.window_size     = window_size
        self.initial_balance = initial_balance
        self.commission      = commission
        self.reward_scaling  = reward_scaling
        self.render_mode     = render_mode

        # Feature matrisini hazırla ve normalize et
        self._features = self._build_features()
        self.n_features = self._features.shape[1]

        # Aksiyon: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Gözlem: (window_size × n_features) düzleştirilmiş vektör
        obs_size = self.window_size * self.n_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32
        )

        # Durum değişkenleri (reset'te sıfırlanır)
        self._current_step   = 0
        self._balance        = initial_balance
        self._shares_held    = 0.0
        self._entry_price    = 0.0
        self._total_profit   = 0.0
        self._trade_history  = []

    # ------------------------------------------------------------------ #
    #  Yardımcı metodlar                                                   #
    # ------------------------------------------------------------------ #

    def _build_features(self) -> np.ndarray:
        """Feature sütunlarını seçer ve Z-score normalize eder."""
        cols = [c for c in self.FEATURE_COLS if c in self.df.columns]
        arr  = self.df[cols].values.astype(np.float32)

        # Z-score normalizasyonu (her sütun bağımsız)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0) + 1e-8
        return (arr - mean) / std

    def _get_observation(self) -> np.ndarray:
        """Güncel penceredeki feature vektörünü döndürür."""
        start = self._current_step
        end   = self._current_step + self.window_size
        obs   = self._features[start:end]           # (window_size, n_features)
        return obs.flatten().astype(np.float32)

    def _current_price(self) -> float:
        idx = self._current_step + self.window_size - 1
        return float(self.df["Close"].iloc[idx])

    def _portfolio_value(self) -> float:
        return self._balance + self._shares_held * self._current_price()

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                        #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step  = 0
        self._balance       = self.initial_balance
        self._shares_held   = 0.0
        self._entry_price   = 0.0
        self._total_profit  = 0.0
        self._trade_history = []
        return self._get_observation(), {}

    def step(self, action: int):
        price = self._current_price()
        reward = 0.0
        info   = {}

        # --- Aksiyon işle ---
        if action == 1:   # BUY
            if self._shares_held == 0 and self._balance > 0:
                # Tüm bakiye ile al
                cost             = self._balance
                commission_fee   = cost * self.commission
                self._shares_held = (cost - commission_fee) / price
                self._balance    = 0.0
                self._entry_price = price
                self._trade_history.append(("BUY", self._current_step, price))

        elif action == 2:  # SELL
            if self._shares_held > 0:
                # Tüm pozisyonu kapat
                gross            = self._shares_held * price
                commission_fee   = gross * self.commission
                net_proceeds     = gross - commission_fee
                profit           = net_proceeds - (self._shares_held * self._entry_price)
                self._balance    += net_proceeds
                self._total_profit += profit
                reward           = profit
                self._shares_held = 0.0
                self._entry_price = 0.0
                self._trade_history.append(("SELL", self._current_step, price, profit))

        # HOLD cezası: uzun bekleyen ajan hafifçe cezalandırılır
        if action == 0 and self._shares_held > 0:
            reward -= 0.01  # küçük negatif ödül

        # Adımı ilerlet
        self._current_step += 1
        max_steps = len(self._features) - self.window_size - 1
        terminated = self._current_step >= max_steps
        truncated  = False

        # Bölüm sonu ödülü: portföy büyümesi
        if terminated and self._shares_held > 0:
            gross          = self._shares_held * price
            commission_fee = gross * self.commission
            net_proceeds   = gross - commission_fee
            profit         = net_proceeds - (self._shares_held * self._entry_price)
            reward        += profit
            self._balance += net_proceeds
            self._total_profit += profit
            self._shares_held = 0.0

        reward *= self.reward_scaling

        info = {
            "step":           self._current_step,
            "price":          price,
            "balance":        self._balance,
            "shares_held":    self._shares_held,
            "portfolio_value": self._portfolio_value(),
            "total_profit":   self._total_profit,
        }

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, terminated, truncated, info

    def render(self):
        pv = self._portfolio_value()
        print(
            f"Step: {self._current_step:4d} | "
            f"Price: ${self._current_price():.2f} | "
            f"Balance: ${self._balance:.2f} | "
            f"Shares: {self._shares_held:.4f} | "
            f"Portfolio: ${pv:.2f} | "
            f"Profit: ${self._total_profit:.2f}"
        )

    def get_trade_history(self):
        return self._trade_history

    def get_total_return(self) -> float:
        """Toplam getiri yüzdesi."""
        return (self._portfolio_value() / self.initial_balance - 1) * 100
