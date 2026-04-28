"""
trainer.py
----------
PPO ajanını eğiten modül.

Stable-Baselines3'ün PPO implementasyonu kullanılır.
TensorBoard logları 'logs/' klasörüne yazılır.
Eğitilen model 'models/' altına kaydedilir.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import pandas as pd

from trading_env import TradingEnv



class TradingMetricsCallback(BaseCallback):
    """Her episode bitişinde terminal'e özet basar."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_profits = []

    def _on_step(self) -> bool:
      
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    profit = info.get("total_profit", 0)
                    self.episode_profits.append(profit)
                    if self.verbose >= 1 and len(self.episode_profits) % 50 == 0:
                        avg = np.mean(self.episode_profits[-50:])
                        print(
                            f"  [Episode {len(self.episode_profits):4d}] "
                            f"Son kâr: ${profit:.2f} | "
                            f"Son 50 ortalama: ${avg:.2f}"
                        )
        return True


def make_env(df: pd.DataFrame, window_size: int, **env_kwargs):
    """Monitor ile sarılmış env factory fonksiyonu."""
    def _init():
        env = TradingEnv(df, window_size=window_size, **env_kwargs)
        env = Monitor(env)
        return env
    return _init


def build_vec_env(df: pd.DataFrame, window_size: int, n_envs: int = 4, **env_kwargs):
    """
    Paralel ortam (VecEnv) oluşturur.
    n_envs > 1 → PPO daha verimli öğrenir.
    """
    env = DummyVecEnv([make_env(df, window_size, **env_kwargs) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env



def build_ppo_model(env, log_dir: str = "logs/") -> PPO:
    """
    PPO hiper-parametreleri.
    
    n_steps      : Her env için kaç adımda bir güncelleme yapılır
    batch_size   : Mini-batch boyutu (n_steps * n_envs'in böleni olmalı)
    n_epochs     : Her güncelleme turunda kaç epoch
    gamma        : İndirim faktörü (uzun vadeli ödülü ne kadar önemser)
    gae_lambda   : GAE smoothing (bias/variance trade-off)
    clip_range   : PPO clipping (politikanın ne kadar değişebileceği)
    ent_coef     : Entropi bonusu (keşfi teşvik eder)
    learning_rate: Öğrenme hızı
    policy       : MlpPolicy = tam bağlantılı sinir ağı
    """
    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        n_steps         = 2048,
        batch_size      = 256,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        learning_rate   = 3e-4,
        tensorboard_log = log_dir,
        verbose         = 0,
        policy_kwargs   = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
    )
    return model




def train(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    window_size:     int   = 20,
    total_timesteps: int   = 500_000,
    n_envs:          int   = 4,
    model_dir:       str   = "models/",
    log_dir:         str   = "logs/",
    save_freq:       int   = 50_000,
    **env_kwargs,
) -> PPO:
    """
    PPO modelini eğitir.
    
    Args:
        train_df        : Eğitim verisi
        val_df          : Validation verisi (EvalCallback için)
        window_size     : Ortam pencere boyutu
        total_timesteps : Toplam eğitim adımı
        n_envs          : Paralel ortam sayısı
        model_dir       : Model kayıt klasörü
        log_dir         : TensorBoard log klasörü
        save_freq       : Kaç adımda bir checkpoint kaydedilsin
    
    Returns:
        Eğitilmiş PPO modeli
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    print("\n" + "="*60)
    print("  PPO TRADING BOTU EĞİTİMİ BAŞLIYOR")
    print("="*60)
    print(f"  Toplam adım   : {total_timesteps:,}")
    print(f"  Paralel env   : {n_envs}")
    print(f"  Pencere boyutu: {window_size}")
    print("="*60 + "\n")

    train_env = build_vec_env(train_df, window_size, n_envs, **env_kwargs)

    val_env_raw = DummyVecEnv([make_env(val_df, window_size, **env_kwargs)])
    val_env = VecNormalize(val_env_raw, norm_obs=True, norm_reward=False,
                           training=False, clip_obs=10.0)

    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path = os.path.join(model_dir, "best/"),
        log_path             = log_dir,
        eval_freq            = max(10_000 // n_envs, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq = max(save_freq // n_envs, 1),
        save_path = os.path.join(model_dir, "checkpoints/"),
        name_prefix = "ppo_trading",
    )

    metrics_callback = TradingMetricsCallback(verbose=1)

    model = build_ppo_model(train_env, log_dir)

    model.learn(
        total_timesteps = total_timesteps,
        callback        = [eval_callback, checkpoint_callback, metrics_callback],
        tb_log_name     = "PPO_Trading",
        progress_bar    = True,
    )

    final_path = os.path.join(model_dir, "ppo_trading_final")
    model.save(final_path)
    train_env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"\n[Trainer] Model kaydedildi: {final_path}")

    return model


def load_model(model_path: str, env, normalize_path: str = None) -> PPO:
    """Kaydedilmiş modeli yükler."""
    model = PPO.load(model_path, env=env)
    print(f"[Trainer] Model yüklendi: {model_path}")
    return model
