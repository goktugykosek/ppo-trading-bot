"""
main.py
-------
PPO Trading Bot — Ana çalıştırma scripti.

Kullanım:
    python main.py                        # Varsayılan ayarlarla çalıştır
    python main.py --ticker MSFT          # Farklı hisse
    python main.py --timesteps 1000000   # Daha uzun eğitim
    python main.py --mode backtest        # Sadece backtest (model var ise)
    python main.py --ticker THYAO.IS      # Borsa İstanbul hissesi
"""

import argparse
import os
from data_fetcher import fetch_stock_data, add_technical_indicators, prepare_train_test, save_data
from trainer import train, load_model, build_vec_env, make_env
from backtester import run_backtest, compute_metrics, print_metrics, plot_backtest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Trading Bot")
    parser.add_argument("--ticker",     type=str,   default="AAPL",
                        help="Hisse kodu (örn. AAPL, MSFT, THYAO.IS)")
    parser.add_argument("--start",      type=str,   default="2018-01-01",
                        help="Başlangıç tarihi")
    parser.add_argument("--end",        type=str,   default="2024-01-01",
                        help="Bitiş tarihi")
    parser.add_argument("--timesteps",  type=int,   default=300_000,
                        help="Toplam eğitim adımı (daha fazla = daha iyi ama yavaş)")
    parser.add_argument("--window",     type=int,   default=20,
                        help="Gözlem pencere boyutu")
    parser.add_argument("--n-envs",     type=int,   default=4,
                        help="Paralel ortam sayısı")
    parser.add_argument("--balance",    type=float, default=10_000.0,
                        help="Başlangıç sermayesi ($)")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Komisyon oranı (0.001 = %%0.1)")
    parser.add_argument("--mode",       type=str,   default="train",
                        choices=["train", "backtest", "both"],
                        help="train: sadece eğit, backtest: sadece test, both: ikisi")
    parser.add_argument("--model-path", type=str,   default=None,
                        help="Yüklenecek model yolu (backtest modu için)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "🤖 " + "="*50)
    print("   PPO ALGORİTMİK TRADING BOTU")
    print("   " + "="*50)
    print(f"   Ticker  : {args.ticker}")
    print(f"   Dönem   : {args.start} → {args.end}")
    print(f"   Mod     : {args.mode}")
    print("🤖 " + "="*50 + "\n")


    raw_df = fetch_stock_data(args.ticker, args.start, args.end)
    df     = add_technical_indicators(raw_df)
    save_data(df, f"data/{args.ticker}_processed.csv")

    train_df, test_df = prepare_train_test(df, train_ratio=0.8)

    env_kwargs = dict(
        initial_balance = args.balance,
        commission      = args.commission,
    )

  
    model = None

    if args.mode in ("train", "both"):
        model = train(
            train_df        = train_df,
            val_df          = test_df,
            window_size     = args.window,
            total_timesteps = args.timesteps,
            n_envs          = args.n_envs,
            **env_kwargs,
        )

   
    if args.mode in ("backtest", "both"):
        if model is None:
            model_path = args.model_path or "models/best/best_model.zip"
            if not os.path.exists(model_path):
                model_path = "models/ppo_trading_final.zip"
            print(f"[Main] Model yükleniyor: {model_path}")

            dummy_env = DummyVecEnv([make_env(test_df, args.window, **env_kwargs)])
            model = PPO.load(model_path, env=dummy_env)

        print("\n[Main] Backtest başlatılıyor...")
        result  = run_backtest(model, test_df, args.window, **env_kwargs)
        metrics = compute_metrics(result)
        print_metrics(metrics, ticker=args.ticker)

        plot_backtest(
            result, metrics,
            ticker    = args.ticker,
            save_path = f"results/{args.ticker}_backtest.png",
        )

    print("\n✅ İşlem tamamlandı!")
    if args.mode in ("train", "both"):
        print("   TensorBoard için: tensorboard --logdir logs/")
    print()


if __name__ == "__main__":
    main()
