"""
backtester.py
-------------
Eğitilmiş modeli test verisi üzerinde değerlendirir.

Hesaplanan metrikler:
    - Toplam getiri (%)
    - Buy & Hold getirisi (benchmark)
    - Sharpe Ratio
    - Max Drawdown
    - Kazanma oranı (winning trade %)
    - Toplam işlem sayısı

Çıktı:
    - Terminal özeti
    - Grafik (portfolio değeri, al/sat sinyalleri, aksiyon dağılımı)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # GUI olmadan kaydet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trading_env import TradingEnv
import os



def run_backtest(
    model:      PPO,
    test_df:    pd.DataFrame,
    window_size: int  = 20,
    initial_balance: float = 10_000.0,
    commission: float = 0.001,
    render_every: int = 0,   # 0 → hiç render etme
) -> dict:
    """
    Modeli deterministic modda test verisinde çalıştırır.
    
    Returns:
        Metrikler ve trade geçmişi içeren sözlük.
    """
    env = TradingEnv(
        test_df,
        window_size     = window_size,
        initial_balance = initial_balance,
        commission      = commission,
    )

    obs, _ = env.reset()
    done   = False
    step   = 0

    prices       = []
    portfolio    = []
    actions_list = []
    balances     = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        prices.append(info["price"])
        portfolio.append(info["portfolio_value"])
        balances.append(info["balance"])
        actions_list.append(int(action))

        if render_every > 0 and step % render_every == 0:
            env.render()

        step += 1

    trade_history = env.get_trade_history()
    final_value   = env.get_total_return()

    return {
        "prices":        np.array(prices),
        "portfolio":     np.array(portfolio),
        "balances":      np.array(balances),
        "actions":       np.array(actions_list),
        "trade_history": trade_history,
        "total_return":  final_value,
        "final_portfolio": portfolio[-1] if portfolio else initial_balance,
        "initial_balance": initial_balance,
    }




def compute_metrics(result: dict) -> dict:
    """Backtest sonucundan finansal metrikleri hesaplar."""
    portfolio = result["portfolio"]
    prices    = result["prices"]
    actions   = result["actions"]

    # Günlük getiriler
    daily_returns = np.diff(portfolio) / portfolio[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    # Sharpe Ratio (yıllık, risk-free rate = 0)
    if daily_returns.std() > 1e-9:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    rolling_max = np.maximum.accumulate(portfolio)
    drawdowns   = (portfolio - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100

    bh_return = (prices[-1] / prices[0] - 1) * 100

    trades        = result["trade_history"]
    sell_trades   = [t for t in trades if t[0] == "SELL"]
    n_trades      = len(sell_trades)
    winning_trades = [t for t in sell_trades if len(t) > 3 and t[3] > 0]
    win_rate      = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0

    action_counts = {
        "HOLD": int((actions == 0).sum()),
        "BUY":  int((actions == 1).sum()),
        "SELL": int((actions == 2).sum()),
    }

    metrics = {
        "total_return_pct":   result["total_return"],
        "bh_return_pct":      bh_return,
        "sharpe_ratio":       sharpe,
        "max_drawdown_pct":   max_drawdown,
        "n_trades":           n_trades,
        "win_rate_pct":       win_rate,
        "action_distribution": action_counts,
        "final_portfolio_usd": result["final_portfolio"],
        "initial_balance_usd": result["initial_balance"],
    }
    return metrics


def print_metrics(metrics: dict, ticker: str = ""):
    """Metrikleri güzel formatla terminal'e basar."""
    sep = "="*55
    title = f"  BACKTEST SONUÇLARI{f' — {ticker}' if ticker else ''}"
    print(f"\n{sep}\n{title}\n{sep}")
    print(f"  Başlangıç sermayesi : ${metrics['initial_balance_usd']:>10,.2f}")
    print(f"  Final portföy       : ${metrics['final_portfolio_usd']:>10,.2f}")
    print(f"  {'─'*45}")
    print(f"  Toplam getiri (Bot) : %{metrics['total_return_pct']:>+9.2f}")
    print(f"  Buy & Hold getirisi : %{metrics['bh_return_pct']:>+9.2f}")
    print(f"  {'─'*45}")
    print(f"  Sharpe Ratio        : {metrics['sharpe_ratio']:>10.3f}")
    print(f"  Max Drawdown        : %{metrics['max_drawdown_pct']:>+9.2f}")
    print(f"  {'─'*45}")
    print(f"  Toplam işlem        : {metrics['n_trades']:>10}")
    print(f"  Kazanma oranı       : %{metrics['win_rate_pct']:>9.1f}")
    print(f"  {'─'*45}")
    ad = metrics['action_distribution']
    total_steps = sum(ad.values())
    for k, v in ad.items():
        print(f"  {k:<20}: {v:>6} (%{v/total_steps*100:.1f})")
    print(f"{sep}\n")



def plot_backtest(
    result:  dict,
    metrics: dict,
    ticker:  str = "Stock",
    save_path: str = "results/backtest.png",
):
    """4 panelli backtest grafiği oluşturur ve kaydeder."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    prices   = result["prices"]
    portfolio = result["portfolio"]
    actions  = result["actions"]
    trades   = result["trade_history"]
    steps    = np.arange(len(prices))

    fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
    fig.suptitle(
        f"PPO Trading Bot — {ticker}",
        fontsize=18, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
    ax_price    = fig.add_subplot(gs[0, :])   # Tam genişlik
    ax_port     = fig.add_subplot(gs[1, :])   # Tam genişlik
    ax_action   = fig.add_subplot(gs[2, 0])
    ax_metrics  = fig.add_subplot(gs[2, 1])

    DARK_BG  = "#0d1117"
    GRID_COL = "#21262d"
    TEXT_COL = "#e6edf3"

    for ax in [ax_price, ax_port, ax_action, ax_metrics]:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        ax.spines[:].set_color(GRID_COL)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(TEXT_COL)

    # --- Panel 1: Fiyat + Al/Sat sinyalleri ---
    ax_price.plot(steps, prices, color="#58a6ff", linewidth=1.2, label="Fiyat", zorder=2)

    buys  = [(t[1], t[2]) for t in trades if t[0] == "BUY"]
    sells = [(t[1], t[2]) for t in trades if t[0] == "SELL"]

    if buys:
        bx, by = zip(*buys)
        ax_price.scatter(bx, by, marker="^", color="#3fb950", s=80,
                         zorder=5, label=f"BUY ({len(buys)})")
    if sells:
        sx, sy = zip(*sells)
        ax_price.scatter(sx, sy, marker="v", color="#f85149", s=80,
                         zorder=5, label=f"SELL ({len(sells)})")

    ax_price.set_title("Fiyat & İşlem Sinyalleri", color=TEXT_COL, fontsize=11)
    ax_price.set_ylabel("Fiyat ($)", color=TEXT_COL)
    ax_price.legend(facecolor="#161b22", labelcolor=TEXT_COL, fontsize=9)
    ax_price.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)

    # --- Panel 2: Portföy değeri vs Buy&Hold ---
    bh = result["initial_balance"] * (prices / prices[0])
    ax_port.plot(steps, portfolio, color="#79c0ff", linewidth=1.5,
                 label=f"PPO Bot (%{metrics['total_return_pct']:+.1f})", zorder=3)
    ax_port.plot(steps, bh, color="#f0883e", linewidth=1.2, linestyle="--",
                 label=f"Buy & Hold (%{metrics['bh_return_pct']:+.1f})", zorder=2)
    ax_port.fill_between(steps, portfolio, bh,
                         where=(portfolio > bh), alpha=0.15, color="#3fb950")
    ax_port.fill_between(steps, portfolio, bh,
                         where=(portfolio < bh), alpha=0.15, color="#f85149")
    ax_port.set_title("Portföy Değeri vs Buy & Hold", color=TEXT_COL, fontsize=11)
    ax_port.set_ylabel("Değer ($)", color=TEXT_COL)
    ax_port.legend(facecolor="#161b22", labelcolor=TEXT_COL, fontsize=9)
    ax_port.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)

    # --- Panel 3: Aksiyon dağılımı ---
    ad     = metrics["action_distribution"]
    labels = list(ad.keys())
    vals   = list(ad.values())
    colors = ["#58a6ff", "#3fb950", "#f85149"]
    bars   = ax_action.bar(labels, vals, color=colors, edgecolor=DARK_BG, linewidth=0.5)
    ax_action.set_title("Aksiyon Dağılımı", color=TEXT_COL, fontsize=11)
    ax_action.set_ylabel("Adım sayısı", color=TEXT_COL)
    for bar, val in zip(bars, vals):
        ax_action.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val), ha="center", va="bottom",
            color=TEXT_COL, fontsize=9
        )
    ax_action.grid(axis="y", color=GRID_COL, linestyle="--", alpha=0.5)

    # --- Panel 4: Metrik tablosu ---
    ax_metrics.axis("off")
    rows = [
        ["Toplam Getiri",  f"%{metrics['total_return_pct']:+.2f}"],
        ["Buy&Hold",       f"%{metrics['bh_return_pct']:+.2f}"],
        ["Sharpe Ratio",   f"{metrics['sharpe_ratio']:.3f}"],
        ["Max Drawdown",   f"%{metrics['max_drawdown_pct']:.2f}"],
        ["İşlem Sayısı",   str(metrics['n_trades'])],
        ["Kazanma Oranı",  f"%{metrics['win_rate_pct']:.1f}"],
    ]
    tbl = ax_metrics.table(
        cellText    = rows,
        colLabels   = ["Metrik", "Değer"],
        loc         = "center",
        cellLoc     = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if r == 0 else DARK_BG)
        cell.set_text_props(color=TEXT_COL)
        cell.set_edgecolor(GRID_COL)
    ax_metrics.set_title("Performans Özeti", color=TEXT_COL, fontsize=11)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    print(f"[Backtester] Grafik kaydedildi: {save_path}")
    plt.close()
