import cudf
import cupy as cp


def generate_synthetic_lob(n_ticks: int, seed: int = 42) -> cudf.DataFrame:
    rng = cp.random.default_rng(seed)

    dt = cp.full(n_ticks, 1e-3, dtype=cp.float64)
    timestamps = cp.cumsum(dt)

    mid_price_drift = rng.normal(0, 0.0001, n_ticks)
    mid_price = 100.0 + cp.cumsum(mid_price_drift)

    spread = cp.abs(rng.normal(0.02, 0.005, n_ticks)) + 1e-4
    best_bid = mid_price - spread / 2
    best_ask = mid_price + spread / 2

    bid_size = cp.abs(rng.normal(500, 150, n_ticks)) + 1.0
    ask_size = cp.abs(rng.normal(500, 150, n_ticks)) + 1.0

    order_flow_sign = rng.choice(cp.array([-1.0, 1.0]), size=n_ticks)
    trade_size = cp.abs(rng.normal(100, 40, n_ticks)) + 1.0
    trade_price = cp.where(order_flow_sign > 0, best_ask, best_bid)

    regime_shock = rng.integers(0, n_ticks, size=n_ticks // 5000)
    volatility_multiplier = cp.ones(n_ticks)
    for shock_idx in regime_shock.tolist():
        window_end = min(shock_idx + 2000, n_ticks)
        volatility_multiplier[shock_idx:window_end] *= 4.0

    mid_price = mid_price * volatility_multiplier / volatility_multiplier[0]

    df = cudf.DataFrame({
        "timestamp": timestamps,
        "mid_price": mid_price,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "trade_price": trade_price,
        "trade_size": trade_size,
        "order_flow_sign": order_flow_sign,
    })

    return df


def generate_multi_regime_lob(n_ticks_per_regime: int, n_regimes: int, seed: int = 42) -> cudf.DataFrame:
    frames = []
    for regime_idx in range(n_regimes):
        regime_df = generate_synthetic_lob(n_ticks_per_regime, seed=seed + regime_idx)
        regime_df["true_regime_label"] = regime_idx
        frames.append(regime_df)

    combined = cudf.concat(frames, ignore_index=True)
    combined["timestamp"] = cp.cumsum(cp.full(len(combined), 1e-3, dtype=cp.float64))
    return combined