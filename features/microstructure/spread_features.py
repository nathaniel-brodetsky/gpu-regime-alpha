import cudf


def compute_spread(df: cudf.DataFrame) -> cudf.Series:
    return df["best_ask"] - df["best_bid"]


def compute_relative_spread(df: cudf.DataFrame) -> cudf.Series:
    spread = compute_spread(df)
    mid = (df["best_bid"] + df["best_ask"]) / 2
    return spread / mid


def compute_micro_price(df: cudf.DataFrame) -> cudf.Series:
    total_size = df["bid_size"] + df["ask_size"]
    weighted_bid = df["best_bid"] * df["ask_size"]
    weighted_ask = df["best_ask"] * df["bid_size"]
    return (weighted_bid + weighted_ask) / total_size


def compute_rolling_micro_price_variance(df: cudf.DataFrame, window: int) -> cudf.Series:
    micro_price = compute_micro_price(df)
    return micro_price.rolling(window=window, min_periods=1).var()


def compute_rolling_spread_mean(df: cudf.DataFrame, window: int) -> cudf.Series:
    spread = compute_spread(df)
    return spread.rolling(window=window, min_periods=1).mean()