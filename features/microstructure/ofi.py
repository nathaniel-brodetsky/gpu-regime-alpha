import cudf
import cupy as cp


def compute_ofi(df: cudf.DataFrame) -> cudf.Series:
    bid_price_delta = df["best_bid"].diff().fillna(0)
    ask_price_delta = df["best_ask"].diff().fillna(0)

    bid_size_delta = df["bid_size"].diff().fillna(0)
    ask_size_delta = df["ask_size"].diff().fillna(0)

    bid_contribution = cudf.Series(
        cp.where(
            bid_price_delta.to_cupy() > 0,
            df["bid_size"].to_cupy(),
            cp.where(
                bid_price_delta.to_cupy() == 0,
                bid_size_delta.to_cupy(),
                0.0,
            ),
        )
    )

    ask_contribution = cudf.Series(
        cp.where(
            ask_price_delta.to_cupy() < 0,
            df["ask_size"].to_cupy(),
            cp.where(
                ask_price_delta.to_cupy() == 0,
                ask_size_delta.to_cupy(),
                0.0,
            ),
        )
    )

    ofi = bid_contribution - ask_contribution
    return ofi


def compute_rolling_ofi(df: cudf.DataFrame, window: int) -> cudf.Series:
    ofi = compute_ofi(df)
    return ofi.rolling(window=window, min_periods=1).sum()