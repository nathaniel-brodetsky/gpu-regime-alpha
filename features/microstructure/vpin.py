import cudf
import cupy as cp


def compute_volume_buckets(df: cudf.DataFrame, bucket_volume: float) -> cudf.DataFrame:
    trade_size_arr = df["trade_size"].to_cupy()
    order_flow_sign_arr = df["order_flow_sign"].to_cupy()

    cumulative_volume = cp.cumsum(trade_size_arr)
    bucket_id = cp.floor(cumulative_volume / bucket_volume).astype(cp.int64)

    buy_volume = cp.where(order_flow_sign_arr > 0, trade_size_arr, 0.0)
    sell_volume = cp.where(order_flow_sign_arr < 0, trade_size_arr, 0.0)

    result = cudf.DataFrame({
        "bucket_id": bucket_id,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "trade_size": trade_size_arr,
    })

    return result


def compute_vpin(df: cudf.DataFrame, bucket_volume: float, n_buckets_window: int) -> cudf.DataFrame:
    bucketed = compute_volume_buckets(df, bucket_volume)

    grouped = bucketed.groupby("bucket_id").agg({
        "buy_volume": "sum",
        "sell_volume": "sum",
        "trade_size": "sum",
    }).reset_index()

    grouped = grouped.sort_values("bucket_id").reset_index(drop=True)

    imbalance = (grouped["buy_volume"] - grouped["sell_volume"]).abs()
    total_volume = grouped["trade_size"]

    rolling_imbalance = imbalance.rolling(window=n_buckets_window, min_periods=1).sum()
    rolling_volume = total_volume.rolling(window=n_buckets_window, min_periods=1).sum()

    vpin = rolling_imbalance / rolling_volume

    grouped["vpin"] = vpin
    return grouped[["bucket_id", "vpin"]]