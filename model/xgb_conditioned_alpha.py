import cudf
import cupy as cp
import xgboost as xgb


def compute_forward_return(df: cudf.DataFrame, horizon: int, price_col: str = "mid_price") -> cudf.Series:
    future_price = df[price_col].shift(-horizon)
    current_price = df[price_col]
    forward_return = (future_price - current_price) / current_price
    return forward_return


def build_training_frame(clustered_features: cudf.DataFrame, raw_df: cudf.DataFrame, horizon: int, feature_columns: list) -> cudf.DataFrame:
    aligned_raw = raw_df.iloc[-len(clustered_features):].reset_index(drop=True)

    target = compute_forward_return(aligned_raw, horizon)

    training_frame = clustered_features[feature_columns + ["cluster_id"]].copy()
    training_frame["target"] = target.reset_index(drop=True)

    training_frame = training_frame.dropna()
    return training_frame


def walk_forward_split(training_frame: cudf.DataFrame, train_fraction: float = 0.7) -> tuple:
    n = len(training_frame)
    split_idx = int(n * train_fraction)

    train = training_frame.iloc[:split_idx].reset_index(drop=True)
    test = training_frame.iloc[split_idx:].reset_index(drop=True)

    return train, test


def train_conditioned_xgb(train_frame: cudf.DataFrame, feature_columns: list, num_boost_round: int = 200, max_depth: int = 6, learning_rate: float = 0.05) -> xgb.Booster:
    all_features = feature_columns + ["cluster_id"]

    train_frame = train_frame.copy()
    train_frame["cluster_id"] = train_frame["cluster_id"].astype("category")

    dtrain = xgb.DMatrix(train_frame[all_features], label=train_frame["target"], enable_categorical=True)

    params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": max_depth,
        "eta": learning_rate,
        "objective": "reg:squarederror",
    }

    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return booster


def predict_alpha(booster: xgb.Booster, test_frame: cudf.DataFrame, feature_columns: list) -> cp.ndarray:
    all_features = feature_columns + ["cluster_id"]

    test_frame = test_frame.copy()
    test_frame["cluster_id"] = test_frame["cluster_id"].astype("category")

    dtest = xgb.DMatrix(test_frame[all_features], enable_categorical=True)
    predictions = booster.predict(dtest)
    return cp.asarray(predictions)