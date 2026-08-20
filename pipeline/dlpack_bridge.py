import cupy as cp
import cudf


def cudf_series_to_cupy(series: cudf.Series) -> cp.ndarray:
    return cp.from_dlpack(series.to_dlpack())


def cudf_df_to_cupy_matrix(df: cudf.DataFrame) -> cp.ndarray:
    return cp.from_dlpack(df.to_dlpack())


def cupy_to_cudf_series(arr: cp.ndarray, name: str) -> cudf.Series:
    return cudf.Series(arr, name=name)


def cupy_matrix_to_cudf_df(arr: cp.ndarray, columns: list) -> cudf.DataFrame:
    return cudf.DataFrame(arr, columns=columns)


def assert_device_resident(obj) -> None:
    if isinstance(obj, cudf.DataFrame) or isinstance(obj, cudf.Series):
        return
    if isinstance(obj, cp.ndarray):
        return
    if hasattr(obj, "__cuda_array_interface__"):
        return
    raise TypeError(f"object of type {type(obj)} is not GPU-resident")


def zero_copy_bridge(obj, target: str):
    assert_device_resident(obj)

    if target == "cupy":
        if isinstance(obj, cp.ndarray):
            return obj
        return cp.from_dlpack(obj.to_dlpack())

    if target == "cudf_series":
        if isinstance(obj, cudf.Series):
            return obj
        return cudf.Series(obj)

    if target == "cudf_df":
        if isinstance(obj, cudf.DataFrame):
            return obj
        return cudf.DataFrame(obj)

    raise ValueError(f"unknown target: {target}")