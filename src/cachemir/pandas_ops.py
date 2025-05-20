import numpy.typing as npt
import pandas as pd


def df2dct(df: pd.DataFrame) -> dict[str, npt.NDArray]:
    return {col: df[col].to_numpy() for col in df}
