import numpy.typing as npt
import pandas as pd


def pandas2dct(x: pd.DataFrame | pd.Series) -> dict[str, npt.NDArray]:
    if isinstance(x, pd.Series):
        return {x.name if x.name is not None else "no_name": x.to_numpy()}
    if isinstance(x, pd.DataFrame):
        return {col: x[col].to_numpy() for col in x}
    return x
