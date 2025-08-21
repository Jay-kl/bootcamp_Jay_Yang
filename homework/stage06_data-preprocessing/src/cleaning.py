"""Utility functions for common data cleaning tasks.

This module provides helpers to fill missing values, drop rows with missing
data under different rules, and normalize numeric columns. Behavior is kept
simple and explicit to avoid surprises.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore[reportMissingImports]

__all__ = [
    'fill_missing_median',
    'drop_missing',
    'normalize_data',
    'fill_missing_general',
]


def _select_numeric_columns(df: pd.DataFrame, columns: Optional[Sequence[str]]) -> list[str]:
    """Return numeric column names from ``df`` or the provided ``columns`` as a list."""
    if columns is None:
        return list(df.select_dtypes(include=np.number).columns)
    return list(columns)

def fill_missing_median(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Fill missing values in numeric columns with the column median.

    - If ``columns`` is None, all numeric columns are targeted.
    - Otherwise, only the specified columns are processed.
    """

    df_copy = df.copy()
    columns_to_fill = _select_numeric_columns(df_copy, columns)
    if columns_to_fill:
        df_copy[columns_to_fill] = df_copy[columns_to_fill].apply(lambda s: s.fillna(s.median()))
    return df_copy


def drop_missing(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Drop rows with missing values according to the provided rule.

    - If ``columns`` is provided, drop rows with NA in any of these columns.
    - Else if ``threshold`` is provided, keep rows that have at least
      ``int(threshold * n_columns)`` non-NA values.
    - Else, drop any rows with at least one NA across all columns.
    """

    df_copy = df.copy()
    if columns is not None:
        return df_copy.dropna(subset=columns)
    if threshold is not None:
        required_non_na = int(threshold * df_copy.shape[1])
        return df_copy.dropna(thresh=required_non_na)
    return df_copy.dropna()


def normalize_data(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: str = 'minmax',
) -> pd.DataFrame:
    """Normalize selected columns using a scaling ``method``.

    - If ``columns`` is None, all numeric columns are scaled.
    - ``method`` must be 'minmax' or 'standard'.
    """

    df_copy = df.copy()
    columns_to_scale = _select_numeric_columns(df_copy, columns)

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("method must be 'minmax' or 'standard'")

    df_copy[columns_to_scale] = scaler.fit_transform(df_copy[columns_to_scale])
    return df_copy

def fill_missing_general(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values by column type.

    - Numeric columns: fill with median of each column
    - Object (categorical) columns: fill with 'unknown'
    - Datetime columns: forward then backward fill
    """

    df_copy = df.copy()

    num_cols = df_copy.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        df_copy[num_cols] = df_copy[num_cols].apply(lambda s: s.fillna(s.median()))

    cat_cols = df_copy.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        df_copy[cat_cols] = df_copy[cat_cols].fillna('unknown')

    dt_cols = df_copy.select_dtypes(include='datetime64[ns]').columns
    if len(dt_cols) > 0:
        df_copy[dt_cols] = df_copy[dt_cols].ffill().bfill()

    return df_copy


