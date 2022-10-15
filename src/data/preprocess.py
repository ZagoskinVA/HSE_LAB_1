import src.config as cfg
import pandas as pd
import numpy as np
import src.utils



def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    if idx_col in df.columns:
        df = df.set_index(idx_col)
    return df


def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[cfg.SEX_COL].value_counts().index[0]
    df[cfg.SEX_COL] = df[cfg.SEX_COL].fillna(most_freq)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')

    ohe_int_cols = df[cfg.OHE_COLS].select_dtypes('number').columns
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)

    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df

def replace_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in cfg.NAN_COLS:
        df[col] = df[col].fillna(0)
    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.pipe(set_idx, cfg.ID_COL).pipe(drop_unnecesary_id).pipe(fill_sex).pipe(replace_nan_values).pipe(cast_types)

def process_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int8)
    return df