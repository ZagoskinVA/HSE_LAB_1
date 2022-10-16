import src.config as cfg
import pandas as pd
import numpy as np
from src.utils import compute_sleep_time

def add_ord_edu(df: pd.DataFrame) -> pd.DataFrame:
    df[f'{cfg.EDU_COL}_ord'] = df[cfg.EDU_COL].str.slice(0, 1).astype(np.int8).values
    return df

def add_sleep_time(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.SLEEP_COL] = df[[cfg.SLEEP_TIME_COL, cfg.WAKE_UP_TIME_COL]].apply(lambda x: compute_sleep_time(x), axis=1)
    return df

def add_features(df:pd.DataFrame) -> pd.DataFrame:
    return df.pipe(add_ord_edu).pipe(add_sleep_time).pipe(split_gender)



def split_gender(df: pd.DataFrame) -> pd.DataFrame:
    if(cfg.SEX_COL in df.columns):
        df[cfg.IS_MAN_COL] = df[cfg.SEX_COL].apply(lambda x: 1 if x == 'М' else 0)
        df[cfg.IS_WOMEN_COL] = df[cfg.SEX_COL].apply(lambda x: 1 if x == 'Ж' else 0)
        df.drop(cfg.SEX_COL, axis=1)
    return df