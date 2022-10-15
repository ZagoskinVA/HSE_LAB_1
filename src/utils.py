import pandas as pd
from datetime import datetime, time as datetime_time, timedelta
import src.config as cfg

def time_diff(start: datetime, end: datetime):
    if isinstance(start, datetime_time): # convert to datetime
        assert isinstance(end, datetime_time)
        start, end = [datetime.combine(datetime.min, t) for t in [start, end]]
    if start <= end: # e.g., 10:33:26-11:15:49
        return end - start
    else: # end < start e.g., 23:55:00-00:25:00
        end += timedelta(1) # +day
        assert end > start
        return end - start

def compute_sleep_time(df: pd.DataFrame) -> pd.DataFrame:
    return time_diff(datetime.strptime(df[cfg.SLEEP_TIME_COL], '%H:%M:%S'), datetime.strptime(df[cfg.WAKE_UP_TIME_COL], '%H:%M:%S')).seconds//3600

def save_as_pickle(df: pd.DataFrame, path: str) -> None:
    df.to_pickle(path)

def extract_target(df: pd.DataFrame):
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]
    return df, target