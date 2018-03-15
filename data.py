import numpy as np
import pandas as pd

FILENAME = 'clean-data.csv'


def clean():

    df = pd.read_csv(FILENAME)

    df = df.dropna()

    # df['month'] = pd.DatetimeIndex(df['Date']).month  # Convert all dates into months
    df = df.drop('Date', axis=1)  # Drop the date column all together

    df = df[df.applymap(np.isreal).all(1)]  # Remove all rows with non-numeric data

    df = df[(df['T'] <= 57) & df['T'] >= -43]  # Remove all data whose temp is above/below records in Cali
    df = df[df['W'] <= 199]  # Remove all data whose wind speeds are above/below records in Cali

    mins_maxes = (df.min(), df.max())

    df = (df - df.min()) / (df.max() - df.min())  # Normalise every row to value between [0, 1]

    return np.array(df), mins_maxes


def get_data():
    pass


def clean_and_read():
    return clean()
