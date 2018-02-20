import numpy as np
import pandas as pd

FILENAME = 'clean-data.csv'


def clean():

    df = pd.read_csv(FILENAME)

    df = df.drop('Date', axis=1)

    df = (df - df.min()) / (df.max() - df.min())

    df.to_csv('normalised-nodate.csv', index=False)


def get_data():
    return pd.read_csv('normalised-nodate.csv')


def clean_and_read():
    clean()
    return get_data()

