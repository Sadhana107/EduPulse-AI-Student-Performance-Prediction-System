import pandas as pd


def load_data():

    df = pd.read_csv(
        "data/students.csv"
    )

    return df