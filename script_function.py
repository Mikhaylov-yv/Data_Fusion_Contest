# Функции которые работают для обучения и для предсказания


import pandas as df

def data_preparation(df):
    df.item_name = df.item_name.str.lower()
    return df