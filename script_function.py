# Функции которые работают для обучения и для предсказания


import pandas as df

def data_preparation(df):
    df.item_name = df.item_name.str.lower()
    df.item_name = df.item_name.str.replace('[^a-zа-я0-9,.]', ' ', regex=True)
    return df