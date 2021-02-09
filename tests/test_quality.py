import function as fn
from main import main
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from datetime import datetime
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = None

os.chdir('../')

def test_script_quality():
    df = pd.read_parquet('data/input/data_fusion_train.parquet')
    df = df[df.category_id != -1]
    df['id'] = df['receipt_id']
    df = df.reset_index()
    pred = main(df).pred
    df['pred'] = pred
    # Проверка сходимости
    assert pred.shape[0] == df.index.shape[0]
    # print(df.columns)
    # print(pred)
    # print(df['category_id'])
    # print(df[['category_id','pred']])
    score = metrics.f1_score(df.category_id, df.pred, average='weighted')
    print(f"Точность ответов: {round(score * 100, 2)}%")
    seve_data_to_report(score)

def seve_data_to_report(score, test = False):
    encoding = '1251'
    path = 'reports/Результаты тестов модели.csv'
    df = pd.read_csv(path, encoding=encoding, sep= ';', index_col = 'Дата')
    df.loc[pd.to_datetime(datetime.today().strftime('%d/%m/%Y %H:%M')), 'Результат'] = str(score).replace('.', ',')
    if ~test:
        df.to_csv(path, sep= ';', encoding=encoding)
    return df


def test_seve_data_to_report():
    print(seve_data_to_report(0.228, test = True))
