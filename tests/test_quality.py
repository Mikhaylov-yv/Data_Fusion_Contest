import function as fn
from main import main
import pandas as pd
from pandas.testing import assert_series_equal
import numpy as np
import pickle
import os
from sklearn import metrics
from datetime import datetime
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = None

os.chdir('../')

def test_script_quality():
    X_test = pickle.load(open('X_test', 'rb'))
    X_test['id'] = X_test['receipt_id']
    y_test = pickle.load(open('y_test', 'rb'))
    # y_test = y_test.reset_index()
    # Проверка индексов в тестовых данных
    assert_series_equal(pd.Series(X_test.index), pd.Series(y_test.index))
    pred = main(X_test, test=True)
    assert y_test.shape[0], pred.shape[0]
    assert_series_equal(pd.Series(pred.index), pd.Series(y_test.index))
    df_out = X_test
    assert df_out.shape[0], pred.shape[0]
    df_out['pred'] = pred
    df_out['category_id'] = y_test
    # Проверка сходимости
    # print(df_out.columns)
    # print(pred)
    # print(df_out['category_id'])
    # print(df_out[['category_id','pred']])
    assert pred.shape[0] == df_out.index.shape[0]
    score = metrics.f1_score(df_out.category_id, df_out.pred, average='weighted')
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


# def test_seve_data_to_report():
#     print(seve_data_to_report(0.228, test = True))
