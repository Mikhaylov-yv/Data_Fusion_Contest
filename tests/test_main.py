import pytest
import function as fn
import lern
from main import main
import pandas as pd
import os
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = None

os.chdir('../')
path = 'tests/test_data_fusion_train.parquet'
# Проверка работоспособности основного отправляемого файла

@pytest.fixture()
def df():
    df_, cat_dict = fn.loading_data(path)
    return df_

def test_main(df):
    df['id'] = df.receipt_id
    main(df)

# Проверка функции загрузки данных
def test_loading_data(df):
    is_pandas = type(df) is pd.core.frame.DataFrame
    print(is_pandas)
    print(df.columns)
    assert is_pandas

# def test_save_output_zip():
#     fn.save_output_zip('clf_task1')

# Проверка добавления столбца единиц измерения
def test_add_ed_izm():
    df = pd.DataFrame(
        {'item_name':
             ['хлеб на сыворотке 350г',
              'напиток энерг. ред булл 0,355л',
              'пиво светлое "халзан" 4,5 % об, пл/б. 1,5 л(шт)',
              'конфеты харитоша 1кг мол. ваф яшкино',
              'сок яблочный, 250 мл,шт',
              'з/п smokers 75 мл',
              '!амбробене сироп 15мг/5мл 100мл фл',
              'хеменгуэй дайкири']
         }
    )
    test_df = fn.add_ed_izm(df)
    test_df['item_name_in'] = df.item_name
    print(test_df)

def test_get_cv(df):
    cv = fn.get_cv(df.item_name)
    print(cv)
    print(pd.Series(cv.vocabulary_).sort_values())

def test_lern_main():
    lern.main(path, test = True)
