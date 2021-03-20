import pytest
import function as fn
import lern
from main import main
import pandas as pd
import os
from to_wrap_up import warp_data
import pickle
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
    predict = main(df, True)
    df['predict'] = predict
    print(df)



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

def test_add_ed_izm_full_data():
    data_path = 'data/input/data_fusion_train.parquet'
    test_df = fn.loading_data(data_path, pickle.load(open('cat_dict', 'rb')))
    test_df = test_df[test_df.category_id != -1].drop_duplicates(subset=['item_name', 'category_id'])
    test_df['item_name_new'] = fn.add_ed_izm(test_df).item_name
    print(test_df)

def test_lern_main():
    lern.main(path, test = True)

def test_out_data():
    warp_data('task1_test_for_user.parquet')
    os.chdir('data/output/Model_to_send')
    import script
    df_in = pd.read_parquet('data/task1_test_for_user.parquet')
    df_in['pred'] = pd.read_csv('answers.csv')['pred']
    print(df_in[['item_name', 'category_id', 'pred']])


