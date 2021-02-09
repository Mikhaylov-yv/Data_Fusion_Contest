import function as fn
import lern
from main import main
import pandas as pd
import os
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = None

os.chdir('../')

# Проверка работоспособности основного отправляемого файла
def test_main():
    df = pd.read_parquet('tests/test_data_fusion_train.parquet')
    df['id'] = df.receipt_id
    main(df)

# Проверка функции загрузки данных
def test_loading_data():
    df = fn.loading_data('tests/test_data_fusion_train.parquet')
    is_pandas = type(df) is pd.core.frame.DataFrame
    print(is_pandas)
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


def test_lern_main():
    pikl_data = lern.main(pd.read_parquet('tests/test_data_fusion_train.parquet'))
    print(len(pikl_data))
    lern.seve_model(pikl_data[0], pikl_data[1])