import function as fn
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = None

# def test_loading_data():
#     df = fn.loading_data('../data/input/data_fusion_train.parquet')
#     is_pandas = type(df) is pd.core.frame.DataFrame
#     print(is_pandas)
#     assert is_pandas

# def test_save_output_zip():
#     fn.save_output_zip('clf_task1')

def test_add_ed_izm():
    df = pd.DataFrame(
        {'item_name':
             ['хлеб на сыворотке 350г',
              'напиток энерг. ред булл 0,355л',
              'пиво светлое "халзан" 4,5 % об, пл/б. 1,5 л(шт)',
              'конфеты харитоша 1кг мол. ваф яшкино',
              'хеменгуэй дайкири']
         }
    )
    test_df = fn.add_ed_izm(df)
    test_df['item_name_in'] = df.item_name
    print(test_df)
