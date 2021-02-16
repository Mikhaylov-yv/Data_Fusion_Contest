import pandas as pd
import os
import function as fn
import script_function as sfn
from sklearn.feature_extraction.text import CountVectorizer

pd.options.display.max_rows = 300

os.chdir('../')

def test_count_words():
    # Тестируем качество
    df = pd.read_parquet('tests/test_data_fusion_train.parquet')
    cv = fn.get_cv()
    cv.fit_transform(df.item_name)
    count_in = len(cv.vocabulary_)
    df_transform = df[['item_name']]
    df = sfn.data_preparation(df)
    df_transform['item_name_new'] = df['item_name']
    cv = fn.get_cv()
    cv.fit_transform(df.item_name)
    count_out = len(cv.vocabulary_)
    print(f"""\nБез обработки найденоо: {count_in} слов
После обработки: {count_out} слов""")
    print(df_transform)
    print(pd.Series(cv.vocabulary_).sort_values(ascending=False))
