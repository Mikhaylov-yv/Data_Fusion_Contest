import pandas as pd
import shutil
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def loading_data(path):
    df = pd.read_parquet(path)
    return df

# Разделение на тренировочные и тестовые данные
def separation_data(df):
    train = df[df.category_id != -1]
    test = df[df.category_id == -1]
    return train, test

def add_ed_izm(df):
    reg_dict = {'gram': '\d{1,5}.{0,1}г',
                'kg': '\d{1,5}.{0,1}кг',
                'litr': '\d{1,3}[.|,]{0,1}\d{1,3}.{0,1}л'}
    df = df[['item_name']].drop_duplicates()
    df.item_name = df.item_name.str.lower()
    for typ in reg_dict.keys():
        ser_filtr = df.item_name.str.contains(reg_dict[typ], na=False)
        df.loc[ser_filtr, 'ed_izm'] = typ
        col = df.item_name.str.extract(
            f"({reg_dict[typ]})").loc[:, 0]
        df.loc[ser_filtr, 'col'] = col
        df.loc[ser_filtr, 'item_name'] = df.loc[ser_filtr,
                                                'item_name'].replace(regex={
                                                reg_dict[typ]: ''})
    df.col = df.col.str.replace(' ', '', regex=True)
    df.col = df.col.str.replace(',', '.', regex=True)
    df.col = df.col.str.replace('[^0-9,]', '', regex=True)
    df.col = pd.to_numeric(df.col)
    # # Концеритруем все в близкое к кг
    for ed_izm in ['gram']:
        df.loc[df.ed_izm == ed_izm, 'col'
                ] = df.loc[df.ed_izm == ed_izm, 'col'] / 1000
    return df


# Сохранение готово к отправке zip архива
def save_output_zip(name):
    for fill in ['script.py', 'tfidf', 'clf_task1']:
        shutil.copy(f"{ROOT_DIR}/{fill}", f"{ROOT_DIR}/data/output/{fill}")
    shutil.make_archive(f"{ROOT_DIR}/data/output/{name}", 'zip', f"{ROOT_DIR}/data/output")