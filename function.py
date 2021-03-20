import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def loading_data(path, cat_dict = {}):
    df = pd.read_parquet(path)[[
        'category_id', 'item_name']]
    if cat_dict == {}:
        cat_num = 0
        for category_id in df.category_id.drop_duplicates():
            cat_dict[category_id] = cat_num
            df.loc[df.category_id == category_id, 'category_id_new'] = cat_num
            cat_num += 1
        return df, cat_dict
    else:
        df['category_id_new'] = df.category_id.map(cat_dict)
        return df

# Разделение на тренировочные и тестовые данные
def separation_data(df):
    train = df[df.category_id != -1]
    test = df[df.category_id == -1]
    return train, test

# Выделение единиц измерения с помощью регулярного выражения
def add_ed_izm(df):
    reg_dict = {'gram': '\d{1,5}.{0,1}г',
                'kg': '\d{1,5}.{0,1}кг',
                'litr': '\d{1,3}[.|,]{0,1}\d{1,3}.{0,1}л',
                'mlitr' : '\d{1,4}.{0,2}мл'}
    df = df[['item_name']]#.drop_duplicates()
    df.item_name = df.item_name.str.lower()
    for typ in reg_dict.keys():
        ser_filtr = df.item_name.str.contains(reg_dict[typ], na=False)
        df.loc[ser_filtr, 'ed_izm'] = typ
        # col = df.item_name.str.extract(
        #     f"({reg_dict[typ]})").loc[:, 0]
        # df.loc[ser_filtr, 'col'] = col
        df.loc[ser_filtr, 'item_name'] = df.loc[ser_filtr,
                                                'item_name'].replace(regex={
                                                reg_dict[typ]: ''})
    # df.col = df.col.str.replace(' ', '', regex=True)
    # df.col = df.col.str.replace(',', '.', regex=True)
    # df.col = df.col.str.replace('[^0-9,]', '', regex=True)
    # df.col = pd.to_numeric(df.col)
    # # Концеритруем все в близкое к кг
    # for ed_izm in ['gram', 'mlitr']:
    #     df.loc[df.ed_izm == ed_izm, 'col'
    #             ] = df.loc[df.ed_izm == ed_izm, 'col'] / 1000
    df.item_name = df.item_name + ' ' + df.ed_izm.fillna('')
    # df.col = df.col.fillna(0)
    return df