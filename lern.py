import function as fn
import script_function as sfn
import pandas as pd

import pickle


def main(train):
    train = train[train.category_id != -1]

    # Предварительная обработка текста
    train = sfn.data_preparation(train)

    # Выделение item_name с индексами category_id
    train_item_name_ser = train['item_name']
    train_item_name_ser.index = train['category_id']
    train_item_name_ser = train_item_name_ser.drop_duplicates()

    # Состаления словаря для перевода векторы
    cv = fn.get_cv(train_item_name_ser)
    # Перевод слов векторы
    cv_fit = cv.fit_transform(train_item_name_ser)
    # Постороение модели
    clf = fn.get_model(cv_fit, train_item_name_ser.index)
    # Обучение модели
    clf.fit(cv_fit, train_item_name_ser.index)
    # Сохранение моделей
    return cv, clf

def seve_model(tfidf, clf):
    pickle.dump(tfidf, open('tfidf', 'wb'))
    pickle.dump(clf, open('clf_task1', 'wb'))




if __name__ == '__main__':
    pickle_data = main(pd.read_parquet('data/input/data_fusion_train.parquet'))
    seve_model(pickle_data[0], pickle_data[1])