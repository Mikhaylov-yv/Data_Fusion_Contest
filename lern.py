import function as fn
import script_function as sfn
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from scipy.sparse import csr_matrix

def main(train, test = False):
    df = train[train.category_id != -1]
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('category_id', 1), df.category_id, test_size=0.2, random_state=42)
    # Сохранение тестовых данных
    if ~test:
        pickle.dump(X_test, open('X_test', 'wb'))
        pickle.dump(y_test, open('y_test', 'wb'))
    # Предварительная обработка текста
    # X_train = sfn.data_preparation(X_train)
    # Выделение дополнительных данных
    X_train = fn.add_ed_izm(X_train)

    # Выделение item_name с индексами category_id
    train_item_name_ser = X_train['item_name']
    train_item_name_ser.index = y_train
    train_item_name_ser = train_item_name_ser.drop_duplicates()

    # Состаления словаря для перевода векторы
    cv = fn.get_cv(train_item_name_ser)
    # Перевод слов векторы
    cv_fit = csr_matrix(cv.transform(train_item_name_ser))
    # Постороение модели
    clf = fn.get_model(cv_fit, train_item_name_ser.index)
    # Обучение модели
    clf.fit(cv_fit, train_item_name_ser.index)
    # Сохранение моделей
    tfidf = cv
    if ~test:
        pickle.dump(tfidf, open('tfidf', 'wb'))
        pickle.dump(clf, open('clf_task1', 'wb'))


if __name__ == '__main__':
    main(pd.read_parquet('data/input/data_fusion_train.parquet'))
