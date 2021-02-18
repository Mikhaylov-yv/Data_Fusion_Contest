import function as fn
import script_function as sfn
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from scipy.sparse import csr_matrix

def main(train, test = False):
    df = train[train.category_id != -1]
    df = df.drop_duplicates(subset=['item_name'])
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
    # Состаления словаря для перевода векторы
    cv = fn.get_cv(X_train['item_name'])
    # Перевод слов векторы
    cv_fit = csr_matrix(cv.transform(X_train['item_name']))
    # Постороение модели
    clf = fn.get_model(cv_fit, y_train)
    # Обучение модели
    clf.fit(cv_fit, y_train)
    # Сохранение моделей
    tfidf = cv
    if ~test:
        pickle.dump(tfidf, open('tfidf', 'wb'))
        pickle.dump(clf, open('clf_task1', 'wb'))
    return


if __name__ == '__main__':
    main(pd.read_parquet('data/input/data_fusion_train.parquet'))
