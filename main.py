import pickle
import pandas as pd

def main(df):
    tfidf = pickle.load(open('tfidf', 'rb'))
    clf = pickle.load(open('clf_task1', 'rb'))

    X_test = tfidf.transform(df.item_name)

    pred = clf.predict(X_test)

    res = pd.DataFrame(pred, columns=['pred'])
    res['id'] = df['id']

    res[['id', 'pred']].to_csv('answers.csv', index=None)
    return res[['id', 'pred']]