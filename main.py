import pickle
import pandas as pd

import script_function as sfn

def main(df, test = False):
    # Load model
    tfidf = pickle.load(open('tfidf', 'rb'))
    clf = pickle.load(open('clf_task1', 'rb'))
    # Edit data
    df = sfn.data_preparation(df)
    X_test = tfidf.transform(df.item_name)
    # Predict
    pred = clf.predict(X_test)
    # generation report
    if test:
        df['pred'] = pred
        return df['pred']
    res = pd.DataFrame(pred, columns=['pred'])
    res['id'] = df['id']

    res[['id', 'pred']].to_csv('answers.csv', index=None)