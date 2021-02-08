import pickle
import pandas as pd

def main(df):
    # Load model
    tfidf = pickle.load(open('tfidf', 'rb'))
    clf = pickle.load(open('clf_task1', 'rb'))
    # Edit data
    X_test = tfidf.transform(df.item_name)
    # Predict
    pred = clf.predict(X_test)
    # generation report
    res = pd.DataFrame(pred, columns=['pred'])
    res['id'] = df['id']

    res[['id', 'pred']].to_csv('answers.csv', index=None)
    return res[['id', 'pred']]