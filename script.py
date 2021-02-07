import pandas as pd
import pickle


test = pd.read_parquet('data/task1_test_for_user.parquet')

tfidf = pickle.load(open('tfidf', 'rb'))
clf = pickle.load(open('clf_task1', 'rb'))

X_test = tfidf.transform(test.item_name)

pred = clf.predict(X_test)

res = pd.DataFrame(pred, columns=['pred'])
res['id'] = test['id']

res[['id', 'pred']].to_csv('answers.csv', index=None)