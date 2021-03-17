import pickle
import pandas as pd
import torch
from model import TextClassificationModel


def main(df, test = False):
    # Load model
    # tokenizer, model, cat_dict
    cat_dict = pickle.load(open('cat_dict', 'rb'))
    # Edit data
    # df = sfn.data_preparation(df)
    # df = fn.add_ed_izm(df)
    # Predict
    pred = df.item_name.apply(predict).map(dict(map(reversed, cat_dict.items())))

    # generation report
    if ~test:
        df['pred'] = pred
        return df['pred']
    else:
        res = pd.DataFrame(pred, columns=['pred'])
        res['id'] = df['id']

        res[['id', 'pred']].to_csv('answers.csv', index=None)

def predict(text):
    with torch.no_grad():
        tokenizer = pickle.load(open('tokenizer', 'rb'))
        vocab = pickle.load(open('vocab', 'rb'))
        text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
        device = torch.device('cpu')
        model_param = pickle.load(open('model_param', 'rb'))
        model = TextClassificationModel(model_param['vocab_size'],
                                        model_param['embed_dim'],
                                        model_param['num_class'])
        model.load_state_dict(torch.load('model', map_location=device))
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()