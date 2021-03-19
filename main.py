import pickle
import pandas as pd
import torch
from model import TextClassificationModel


def main(df, test = False):
    # Load model
    # tokenizer, model, cat_dict
    # cat_dict = pickle.load(open('cat_dict', 'rb'))
    # Edit data
    # df = sfn.data_preparation(df)
    # df = fn.add_ed_izm(df)
    # Predict
    # pred = df.item_name.apply(predict).map(dict(map(reversed, cat_dict.items())))
    pred = predict(df.item_name)

    # generation report
    if test:
        df['pred'] = pred
        return df['pred']
    else:
        print('Сохранение данных')
        res = pd.DataFrame(pred, columns=['pred'])
        res['id'] = df['id']

        res[['id', 'pred']].to_csv('answers.csv', index=None)

def predict(text_list):
    with torch.no_grad():
        text_list = text_list.values
        # print(text_list)
        tokenizer = pickle.load(open('tokenizer', 'rb'))
        vocab = pickle.load(open('vocab', 'rb'))
        cat_dict = pickle.load(open('cat_dict', 'rb'))
        text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
        device = torch.device('cpu')
        model_param = pickle.load(open('model_param', 'rb'))
        model = TextClassificationModel(model_param['vocab_size'],
                                        model_param['embed_dim'],
                                        model_param['num_class'])
        print('+' * 50)
        model.load_state_dict(torch.load('model', map_location=device))
        output = []
        for text in text_list:
            text_tensor = torch.tensor(text_pipeline(text))
            # Если строка пустая
            if len(text_tensor) == 0:
                print('Нулевой')
                text_tensor = torch.tensor([0])
            pred = model(text_tensor, torch.tensor([0])).argmax(1).item()
            pred = dict(map(reversed, cat_dict.items()))[pred]
            output.append(pred)
        return output