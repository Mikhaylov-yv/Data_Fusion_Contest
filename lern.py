import function as fn
import script_function as sfn
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from scipy.sparse import csr_matrix
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
import torch
from torch import nn
import time


def main(train, test = False):
    # Загрузка данных
    train_df, cat_dict = fn.loading_data(train)
    train_data = train_df[['category_id_new', 'item_name']].to_numpy()

    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for label, line in train_data:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=1)

    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    label_pipeline = lambda x: int(x)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    dataloader = DataLoader(train_data, batch_size=8, shuffle=False, collate_fn=collate_batch)

    class TextClassificationModel(nn.Module):

        def __init__(self, vocab_size, embed_dim, num_class):
            super(TextClassificationModel, self).__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
            self.fc = nn.Linear(embed_dim, num_class)
            self.init_weights()

        def init_weights(self):
            initrange = 0.5
            self.embedding.weight.data.uniform_(-initrange, initrange)
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()

        def forward(self, text, offsets):
            embedded = self.embedding(text, offsets)
            return self.fc(embedded)

    num_class = len(cat_dict)
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()
        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predited_label = model(text, offsets)
            #         print(sorted(label))
            loss = criterion(predited_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                  total_acc / total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predited_label = model(text, offsets)
                loss = criterion(predited_label, label)
                total_acc += (predited_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    from torch.utils.data.dataset import random_split
    # Hyperparameters
    EPOCHS = 10  # epoch
    LR = 5  # скорость обучения
    BATCH_SIZE = 16  # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    # train_iter, test_iter = AG_NEWS()
    train_dataset = list(train_data)
    num_train = int(len(train_dataset) * 0.9)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    # print(train_dataset)
    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)


    # # Сохранение тестовых данных
    # if ~test:
    #     pickle.dump(train_data.item_name, open('X_test', 'wb'))
    #     pickle.dump(train_data.category_id_new, open('y_test', 'wb'))



    # Сохранение моделей
    # tfidf = cv
    # if ~test:
    #     pickle.dump(tfidf, open('tfidf', 'wb'))
    #     pickle.dump(clf, open('clf_task1', 'wb'))


if __name__ == '__main__':
    main('data/input/data_fusion_train.parquet')
