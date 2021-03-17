from torch import nn
import pickle

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        print(f"Данные для модели \n vocab_size, embed_dim, num_class: {vocab_size, embed_dim, num_class}")
        pickle.dump({'vocab_size' : vocab_size,
                     'embed_dim' : embed_dim,
                     'num_class' : num_class}, open('model_param', 'wb'))
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