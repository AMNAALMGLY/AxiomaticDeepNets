import torch, torchtext
from torchtext.legacy import data
from torchtext.legacy import datasets
import spacy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_IMDB_loaders(embedding, vocab_size, split_ratio, batch_size):
    TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)

    # make splits for data
    print('hi')
    train, test = datasets.IMDB.splits(TEXT, LABEL, )
    print(len(train), len(test))
    traindata, valid = train.split(split_ratio=split_ratio)
    # build the vocabulary
    TEXT.build_vocab(traindata, unk_init=torch.normal, max_size=vocab_size, vectors=embedding)
    LABEL.build_vocab(traindata)

    trainLoader, validLoader, testLoader = data.BucketIterator.splits(datasets=(traindata, valid, test),
                                                                      batch_size=batch_size,
                                                                      sort_key=lambda x: len(x.text),
                                                                      sort_within_batch=True, device=device)
    return trainLoader, validLoader, testLoader, TEXT, LABEL


def preprocess(sent, textField):
    sent = sent.lower()
    tokenize = spacy.load('en_core_web_sm')
    tokens = [tok.text for tok in tokenize.tokenizer(sent)]

    onehot = [textField.vocab.stoi[tok] for tok in tokens]
    onehot = torch.LongTensor(onehot)
    return onehot, tokens
