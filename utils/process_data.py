import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, XLNetTokenizer
import csv
from utils import config


class PairDS(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        s1, s2, l = self.pairs[index]
        if config.JIONT:
            s_t = self.tokenizer.encode(s1, s2, pad_to_max_length=True, max_length=config.sentence_max_len)
            s_t = torch.tensor(s_t).cuda()
            label = torch.tensor([l], dtype=torch.float).cuda()
            return s_t, label
        else:
            # s1_t = self.tokenizer.encode_plus(s1, pad_to_max_length=True, max_length=config.sentence_max_len)
            # s2_t = self.tokenizer.encode_plus(s2, pad_to_max_length=True, max_length=config.sentence_max_len)
            s1_t = self.tokenizer.encode(s1, pad_to_max_length=True, max_length=config.sentence_max_len)
            s2_t = self.tokenizer.encode(s2, pad_to_max_length=True, max_length=config.sentence_max_len)

            s1_t = torch.tensor(s1_t).cuda()
            s2_t = torch.tensor(s2_t).cuda()
            label = torch.tensor([l], dtype=torch.float).cuda()
            return s1_t, s2_t, label

    def __len__(self):
        return len(self.pairs)


def read_corp(path):
    ste_pairs = []
    csv_reader = csv.reader(open(path, encoding='utf-8'))
    # 去掉头
    next(csv_reader)
    for row in csv_reader:
        ste1, ste2, label = row[1], row[2], row[3]
        ste_pairs.append((ste1, ste2, int(label)))
    return ste_pairs


def get_dataloader(tokenizer, joint=False):
    p_pairs = read_corp('F:/workcode/FAQ/data/subwayQq_positive_label.csv')
    n_pairs = read_corp('F:/workcode/FAQ/data/subwayQq_negative_label.csv')
    all_paris = p_pairs + n_pairs
    train_pairs = []
    test_pairs = []
    idx = 0
    for p in all_paris:
        if idx % 5 == 4:
            test_pairs.append(p)
        else:
            train_pairs.append(p)
        idx += 1

    train_ds = PairDS(train_pairs, tokenizer)
    test_ds = PairDS(test_pairs, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)
    return train_loader, test_loader


def get_bert_dataloader():
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)
    return get_dataloader(bert_tokenizer)


def get_xlnet_dataloader():
    xlnet_tokenizer = XLNetTokenizer.from_pretrained(config.XLNet_model_path)
    return get_dataloader(xlnet_tokenizer)
