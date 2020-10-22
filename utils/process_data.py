import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, XLNetTokenizer
import csv
from utils import config
from distance.sentence_simi import SentenceSimilarity


class PairDS(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

        stand_docs = list(set([d[1] for d in pairs]))
        self.sd = SentenceSimilarity(stand_docs)

    def __getitem__(self, index):
        str1, str2, l = self.pairs[index]
        if config.JIONT:
            s_t = self.tokenizer.encode(str1, str2, pad_to_max_length=True, max_length=config.SENTENCE_MAX_LEN)
            s_t = torch.tensor(s_t).cuda()
            label = torch.tensor([l], dtype=torch.float).cuda()
            return s_t, label
        else:
            # s1_t = self.tokenizer.encode_plus(s1, pad_to_max_length=True, max_length=config.SENTENCE_MAX_LEN)
            # s2_t = self.tokenizer.encode_plus(s2, pad_to_max_length=True, max_length=config.SENTENCE_MAX_LEN)
            s1 = self.tokenizer.encode_plus(str1, pad_to_max_length=True, max_length=config.SENTENCE_MAX_LEN)
            s2 = self.tokenizer.encode_plus(str2, pad_to_max_length=True, max_length=config.SENTENCE_MAX_LEN)

            s1_t = torch.tensor(s1['input_ids'], device=config.DEVICE)
            s2_t = torch.tensor(s2['input_ids'], device=config.DEVICE)
            s1_mask = torch.tensor(s1['attention_mask'], device=config.DEVICE)
            s2_mask = torch.tensor(s2['attention_mask'], device=config.DEVICE)
            s1_idf = self.build_idf_weight(str1)
            s2_idf = self.build_idf_weight(str2)
            l = torch.tensor([l], dtype=torch.float).cuda()

            # s1_t = torch.tensor(s1_t).cuda()
            # s2_t = torch.tensor(s2_t).cuda()
            # label = torch.tensor([l], dtype=torch.float).cuda()
            return s1_t, s2_t, s1_mask, s2_mask, s1_idf, s2_idf, l

    def build_idf_weight(self, ste):
        ste = ste[:49]
        s_idf = [self.sd.cal_IDF(s) for s in ste]
        s_idf.insert(0, 0)
        pad_len = config.SENTENCE_MAX_LEN - len(s_idf)
        idf_w = s_idf + [0] * pad_len
        idf_t = torch.tensor(idf_w, device=config.DEVICE)
        return torch.nn.functional.softmax(idf_t, dim=0)

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
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    return train_loader, test_loader


def get_bert_dataloader():
    bert_tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_PATH)
    return get_dataloader(bert_tokenizer)


def get_xlnet_dataloader():
    xlnet_tokenizer = XLNetTokenizer.from_pretrained(config.XLNET_MODEL_PATH)
    return get_dataloader(xlnet_tokenizer)
