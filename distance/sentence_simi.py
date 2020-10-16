import jieba
import math
from simhash import Simhash
from collections import defaultdict

import transformers
import torch
import numpy as np

from distance.WMD_base import word_rotator_distance, word_mover_distance


class SentenceSimilarity(object):
    def __init__(self, docs, MODEL_PATH='F:/workcode/FAQ/pretrain_model/bert-chinese-base/'):
        self.docs = docs
        self.docs_len = len(docs)
        self.word_in_docs_count = defaultdict(lambda: 0)
        self.cal_word_fred()
        self.stop_words = set(c for c in '~!@#$%^&*():,./;`')
        self.init_BERT(MODEL_PATH)

    def init_BERT(self, model_path):
        self.tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
        self.model = transformers.BertModel.from_pretrained(model_path)
        self.model.eval()  # 将模型设为验证模式

    def cal_word_fred(self):
        len_sum = 0
        for d in self.docs:
            ls = self.split_word(d)
            len_sum += len(ls)
            set_w = set(ls)
            for w in set_w:
                self.word_in_docs_count[w] += 1
        self.avgdl = len_sum / self.docs_len

    def split_word(self, ste):
        # ls=jieba.lcut(d)
        ls = [c for c in ste]
        return ls

    def cal_IDF(self, word):
        '''值越大这个词越重要'''
        idf = math.log((self.docs_len - self.word_in_docs_count[word] + 0.5) / (self.word_in_docs_count[word] + 0.5))
        return idf

    def cal_R(self, word, doc, k1=2, b=0.75):
        '''值越大word在doc中越重要'''
        K = k1 * (1 - b + b * len(doc) / self.avgdl)

        ls = self.split_word(doc)
        f1 = ls.count(word)

        r = f1 * (k1 + 1) / (f1 + K)
        return r

    def bm25(self, q, doc):
        '''越高句子越相似'''
        ls = self.split_word(q)
        score = 0
        for w in ls:
            idf = self.cal_IDF(w)
            r = self.cal_R(w, doc)
            temp = idf * r
            score += temp
        return score / len(q)

    def jaccard(self, q, doc):
        '''jaccard相似度'''
        s1, s2 = set(q), set(doc)
        ret1 = s1.intersection(s2)  # 交集
        ret2 = s1.union(s2)  # 并集
        sim = 1.0 * len(ret1) / len(ret2)
        return sim

    def sim_hash(self, q, doc):
        '''距离越小越相似'''
        q = self.fomrat_str(q)
        doc = self.fomrat_str(doc)
        s1, s2 = Simhash(q), Simhash(doc)
        return s1.distance(s2)

    def get_ste_ebd(self, ste):
        '''获得stence中每个词的编码向量，包含了[cls]xxxx[sep]xxxx[sep]符号'''
        sen_code = self.tokenizer.encode_plus(ste)
        input_ids = torch.tensor([sen_code['input_ids']])
        token_type_ids = torch.tensor([sen_code['token_type_ids']])
        ebds, max_pool = self.model(input_ids, token_type_ids=token_type_ids)
        return ebds.detach().numpy()[0], max_pool.detach().numpy()[0]

    def cos_simi_v(self, v1, v2):
        '''值越小越相似'''
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        rs = (v1 * v2).sum() / (v1_norm * v2_norm)
        return rs

    def cos_simi(self, ste1, ste2):
        '''值越小越相似'''
        ebd1, _ = self.get_ste_ebd(ste1)
        edd2, _ = self.get_ste_ebd(ste2)
        return self.cos_simi_v(ebd1.mean(axis=0), edd2.mean(axis=0))

    def wmd(self, s1, s2):
        '''0-正无穷 0最相似'''
        ebd1, _ = self.get_ste_ebd(s1)
        edd2, _ = self.get_ste_ebd(s2)
        d = word_mover_distance(ebd1, edd2)
        return d

    def wrd(self, s1, s2):
        '''0-2 0最相似'''
        ebd1, _ = self.get_ste_ebd(s1)
        edd2, _ = self.get_ste_ebd(s2)
        d = word_rotator_distance(ebd1, edd2)
        return d

    def fomrat_str(self, s):
        ls = [c for c in s if c not in self.stop_words]
        return ''.join(ls)
