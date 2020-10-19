from distance.sentence_simi import SentenceSimilarity
import csv


def bm25_cls(s1, s2, sd, th=1.1):
    d = sd.bm25(s1, s2)
    l = 0
    if d > th:
        l = 1
    return l


def read_corp(path):
    ste_pairs = []
    csv_reader = csv.reader(open(path, encoding='utf-8'))
    # 去掉头
    next(csv_reader)
    for row in csv_reader:
        ste1, ste2, label = row[1], row[2], row[3]
        ste_pairs.append((ste1, ste2, int(label)))
    return ste_pairs


def vaild_bm25(all_paris, th=0.96):
    p_right = 0
    p_sum = 0
    r_sum = 0
    for s1, s2, l in all_paris:
        pdt = bm25_cls(s1, s2, sd, th)
        if pdt == 1 and l == 1:
            p_right += 1
        if pdt == 1:
            p_sum += 1
        if l == 1:
            r_sum += 1
    a, b = p_right / p_sum, p_right / r_sum
    f1 = 2 * a * b / (a + b)
    return f1


def do_qa(q, stand_docs, sd):
    stand_docs = list(stand_docs)
    ds = [(d, sd.bm25(q, d)) for d in stand_docs]
    ds = sorted(ds, key=lambda x: x[1], reverse=True)
    return ds[:10]


if __name__ == '__main__':
    p_pairs = read_corp('F:/workcode/FAQ/data/subwayQq_positive_label.csv')
    n_pairs = read_corp('F:/workcode/FAQ/data/subwayQq_negative_label.csv')

    docs = []
    stand_docs = []
    for d in p_pairs:
        docs.append(d[0])
        docs.append(d[1])
        stand_docs.append(d[1])
    for d in n_pairs:
        docs.append(d[0])
        docs.append(d[1])
        stand_docs.append(d[1])

    sd = SentenceSimilarity(set(docs))

    f1 = vaild_bm25(p_pairs + n_pairs)
    print(f1)

    while True:
        print('请输入问题：')
        q = input()
        rs = do_qa(q, set(stand_docs), sd)
        print(rs)
