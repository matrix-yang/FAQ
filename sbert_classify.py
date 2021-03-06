import torch
from transformers import BertModel
import torch.optim as optim
from utils import config
from utils.process_data import get_bert_dataloader
from utils.model_selection import ModelManager


class SBertCls(torch.nn.Module):
    def __init__(self):
        super(SBertCls, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_PATH)
        self.encoder = torch.nn.TransformerEncoderLayer(config.EMBEDDING_DIM, 8)
        self.liner = torch.nn.Sequential(
            torch.nn.BatchNorm1d(config.EMBEDDING_DIM * 3),
            torch.nn.Dropout(),
            torch.nn.Linear(config.EMBEDDING_DIM * 3, config.EMBEDDING_DIM),
            torch.nn.BatchNorm1d(config.EMBEDDING_DIM),
            torch.nn.Dropout(),
            torch.nn.Linear(config.EMBEDDING_DIM, 1),
            torch.nn.Sigmoid()
        )

        # 句子的注意力变量(1*50),各个子的ebd
        self.weight1 = torch.nn.Linear(1, config.SENTENCE_MAX_LEN, bias=False)
        self.weight2 = torch.nn.Linear(1, config.SENTENCE_MAX_LEN, bias=False)

    def cal_ste_ebd(self, ebd1, ebd2):
        batch, len_, dim = ebd1.shape
        one = torch.ones((batch, 1, 1), device=config.DEVICE)
        atte1 = self.weight1(one)
        atte2 = self.weight2(one)
        ste_ebd1 = torch.bmm(atte1, ebd1).view(batch, config.EMBEDDING_DIM)
        ste_ebd2 = torch.bmm(atte2, ebd2).view(batch, config.EMBEDDING_DIM)
        return ste_ebd1, ste_ebd2

    def masked_max_pooling(self, states, masks):
        # batch_size, seq_len, hidden_dim
        m = masks.unsqueeze(2)
        # Set masked units to lower bound
        min_val = torch.min(states)
        preserv_val = states * m
        lower_bound = min_val * (1 - m)
        last_hidden_states = preserv_val + lower_bound
        # Max pooling, masked units will not be chosen
        pooled = torch.max(last_hidden_states, axis=1)[0]
        return pooled

    def masked_avg_pooling(self, states, masks):
        # batch_size,1
        batch_len = masks.sum(dim=1).unsqueeze(dim=1)

        # batch_size, seq_len, hidden_dim
        m = masks.unsqueeze(2)

        # Set masked units to zero
        pad_zero = states * m
        len_sum = pad_zero.sum(dim=1)

        # mean
        avg_pool = len_sum / batch_len
        return avg_pool

    def forward(self, ste1, ste2, mask1, mask2, idf1, idf2):
        ebd1, _ = self.bert(ste1)
        ebd2, _ = self.bert(ste2)

        # 使用max_pooling
        max_pool1 = self.masked_max_pooling(ebd1, mask1)
        max_pool2 = self.masked_max_pooling(ebd2, mask2)

        # 使用avg_pooling
        # max_pool1 = self.masked_avg_pooling(ebd1, mask1)
        # max_pool2 = self.masked_avg_pooling(ebd2, mask2)
        # 分类输出
        contact = torch.cat((max_pool1, max_pool2, torch.abs(max_pool1 - max_pool2)), dim=1)
        out = self.liner(contact)
        # 回归输出
        cos = torch.nn.functional.cosine_similarity(max_pool1, max_pool2, dim=1)
        # 使用atten
        # ste_ebd1, ste_ebd2 = self.cal_ste_ebd(ebd1, ebd2)
        # contact = torch.cat((ste_ebd1, ste_ebd2), dim=1)
        # out = self.liner(contact)

        # 使用idf加权
        # ste1_ebd = idf1.unsqueeze(2).mul(ebd1).sum(dim=1)
        # ste2_ebd = idf2.unsqueeze(2).mul(ebd2).sum(dim=1)
        # contact = torch.cat((ste1_ebd, ste2_ebd), dim=1)
        # out = self.liner(contact)
        cos.unsqueeze_(dim=1)
        return out, cos


def freeze_parameter(cls_model):
    for n, p in cls_model.named_parameters():
        if 'bert' in n:
            p.requires_grad = False
    for n, p in cls_model.named_parameters():
        if 'bert.encoder.layer.11' in n:
            p.requires_grad = True


def model_forward(td, model):
    s1_t, s2_t, s1_mask, s2_mask, s1_idf, s2_idf, l = td
    y, cos = model(s1_t, s2_t, s1_mask, s2_mask, s1_idf, s2_idf)
    return y, cos, l


def train(model, train_data, test_data, epoch=30):
    # 损失函数
    classify_loss_fn = torch.nn.BCELoss()
    regression_loss_fn = torch.nn.MSELoss()
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.5)
    # 模型保存/提前终止
    model_manager = ModelManager(model, name='sbert_cls_reg')
    loss_sum = 0.7
    idx = 0
    for e in range(epoch):
        for td in train_data:
            optimizer.zero_grad()
            y, cos, l = model_forward(td, model)
            loss1 = classify_loss_fn(y, l)

            l_cos = l.clone().detach()
            # l_cos[l_cos == 0] = -1
            loss2 = regression_loss_fn(cos, l_cos)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            # 指数平均
            loss_sum = 0.9 * loss_sum + 0.1 * loss
            if idx % 100 == 0:
                test_loss = cal_loss(model, test_data)
                P, R, F1 = evaluate(model, test_data)
                print('epoch:{} iter:{} loss:{} test_loss:{} P:{} R:{} F1:{}'
                      .format(e, idx, loss_sum, test_loss, P, R, F1))

                # 选择模型
                model_manager.select_model(F1)
            idx += 1
    # 输出最优模型到文件
    model_manager.save_best()


def cal_loss(model, data):
    loss_sum = 0.7
    classify_loss_fn = torch.nn.BCELoss()
    regression_loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for td in data:
            y, cos, l = model_forward(td, model)
            loss1 = classify_loss_fn(y, l)

            # l[l == 0] = -1
            loss2 = regression_loss_fn(cos, l)
            loss = loss1 + loss2
            loss_sum = 0.99 * loss_sum + 0.01 * loss
    return loss_sum


def evaluate(model, test_data):
    model.eval()
    right = 0.1
    preidt_p = 0.1
    positive = 0.1
    with torch.no_grad():
        for td in test_data:
            y, cos, l = model_forward(td, model)
            y = y.cpu().view(-1).numpy()
            y[y > 0.5] = 1
            y[y <= 0.5] = 0
            preidt_p += y.sum()

            l = l.cpu().view(-1).numpy()
            positive += l.sum()
            l[l == 0] = -1
            right += (y == l).sum()
    P = right / preidt_p
    R = right / positive
    F1 = 2 * P * R / (P + R)
    return P, R, F1


if __name__ == '__main__':
    # bert 2 ste
    # bert joint ste
    cls_model = SBertCls()
    freeze_parameter(cls_model)
    cls_model.cuda()
    train_data, test_data = get_bert_dataloader()
    train(cls_model, train_data, test_data, epoch=20)

    P, R, F1 = evaluate(cls_model, train_data)
    print('train P:{} R:{} F1:{}'.format(P, R, F1))
    P, R, F1 = evaluate(cls_model, test_data)
    print('test P:{} R:{} F1:{}'.format(P, R, F1))

    # # xlnet
    # cls_model = XLNetCls()
    # cls_model.cuda()
    # train_data, test_data = get_xlnet_dataloader()
    # train(cls_model, train_data, test_data, epoch=50)
    # evaluate(cls_model, train_data)
    # evaluate(cls_model, test_data)
