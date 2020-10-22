import torch
from transformers import BertModel, XLNetModel
import torch.optim as optim
from utils import config
from utils.process_data import get_bert_dataloader, get_xlnet_dataloader


class BertCls(torch.nn.Module):
    def __init__(self):
        super(BertCls, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_PATH)
        self.liner = torch.nn.Sequential(
            torch.nn.BatchNorm1d(config.EMBEDDING_DIM * 2),
            torch.nn.Dropout(),
            torch.nn.Linear(config.EMBEDDING_DIM * 2, 1),
            torch.nn.Sigmoid()
        )

        #句子的注意力变量(1*50),各个子的ebd
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

    def forward(self, ste1, ste2):
        ebd1, cls1 = self.bert(ste1)
        ebd2, cls2 = self.bert(ste2)
        ste_ebd1, ste_ebd2 = self.cal_ste_ebd(ebd1, ebd2)
        # max_pool1, indices1 = torch.max(ebd1, dim=1)
        # max_pool2, indices2 = torch.max(ebd2, dim=1)
        conact = torch.cat((ste_ebd1, ste_ebd2), dim=1)
        out = self.liner(conact)
        return out


class BertClsJoint(torch.nn.Module):
    def __init__(self):
        super(BertClsJoint, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_PATH)
        self.liner = torch.nn.Sequential(
            torch.nn.BatchNorm1d(config.EMBEDDING_DIM),
            torch.nn.Dropout(),
            torch.nn.Linear(config.EMBEDDING_DIM, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, ste12):
        ebd1, cls1 = self.bert(ste12)
        cls = ebd1[:, 0, :]
        out = self.liner(cls)
        return out


class XLNetCls(torch.nn.Module):
    def __init__(self):
        super(XLNetCls, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(config.XLNET_MODEL_PATH)
        self.liner = torch.nn.Sequential(
            torch.nn.BatchNorm1d(config.EMBEDDING_DIM * 2),
            torch.nn.Dropout(),
            torch.nn.Linear(config.EMBEDDING_DIM * 2, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, ste1, ste2):
        ebd1 = self.xlnet(ste1)[0]
        ebd2 = self.xlnet(ste2)[0]
        conact = torch.cat((ebd1[:, -1, :], ebd2[:, -1, :]), dim=1)
        out = self.liner(conact)
        return out


def freeze_parameter(cls_model):
    for n, p in cls_model.named_parameters():
        if 'bert' in n:
            p.requires_grad = False
    for n, p in cls_model.named_parameters():
        if 'bert.encoder.layer.11' in n:
            p.requires_grad = True


def train(model, train_data, test_data, epoch=30):
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    loss_sum = 0.7
    idx = 0
    for e in range(epoch):
        for td in train_data:
            optimizer.zero_grad()

            if config.JIONT:
                s1, l = td
                y = model(s1)
            else:
                s1, s2, l = td
                y = model(s1, s2)

            loss = loss_fn(y, l)
            loss.backward()
            optimizer.step()

            # 指数平均
            loss_sum = 0.9 * loss_sum + 0.1 * loss
            if idx % 100 == 99:
                test_loss = cal_loss(model, test_data)
                print('epoch:{} iter:{} loss:{} test_loss:{}'.format(e, idx, loss_sum, test_loss))
            idx += 1


def cal_loss(model, data):
    loss_sum = 0.7
    loss_fn = torch.nn.BCELoss()
    with torch.no_grad():
        for td in data:
            if config.JIONT:
                s1, l = td
                y = model(s1)
            else:
                s1, s2, l = td
                y = model(s1, s2)
            loss = loss_fn(y, l)
            loss_sum = 0.9 * loss_sum + 0.1 * loss
    return loss_sum


def evaluate(model, test_data):
    model.eval()
    right = 0.1
    preidt_p = 0.1
    positive = 0.1
    with torch.no_grad():
        for td in test_data:
            if config.JIONT:
                s1, l = td
                y = model(s1)
            else:
                s1, s2, l = td
                y = model(s1, s2)
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
    print('P:{} R:{} F1:{}'.format(P, R, F1))


if __name__ == '__main__':
    # bert 2 ste
    # bert joint ste
    if config.JIONT:
        cls_model = BertClsJoint()
    else:
        cls_model = BertCls()
    freeze_parameter(cls_model)
    cls_model.cuda()
    train_data, test_data = get_bert_dataloader()
    train(cls_model, train_data, test_data, epoch=50)
    evaluate(cls_model, train_data)
    evaluate(cls_model, test_data)

    # # xlnet
    # cls_model = XLNetCls()
    # cls_model.cuda()
    # train_data, test_data = get_xlnet_dataloader()
    # train(cls_model, train_data, test_data, epoch=50)
    # evaluate(cls_model, train_data)
    # evaluate(cls_model, test_data)
