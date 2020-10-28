PROJECT_DIR = 'F:/workcode/FAQ/'
BERT_MODEL_PATH = PROJECT_DIR + 'pretrain_model/bert-chinese-base/'
XLNET_MODEL_PATH = PROJECT_DIR + 'pretrain_model/XLNet_model/'

MODEL_DIR = PROJECT_DIR + 'model/'

SENTENCE_MAX_LEN = 50
BATCH_SIZE = 32
EMBEDDING_DIM = 768

DEVICE = 'cuda'
# 是否将句子拼接，True [cls]ste1[sep]ste2[sep]  False [cls]ste1[sep],  [cls]ste2[sep]
JIONT = False
