import torch
import numpy as np
import utils.config as config


class ModelManager:
    def __init__(self, model, name):
        self.name = name
        self.score = -np.Inf
        self.model = model

    def select_model(self, score):
        if score > self.score:
            print('better model is saved,score : {}'.format(score))
            name = self.name + '_%.4f' % score
            self.save_model(name)

    def save_model(self, name=None):
        if name is None: name = self.name
        torch.save(self.model.state_dict(), config.MODEL_DIR + name + '.model')

    def load_model(self, name):
        if name is None: name = self.name
        self.model.load_state_dict(torch.load(config.MODEL_DIR + name + '.model'))
        return self.model
