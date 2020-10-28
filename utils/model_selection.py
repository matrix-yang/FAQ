import torch
import numpy as np
import utils.config as config
import copy


class ModelManager:
    def __init__(self, model, name):
        self.name = name
        self.score = -np.Inf
        self.model = model
        self.best_state_dict = None

    def select_model(self, score):
        if score > self.score:
            self.score = score
            print('better model score : {}'.format(score))
            self.best_state_dict = copy.deepcopy(self.model.state_dict())

    def save_best(self, name=None):
        if name is None: name = self.name
        name = self.name + '_%.4f' % self.score
        print('best model score : {}'.format(self.score))
        torch.save(self.best_state_dict, config.MODEL_DIR + name + '.model')

    def save_model(self, name=None):
        if name is None: name = self.name
        torch.save(self.model.state_dict(), config.MODEL_DIR + name + '.model')

    def load_model(self, name):
        if name is None: name = self.name
        self.model.load_state_dict(torch.load(config.MODEL_DIR + name + '.model'))
        return self.model
