import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math
class Negtive_Sampler(object):
    def __init__(self,dict,bili,type="c_r"):
        self.dict = dict
        for key in dict.keys():
            random.shuffle(self.dict[key])
        self.type = type
        self.list = list(dict.keys())
        self.length = len(self.list)
        self.bili = bili
        self.epoch = 0
    def get_neg(self,sen):
        if self.dict.get(sen) is None:
            res = self.list[random.randint(0, self.length - 1)]
            return res
        if random.random()>self.bili:
            res = self.list[random.randint(0, self.length - 1)]
        else:
            neg_sens = self.dict[sen]
            length = len(neg_sens)-1
            res = neg_sens[random.randint(0, length)]
        return res


class Negtive_Sampler2(object):
    def __init__(self, bili):
        self.bili = bili
        self.weights = {1: (1.0, 0.0, 0.0, 0.0),
                   2: (1 / 2, 1 / 2, 0.0, 0.0),
                   3: (1 / 3, 1 / 3, 1 / 3, 0.0),
                   4: (1 / 4, 1 / 4, 1 / 4, 1 / 4)}
    def get_neg(self,it,batch):
        max_score = 0
        max_sen = None
        sen = batch[it]
        for tem in batch:
            if tem == sen: continue
            score = sentence_bleu(sen,tem,  self.weights[1],
                                  smoothing_function=SmoothingFunction().method7)*math.log10(min(len(tem.split(" ")),10)+1)
            if score>max_score:
                max_score = score
                max_sen = tem
        if max_sen is None:
            while True:
                s = random.randint(0, len(batch)-1)
                if s !=it: break
            max_sen = batch[s]
        return max_sen




