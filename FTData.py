import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json
from dataclasses import dataclass
import logging
import random
import nlpaug.augmenter.word as naw
from nlpaug.flow import Sometimes


class FTDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
    ):
        super().__init__()
        data_file = Path(data_dir)
        self.examples = self.load_file(data_file)
        self.max_length = max_length
        self.tokenizer = tokenizer

    def load_file(self, filename):
        with open(filename, 'r') as f:
            contents = json.load(f)
            f.close()
        contexts = contents['contexts']
        responses = contents['responses']
        scores = []
        for item in contents['scores']:
            scores.append(item['Overall'])
        maxScore = 0
        minScore = 10000
        for score in scores:
            score = float(score)
            maxScore = max(maxScore, score)
            minScore = min(minScore, score)
        for i in range(len(scores)):
            scores[i] = (float(scores[i]) - minScore) / (maxScore - minScore)
        contents = {'contexts': contexts, 'responses': responses, 'scores': scores}
        return contents

    def __len__(self):
        return len(self.examples["contexts"])

    def __getitem__(self, index):
        return [self.examples["contexts"][index], self.examples["responses"][index], self.examples["scores"][index],
                "none"]


class FTDataCollator:
    def __init__(self, tokenizer: BertTokenizer,max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        assert (
                self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        # self.data_args = data_args

    def __call__(self, batches):
        batch,labels,contexts,responses,references,c_r_mask = self._encode(batches)
        batch = self.tokenizer.pad(
                batch,
                padding='longest',
                max_length=self.max_length,
                return_tensors="pt",
            )
        length = batch.data["special_tokens_mask"].shape[1]
        c_r_mask = [tem+[0]*(length - len(tem)) for tem in c_r_mask]
        batch.data["c_r_ids"] = torch.tensor(c_r_mask)
        batch.data["labels"] = torch.tensor(labels)
        batch.data["special_tokens_mask"] = batch.data["special_tokens_mask"]*batch.data["attention_mask"]
        # batch.data["responses"] = responses
        # batch.data["contexts"] = contexts
        # batch.data["references"] = references
        return batch

    def _encode(self, batches):
        all_sentences = []
        labels = []
        sen_len = []
        contexts = []
        responses = []
        references = []
        c_r_mask = []
        for tem in batches:
            all_sentences += tem[0]
            sen_len.append(len(tem[0]))
            all_sentences.append(tem[1])
            labels.append(tem[2])
            contexts.append(tem[0])
            responses.append(tem[1])
            references.append(tem[3])

        sent_features = self.tokenizer(
            all_sentences,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        begin_word = sent_features["input_ids"][0][0]
        begin_at_mask = sent_features["attention_mask"][0][0]
        begin_sp_mask = 0
        cur_index = 0
        new_sent_features = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "special_tokens_mask": []}
        for k in range(len(batches)):
            num = sen_len[k]  #################3
            flag = 1
            tem0, tem1, tem2, tem3, tem4 = [], [], [], [], []
            tem0 += sent_features["input_ids"][cur_index + sen_len[k]][1:]
            tem1 += [flag for i in range(len(sent_features["token_type_ids"][cur_index + sen_len[k]][1:]))]
            tem2 += sent_features["attention_mask"][cur_index + sen_len[k]][1:]
            tem3 += [1 for i in range(len(sent_features["attention_mask"][cur_index + sen_len[k]][1:]))]
            tem3[-1] -= 1
            tem4 += [2 for i in range(len(sent_features["attention_mask"][cur_index + sen_len[k]][1:]))]

            for j in range(num):
                i = num - j - 1
                it = i + cur_index
                flag = (flag + 1) % 2
                if len(sent_features["input_ids"][it][1:]) + len(tem0) + 1 > self.max_length:
                    break
                tem0 = sent_features["input_ids"][it][1:] + tem0
                tem1 = [flag for i in range(len(sent_features["token_type_ids"][it][1:]))] + tem1
                tem2 = sent_features["attention_mask"][it][1:] + tem2
                tem3 = [0 for i in range(len(sent_features["attention_mask"][it][1:]))] + tem3
                if j == 0:
                    tem4 = [1 for i in range(len(sent_features["attention_mask"][it][1:]))] + tem4
                else:
                    tem4 = [0 for i in range(len(sent_features["attention_mask"][it][1:]))] + tem4
            tem0 = [begin_word] + tem0
            tem1 = [tem1[0]] + tem1
            tem2 = [begin_at_mask] + tem2
            tem3 = [begin_sp_mask] + tem3
            tem4 = [0] + tem4

            new_sent_features["input_ids"].append(tem0)
            new_sent_features["token_type_ids"].append(tem1)
            new_sent_features["attention_mask"].append(tem2)
            new_sent_features["special_tokens_mask"].append(tem3)
            c_r_mask.append(tem4)
            cur_index += num + 1

        return new_sent_features,labels,contexts,responses,references,c_r_mask
