import random

from transformers import BertTokenizer
from models import BertForCL
import torch
import tqdm
devicenum = 3
device = torch.device("cuda"+":"+str(devicenum) if torch.cuda.is_available() else "cpu")
pre_train_model_dir ="result/"+ "bcr+tripletloss0.3-40-seed0"
model = BertForCL.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.load_checkpoint(pre_train_model_dir+"/model.pth",device)
model.to_eval(True)
model = model.to(device)

while True:
    x=input()
    #x ='wow in japana and taiwan there are cat cafes where you can play with them__eou__that seems fun that reminds me of the anderson house bed and breakfast in minnesota they used to offer a free cat for the night when you booked a room__eou__wow there is a town in alaska where the mayor of the town is a cat'
    if x=="q":
        break
    sen = x.split("__eou__")
    sen_enc = tokenizer(
        sen,
        max_length=256,
        truncation=True,
        padding=False,
    )
    begin_word = sen_enc["input_ids"][0][0]
    begin_at_mask = sen_enc["attention_mask"][0][0]
    begin_sp_mask = 0
    begin_to_type = sen_enc["token_type_ids"][0][0]
    tem0=[begin_word]
    tem1=[begin_to_type]
    tem2=[begin_at_mask]
    tem3=[begin_sp_mask]
    tem4 = [0]
    for i in range(len(sen)):
        tem0 += sen_enc["input_ids"][i][1:]
        tem1 += [i%2 for j in range(len(sen_enc["token_type_ids"][i][1:]))]
        tem2 += sen_enc["attention_mask"][i][1:]
        if i == len(sen)-1:
            tem4 += [2 for j in range(len(sen_enc["attention_mask"][i][1:]))]
            tem3 += [1 for j in range(len(sen_enc["attention_mask"][i][1:]))]
            tem3[-1] -= 1
        elif i == len(sen)-2:
            tem4 +=[1 for j in range(len(sen_enc["attention_mask"][i][1:]))]
            tem3 += [0 for j in range(len(sen_enc["attention_mask"][i][1:]))]
        else:
            tem4 +=[0 for j in range(len(sen_enc["attention_mask"][i][1:]))]
            tem3 += [0 for j in range(len(sen_enc["attention_mask"][i][1:]))]


    batch = {}
    batch["input_ids"] = [tem0]
    batch["token_type_ids"] = [tem1]
    batch["attention_mask"] = [tem2]
    batch["special_tokens_mask"] = [tem3]
    batch = tokenizer.pad(
        batch,
        padding='longest',
        max_length=256,
        return_tensors="pt",
    )
    batch.data["special_tokens_mask"] = batch.data["special_tokens_mask"] * batch.data["attention_mask"]
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    special_tokens_mask = batch["special_tokens_mask"].to(device)
    # length = batch.data["special_tokens_mask"].shape[1]
    # c_r_mask = [tem4 + [0] * (length - len(tem4))]
    c_r = torch.tensor(tem4).to(device)
    if len(c_r)>512:
        continue
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "special_tokens_mask": special_tokens_mask,
        "c_r_ids":c_r,
    }
    output = model(**inputs)
    print(output[0])