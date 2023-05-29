import json
import os
from data import SimcseDataCollator,SimDataset,RobustDataCollator
from FTData import FTDataCollator,FTDataset
import random
import numpy.random
import torch.optim.lr_scheduler as lr_scheduler
from transformers.optimization import AdamW
from transformers import BertTokenizer
from train_func import *
import math
from models import BertForCL
from utils import *


for seed in [1,2,3,4,5]:
    # 参数区
    lr = 3e-5
    lr_decay = 0.02
    epoch = 5
    # seed = 3
    devicenum = 3
    batch_size = 32
    max_length = 128
    hard_neg_bili = 0 #效果不明显，可设为0
    hard_pos_bili = 1 #设为1最好
    train_num = -1  #-1为使用所有训练数据训练，如果想快速实践改后的模型有没有明显问题，可以设置为3000快速跑一下试试
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # eval集组成，如果未来信息可见，则为"c_r_c"，若不可见则为"c_r", "c_r_c"经测试无效
    bert_name = "bert-base-uncased"
    pre_train_model_name = "bert-base-uncased"
    pre_train_model_dir = pre_train_model_name
    device = torch.device("cuda" + ":" + str(devicenum) if torch.cuda.is_available() else "cpu")
    to_train = True
    # train_list = ["daily", "persona", "empa", "topical"]
    # train_list = ["persona", "empa", "topical"]
    train_list = ["dailydialog_grade_transformer_generator-addrank.json",
                "dailydialog_grade_transformer_ranker-addrank.json",
                'ft_daily.json']
    eval_list = ["fed-addrank.json", "engage-addrank.json", "convai2_grade_bert_ranker-addrank.json",
                "convai2_grade_dialogGPT-addrank.json",
                "topicalchat_usr-addrank.json", "personachat_usr-addrank.json",
                "dailydialog_grade_transformer_generator-addrank.json",
                "dailydialog_grade_transformer_ranker-addrank.json",
                "empatheticdialogues_grade_transformer_generator-addrank.json",
                "empatheticdialogues_grade_transformer_ranker-addrank.json",
                "convai2_grade_transformer_generator-addrank.json", "convai2_grade_transformer_ranker-addrank.json",
                "dstc6-addrank.json"
                ]
    for train_data in train_list:
        eval_dstc10_list = ["jsalt_eval-addrank.json", "esl_eval-addrank.json", "ncm_eval-addrank.json",
                            "dstc10-persona_clean_eval-addrank.json",
                            "dstc10-topical_clean_eval-addrank.json"]
        check_list = ["dailydialog_grade_transformer_generator-addrank.json",
                    "dailydialog_grade_transformer_ranker-addrank.json",
                    "engage-addrank.json", "jsalt_eval-addrank.json", "esl_eval-addrank.json", "ncm_eval-addrank.json"]
        save_model = True
        # output_dir = "result/" + "FT-{}{}_4.24_lr{}_decay_{}_negbili_{}_posbili_{}_epoch_{}_seed{}".format(
        #         train_data[:-5], train_num, lr, lr_decay, hard_neg_bili, hard_pos_bili, epoch, seed)
        # output_dir = f'FT-AAU-daily-Epoch:{epoch}-seed:{seed}'
        # output_dir = f'VanillaFT--Epoch:{epoch}-seed:{seed}'
        # train_data_dir = "raw_data/{}_train.txt".format(train_data)
        # train_data_dir = "raw_data/{}_train.txt".format(train_data)
        # output_dir = f"result/VanillaFT{train_data[:-5]}-{seed}"
        output_dir = f"result/FT{train_data[:-5]}-{seed}"
        train_data_dir = f'eval_data/{train_data}'
        # valid_data_dir = "raw_data/daily_valid.txt"
        # prepare_output_dir
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)
        if os.path.exists(output_dir + "/item_check") is False:
            os.makedirs(output_dir + "/item_check")
        if os.path.exists(output_dir + "/score_check") is False:
            os.makedirs(output_dir + "/score_check")

        # prepare data:
        with open("raw_data/daily_neg_dict50.json", "r") as f:
            neg_dict = json.load(f)
            f.close()
        neg_sampler = Negtive_Sampler(neg_dict, hard_neg_bili)
        tokenizer = BertTokenizer.from_pretrained(bert_name)
        train_dataset = FTDataset(tokenizer, train_data_dir, max_length)
        train_data_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_data_sampler, batch_size=batch_size,
                                    collate_fn=FTDataCollator(tokenizer, max_length,))
        # valid_dataset = SimDataset(tokenizer, valid_data_dir, max_length)
        # valid_data_sampler = RandomSampler(valid_dataset)
        # valid_dataloader = DataLoader(valid_dataset, sampler=valid_data_sampler, batch_size=batch_size,
        #                               collate_fn=SimcseDataCollator(tokenizer, max_length, neg_sampler=neg_sampler,
        #                                                             pos_bili=hard_pos_bili))

        # prepare model:
        model = BertForCL.from_pretrained(
            pre_train_model_dir,
        )
        model = model.to(device)
        total = sum([param.nelement() for param in model.parameters()])
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of total parameters: %.2fM" % (total / 1e6))
        print("Number of trainable parameters: %.2fM" % (trainable_num / 1e6))


        load_dir = "./result/AAU-daily-1_4.23_lr5e-05_decay_0.02_negbili_0_posbili_1_epoch_40_seed2"
        # load_dir = ""
        if load_dir != "":
            model.load_checkpoint(load_dir + "/model.pth", device)

        # prepare lr_scheduler and optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in filter(lambda np: np[1].requires_grad, model.named_parameters()) if
                        not any(nd in n for nd in no_decay)],
                "weight_decay": 1e-5,
            },
            {"params": [p for n, p in filter(lambda np: np[1].requires_grad, model.named_parameters()) if
                        any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / epoch)) / 2) * (1 - lr_decay) + lr_decay
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 动态学习率

        # start to train!
        if to_train:
            model.is_ft = True
            for epoch_i in range(1, epoch + 1):
                print('[ Epoch', epoch_i, ']')
                start = time.time()
                train_loss = ft_epoch(
                    model, train_dataloader, optimizer, device=device)
                #model.update_density()
                scheduler.step()
                if save_model:
                    model.save_checkpoint(output_dir + "/model.pth")
                print("Epoch:{}    train_loss:{}    cost_time:{}".format(epoch_i, train_loss, time.time() - start))
                # eval_func(model, eval_list, eval_dstc10_list, check_list, device, output_dir, tokenizer, reload_best_model=False, epoch_name=str(epoch_i))
            # with open(output_dir + "/result.txt", "a") as f:
            #     f.write("train params:\n")
            #     f.write("lr={}\nepoch={}\npre-train model={}\n".format(lr, epoch, pre_train_model_dir))
                # f.write("\nbest loss:{}\n".format(min_loss))

        model.load_checkpoint(output_dir + "/model.pth", device)
        # 用第40个epoch的模型最终评测
        eval_func(model, eval_list, eval_dstc10_list, check_list, device, output_dir, tokenizer, False)
        # 测试模型鲁棒性，需要robust_data.txt，该数据集由dstc10前三个数据集组成
        # valid_dataset = SimDataset(tokenizer, "dstc10_eval_data/robust_data.txt", max_length)
        # neg_sampler.bili = 0
        # robust_dataloader = DataLoader(valid_dataset, sampler=valid_data_sampler, batch_size=batch_size,
        #                                collate_fn=RobustDataCollator(tokenizer, max_length, neg_sampler=neg_sampler,
        #                                                              pos_bili=hard_pos_bili))
        # eval_robust(model, robust_dataloader, device, output_dir)
        # with open(output_dir + "/density.txt", "w") as f:
        #     for i in range(len(model.density_sum)):
        #         f.write(str(model.density_sum[i]) + "\n")
        #     f.close()
        # with open(output_dir + "/result.txt", "a") as f:
        #     f.write("\ntrain score bias of average:{}\n".format(model.get_density_bias()))

