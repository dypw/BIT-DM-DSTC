import math
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist



from Transformers.src.transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead,BertForMaskedLM
from Transformers.src.transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers import BertTokenizer
class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.pooler_type
    cls.pooler = Pooler(cls.pooler_type)
    if cls.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    special_tokens_mask=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    if special_tokens_mask is not None:
        special_tokens_mask = special_tokens_mask.view((-1, special_tokens_mask.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    if special_tokens_mask is not None:
        pooler_output = cls.pooler(attention_mask, outputs)
    else:
        pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    special_tokens_mask=None,
    c_r_ids = None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        c_r_ids = c_r_ids
    )

    if special_tokens_mask is not None:
        pooler_output = cls.pooler(special_tokens_mask, outputs)
    else:
        pooler_output = cls.pooler(attention_mask, outputs)


    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class FinalPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense1_activation = nn.ELU()
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense2_activation = nn.ELU()
        self.pool = nn.Linear(hidden_size, 1)
        self.pool_activation = nn.Sigmoid()

    def forward(self, hidden_state):
        # 加一个residual，可以看一下有没有效果
        output = self.dense1_activation(self.dense1(hidden_state))+hidden_state
        output = self.dense2_activation(self.dense2(output))+output
        output = self.pool_activation(self.pool(output))
        return output

class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, gap=0.3, pooler_type = "avg_top2", temp=0.05,mom_size=50000):
        super().__init__(config)
        self.maskbert = None
        self.bert2 = None
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert_tokenzier = BertTokenizer.from_pretrained("bert-base-uncased")
        self.output = FinalPooler(config.hidden_size)
        self.pooler_type = pooler_type
        self.temp = temp
        self.gap = gap
        self.kk = 1
        self.pownum1 = 7
        self.pownum2 = 3
        self.alpha = 0.01
        self.is_sub = False
        self.is_eval = False
        self.mark_embedding = nn.Embedding(2, config.hidden_size)
        self.density_sum = [int(mom_size / 1000) * i / mom_size for i in range(1001)]
        self.q = queue.Queue(maxsize=mom_size)
        self.mom_size = mom_size
        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        special_tokens_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        c_r_ids = None,
        context = None,
        response = None,
    ):
        sen_results = sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                special_tokens_mask=special_tokens_mask,
                c_r_ids = c_r_ids,
            )
        if not self.is_eval:
            score = self.output(sen_results.pooler_output)
            score1 = score.view(-1,2)
            loss1 = torch.pow(1-score1[:,0],self.pownum1)+self.alpha*torch.pow(1-score1[:,0],self.pownum2)
            loss2 = torch.pow(score1[:,1],self.pownum1)+self.alpha*torch.pow(score1[:,1],self.pownum2)
            loss = loss1 + loss2

            score_v = [tem.cpu().item() * 1000 for tem in score]
            to_add = [0]*1001
            for i in range(len(score_v)):
                to_add[self.q.get()] -= 1
                to_add[int(score_v[i])] += 1
                self.q.put(int(score_v[i]))
            tem = 0
            for i in range(1001):
                tem+=to_add[i]
                self.density_sum[i]+=tem/self.mom_size

            return loss.mean()
        else:
            if self.is_sub:
                score = self.output(sen_results.pooler_output)
                return [tem.cpu().item() for tem in score], sen_results.pooler_output
            else:
                score2,_ = self.bert2(        input_ids,
                attention_mask,
                special_tokens_mask,
                token_type_ids,
                position_ids)
                score = self.output(sen_results.pooler_output)
                score_c = [tem.cpu().item() for tem in score]
                score_mlm = [0]*len(score_c)
                try:
                    for i in range(len(response)):
                            try:
                                score_mlm[i]=self.get_mlm_score([context[i][-1],response[i]])
                            except:
                                score_mlm[i]=0.2
                except:
                    print("test robust")
                score = [score2[i]+score_c[i]+0.1*score_mlm[i] for i in range(len(score_mlm))]
                return score,sen_results.pooler_output

    def save_checkpoint(self,to_dir):
        state_dict = {t: v for t, v in self.state_dict().items()}
        torch.save(state_dict, to_dir)
        print("successfully save model to {} !!".format(to_dir))
    def load_checkpoint(self,from_dir,device):
        checkpoint = torch.load(from_dir, map_location=lambda storage, loc: storage.cuda(device))
        self.load_state_dict(checkpoint, strict=False)
        print("successfully params from: {} !!".format(from_dir))
    def to_eval(self,eval=False):
        self.is_eval = eval
    def get_density_bias(self):
        tem=0
        min_s = 0
        max_s = 1000
        for i in range(1000):
            if self.density_sum[i]>1e-4:
                min_s = i
                break
        for i in range(1000):
            if (1-self.density_sum[i])<1e-4:
                max_s = i
                break
        mean_s = 1/(max_s-min_s)
        for i in range(min_s,max_s):
            tem+=(self.density_sum[i+1]-self.density_sum[i]-mean_s)*(self.density_sum[i+1]-self.density_sum[i]-mean_s)
        tem=tem/(max_s - min_s)
        return math.sqrt(tem)

    def get_mlm_score(self, conversations):
        text = [conversations]
        text_tokens = self.bert_tokenzier(text, add_special_tokens=True,
                                     padding=True, return_tensors='pt')
        in_ids = []
        to_ids = []
        at_masks = []
        raw_ids = text_tokens["input_ids"].clone()
        for i in range(len(text_tokens["input_ids"][0])):
            if text_tokens["token_type_ids"][0][i] == 0:
                continue
            tem = text_tokens["input_ids"][0].clone()
            tem[i] = 103
            to_ids.append(text_tokens["token_type_ids"][0])
            in_ids.append(tem)
            at_masks.append(text_tokens["attention_mask"][0])
        in_ids = torch.stack(in_ids)
        to_ids = torch.stack(to_ids)
        at_masks = torch.stack(at_masks)
        text_tokens["token_type_ids"] = to_ids.to(self.device)
        text_tokens["attention_mask"] = at_masks.to(self.device)
        text_tokens["input_ids"] = in_ids.to(self.device)

        maskbert_outputs = self.maskbert(**text_tokens, return_dict=True,
                                    output_hidden_states=True)
        logits = (maskbert_outputs.logits).softmax(-1)

        indexs = torch.stack([raw_ids[0] for i in range(len(logits))]).to(self.device)
        output = torch.gather(logits, dim=-1, index=indexs.unsqueeze(-1))

        score = 1
        for i in range(len(logits) - 1):
            score *= output[i][len(raw_ids[0]) - len(logits) + i][0]
        if len(logits) - 1<1: return 0
        score = torch.pow(score, 1 / (len(logits) - 1))
        return score.item()

