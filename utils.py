#!/usr/bin/env python3
import torch
import numpy as np


class Preprocessor(object):
    def __init__(self, tokenizer, add_special_tokens=True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def get_ent2token_spans(self, text, entity_list):
        """
        将实体列表转化为token_spans
        Args:
            text(str): 原始文本
            entity_list(list): [(start, end, ent_type), (start, end, ent_type), ...]
        """
        ent2token_spans = []

        inputs = self.tokenizer(
            text,
            add_special_tokens=self.add_special_tokens,
            return_offsets_mapping=True,
        )

        token2char_span_mapping = inputs[
            "offset_mapping"
        ]  # [start, end) in text for each token
        text2tokens = self.tokenizer.tokenize(
            text, add_special_tokens=self.add_special_tokens
        )

        for ent_span in entity_list:
            ent = text[ent_span[0] : ent_span[1] + 1]
            ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)

            token_start_indexs = [
                i for i, v in enumerate(text2tokens) if v == ent2token[0]
            ]
            token_end_indexs = [
                i for i, v in enumerate(text2tokens) if v == ent2token[-1]
            ]

            token_start_index = list(
                filter(
                    lambda x: token2char_span_mapping[x][0] == ent_span[0],
                    token_start_indexs,
                )
            )
            token_end_index = list(
                filter(
                    lambda x: token2char_span_mapping[x][1] - 1 == ent_span[1],
                    token_end_indexs,
                )
            )

            if len(token_start_index) == 0 or len(token_end_index) == 0:
                continue
            token_span = (token_start_index[0], token_end_index[0], ent_span[2])
            ent2token_spans.append(token_span)
        return ent2token_spans  # [(start, end, ent_type)...] # [start, end] in tokenized index


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    多标签分类损失函数
    ref: https://spaces.ac.cn/archives/8373
    说明： y_true和y_pred的shape一致， y_ture的元素非0即1
        1表示对应的类为目标类，0表示对应的类为非目标类
    警告： 请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加
        激活函数，尤其是不能加sigmoid或者softmax！预测阶段则输出
        y_pred大于0的类，如有疑问，请仔细阅读并理解上述博客

    以下具体实现，通过exp函数的性质，使用类似mask机制进行计算
    """

    # 如果真实标签是负样本，预测结果不变，如果真实标签是正样本，则把预测结果变号
    y_pred = (1 - 2 * y_true) * y_pred

    # 对于负样本，不做任何改变，对于正样本，把预测结果减去1e12
    y_pred_neg = y_pred - y_true * 1e12

    # 对于正样本不做任何改变，对于负样本，把预测结果减去1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12

    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_pred * y_true) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)

        try:
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        except:
            f1, precision, recall = 0, 0, 0
        return f1, precision, recall


if __name__ == "__main__":
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained("./roberta_pretrain/")
    text = "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，"
    entity_list = [(9, 11, "叶老桂"), (0, 3, "company")]
    preprocessor = Preprocessor(tokenizer)
    print(preprocessor.get_ent2token_spans(text, entity_list))
