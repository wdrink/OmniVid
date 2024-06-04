import random
import torch
import torchvision
from sklearn.utils import shuffle
from transformers import AutoTokenizer, BartTokenizerFast, T5TokenizerFast


class TokenizerwithIoUtoken(object):
    def __init__(self, config):
        if "bart" in config["tokenizer"]:
            tokenizer = BartTokenizerFast.from_pretrained(config["tokenizer"])
        elif "t5" in config["tokenizer"]:
            tokenizer = T5TokenizerFast.from_pretrained(config["tokenizer"])
        else:
            raise NotImplementedError

        # tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
        addition_tokens = ["<background>"]

        num_iou_bins = config.get("num_iou_bins", 10)
        self.num_iou_bins = num_iou_bins

        if num_iou_bins > 0:
            self.iou_tokens = [f"<iou_bin_{i}>" for i in range(num_iou_bins)]

        else:
            self.iou_tokens = []

        addition_tokens += self.iou_tokens
        tokenizer.add_special_tokens(
            {"additional_special_tokens": addition_tokens}
        )

        self.special_tokens = tokenizer.all_special_tokens
        self.tokenizer = tokenizer

        self.context_length = config["context_len"]
        self.context_length_prompt = config.get("context_len_prompt", 10)
        self.score_type = config.get("score_type", "segment")

    def tokenize(self, input):
        return self.tokenizer.tokenize(input)

    def encode(self, input):
        return self.tokenizer.encode(input)

    def decode(self, input):
        return self.tokenizer.decode(input)

    def convert_tokens_to_ids(self, input):
        return self.tokenizer.convert_tokens_to_ids(input)

    def convert_ids_to_tokens(self, input):
        return self.tokenizer.convert_ids_to_tokens(input)

    def restore_caption_and_iou(self, ids, token_score):
        caption = ""
        score = None
        iou = 0.0
        iou_pos = -1

        for i, id in enumerate(ids):
            token = self.tokenizer.convert_ids_to_tokens(id)
            if token in self.iou_tokens:
                iou = float(token.split("_")[-1][:-1]) / (self.num_iou_bins - 1)
                iou_pos = i
                break

        caption_start_id = 1
        if iou_pos == -1:
            caption = self.decode(ids[caption_start_id:-1])
            if token_score is not None:
                score = sum(token_score[caption_start_id:-1]) / len(
                    token_score[caption_start_id:-1]
                )
        else:
            caption = self.decode(ids[caption_start_id:iou_pos])
            if token_score is not None:
                score = sum(token_score[caption_start_id:iou_pos]) / len(
                    token_score[caption_start_id:iou_pos]
                )

        # print(self.convert_ids_to_tokens(ids), caption)
        return caption.strip(), iou, score

    def restore_sentence_cc(self, ids, token_score=None):
        sen_start = 2
        sen_end = -1
        # print(self.convert_ids_to_tokens(ids))
        while (
            self.convert_ids_to_tokens(ids[sen_end]) in self.special_tokens
        ):  # at least we will stop at </s>
            sen_end -= 1

        sentences = [self.decode(ids[sen_start : sen_end + 1])]
        if token_score is not None:    
            this_score = (
                token_score[sen_start - 1 : sen_end]
                .mean()
                .tolist()
            )
        
            scores = [this_score]
        else:
            scores = [1.0]

        return sentences, scores

    def __call__(
        self,
        caption,
        iou=None,
        tokenize_type="caption",
    ):
        if tokenize_type == "caption" and iou is not None:
            iou_bin = (
                int(iou * (self.num_iou_bins - 1))
                if self.num_iou_bins > 0
                else 0
            )

            if caption[0] != "<background>":
                iou_bin = max(1, iou_bin)

            iou_token = f"<iou_bin_{iou_bin}>" if self.num_iou_bins > 0 else ""
            input = f"{caption[0]} {iou_token}"
            max_length = self.context_length

        elif tokenize_type == "caption":
            input = f"{caption[0]}"
            max_length = self.context_length

        elif tokenize_type == "prompt":
            input = caption[0]
            max_length = self.context_length_prompt

        text = self.tokenizer(
            input,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        """if tokenize_type == "caption":
            print(f"input: {caption}, iou: {iou} => {self.tokenize(input)}")
        else:
            print(f"propmt: {caption} => {self.tokenize(input)}")"""

        return text, input
