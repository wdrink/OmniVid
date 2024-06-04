import random
import torch
import torchvision
from sklearn.utils import shuffle
from transformers import AutoTokenizer, BartTokenizerFast, T5TokenizerFast

class TokenizerwithTimetoken(object):
    def __init__(self, config):
        if "bart" in config["tokenizer"]:
            tokenizer = BartTokenizerFast.from_pretrained(config["tokenizer"])
        elif "t5" in config["tokenizer"]:
            tokenizer = T5TokenizerFast.from_pretrained(config["tokenizer"])
        else:
            raise NotImplementedError

        # tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
        num_bins = config.get("num_bins", 100)
        self.time_tokens = [f"<time_bin_{i}>" for i in range(num_bins)]
        addition_tokens = self.time_tokens + ["<event>", "</event>"]
        addition_tokens += ["<background>"]

        num_iou_bins = config.get("num_iou_bins", 0)
        self.num_iou_bins = num_iou_bins

        if num_iou_bins > 0:
            self.iou_tokens = [f"<iou_bin_{i}>" for i in range(num_iou_bins)]

        else:
            self.iou_tokens = []

        addition_tokens += self.iou_tokens
        
        use_event_counter = config.get("use_event_counter", False)
        self.use_event_counter = use_event_counter

        if use_event_counter:
            max_segments = config.get("max_segments", 30)
            self.counter_tokens = [f"<counter_{i}>" for i in range(max_segments)]
            addition_tokens += self.counter_tokens
        
        tokenizer.add_special_tokens(
            {"additional_special_tokens": addition_tokens}
        )

        self.special_tokens = tokenizer.all_special_tokens

        self.num_frms = config["input_length"] * config["fps"]

        self.num_bins = num_bins
        self.tokenizer = tokenizer

        self.context_length = config["context_len"]
        self.context_length_prompt = config.get("context_len_prompt", 10)

        self.fps = config["fps"]
        self.score_type = config.get("score_type", "segment")
        self.time_triplet = config.get("time_triplet", "start_end_cat")

        self.supervise_with_clipwise_sequence = config.get(
            "supervise_with_clipwise_sequence", False
        )
        # for gebd only:
        timepoint_type = config.get("timepoint_type", "start")
        assert timepoint_type in ["start", "end", "both"]
        self.timepoint_type = timepoint_type

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

    def timepoint2token(self, start_time, duration):
        start_time = min(duration, start_time)
        # end_time = min(duration, end_time)

        start_time_bin = int(start_time / duration * (self.num_bins - 1))
        # end_time_bin = int(end_time / duration * (self.num_bins - 1))
        return f"<time_bin_{start_time_bin}>"

    def token2timepoint(self, start_id, duration):
        if duration is None:
            return None

        try:
            start_time_bin = int(
                self.convert_ids_to_tokens(start_id).split("_")[-1][:-1]
            )

        except:
            return None

        start_time = start_time_bin / (self.num_bins - 1) * duration
        return round(start_time, 2)

    def timestamps2tokens(self, start_time, end_time, duration):
        start_time = min(duration, start_time)
        end_time = min(duration, end_time)

        start_time_bin = int(start_time / duration * (self.num_bins - 1))
        end_time_bin = int(end_time / duration * (self.num_bins - 1))
        return f"<time_bin_{start_time_bin}><time_bin_{end_time_bin}>"

    def tokens2timestamps(self, start_id, end_id, duration):
        if duration is None:
            return None, None

        if self.time_triplet == "start_end_cat":
            start_time_bin = int(
                self.convert_ids_to_tokens(start_id).split("_")[-1][:-1]
            )
            end_time_bin = int(
                self.convert_ids_to_tokens(end_id).split("_")[-1][:-1]
            )

            if start_time_bin == end_time_bin:
                start_time_bin = max(0, start_time_bin - 1)
            elif start_time_bin > end_time_bin:
                tmp = start_time_bin
                start_time_bin = end_time_bin
                end_time_bin = tmp

            start_time = start_time_bin / (self.num_bins - 1) * duration
            end_time = end_time_bin / (self.num_bins - 1) * duration
            start_time = max(0, start_time)
            end_time = min(end_time, duration)

        elif self.time_triplet == "center_duration_cat":
            center_time_bin = int(
                self.convert_ids_to_tokens(start_id).split("_")[-1][:-1]
            )
            duration_bin = int(
                self.convert_ids_to_tokens(end_id).split("_")[-1][:-1]
            )
            center_time = center_time_bin / (self.num_bins - 1) * duration
            duration_time = duration_bin / (self.num_bins - 1) * duration

            start_time = center_time - duration_time / 2
            end_time = center_time + duration_time / 2

            start_time = max(0, start_time)
            end_time = min(end_time, duration)

        return [round(start_time, 2), round(end_time, 2)]
    
    def restore_sentence_clip(self, ids, duration, add_sub=False):
        sentences = []
        timestamps = []
        scores = []

        cur_ptr = 0
        while cur_ptr < len(ids) - 3:
            cur_token = self.convert_ids_to_tokens(ids[cur_ptr])
            next_token = self.convert_ids_to_tokens(ids[cur_ptr + 1])

            if cur_token in self.time_tokens and next_token in self.time_tokens:
                # start to decode a event
                start_time_ptr = cur_ptr
                end_time_ptr = cur_ptr + 1

                sen_end_ptr = cur_ptr + 2

                while (
                    self.convert_ids_to_tokens(ids[sen_end_ptr])
                    not in self.special_tokens
                    or self.convert_ids_to_tokens(ids[sen_end_ptr])
                    == "<background>"
                ):  # at least we will stop at </s>
                    sen_end_ptr += 1

                sentence = self.decode(ids[end_time_ptr + 1 : sen_end_ptr])
                # jump is sentence is background
                if sentence.strip() == "<background>":
                    cur_ptr = sen_end_ptr
                    continue

                sentences.append(sentence.strip())

                tentative_iou_t = self.convert_ids_to_tokens(ids[sen_end_ptr])
                if self.num_iou_bins > 0 and tentative_iou_t in self.iou_tokens:
                    iou = float(tentative_iou_t.split("_")[-1][:-1]) / (
                        self.num_iou_bins - 1
                    )
                    scores.append(iou)
                elif self.num_iou_bins > 0:
                    scores.append(0.1)

                # find event successfully
                timestamp = self.tokens2timestamps(
                    ids[start_time_ptr], ids[end_time_ptr], duration
                )

                if timestamp in timestamps:
                    cur_ptr = sen_end_ptr
                    sentences.pop()
                    scores.pop()
                else:
                    timestamps.append(timestamp)
                    cur_ptr = sen_end_ptr
            else:
                cur_ptr += 1

        # remove the empty sentences and the sentences with only <background>
        new_sentences = []
        new_timestamps = []
        new_scores = []

        for sentence, timestamp, score in zip(sentences, timestamps, scores):
            if sentence.strip() != "" and "<background>" not in sentence:
                new_sentences.append(sentence)
                new_timestamps.append(timestamp)
                new_scores.append(score)

        sentences = new_sentences
        timestamps = new_timestamps
        scores = new_scores

        # print(self.convert_ids_to_tokens(ids), sentences, timestamps, scores)
        if len(sentences) == 0:
            return [""], [[0, duration]], [0.1]

        # sort the sentences, timestamps, scores by the start time
        sentences, timestamps, scores = zip(
            *sorted(
                zip(sentences, timestamps, scores),
                key=lambda x: x[1][0],
            )
        )

        new_sentences = []
        new_timestamps = []
        new_scores = []

        last_consecutive_timestatmp_idx = 0
        ptr = 1
        while ptr < len(timestamps):
            if timestamps[ptr][0] == timestamps[ptr - 1][1]:
                ptr += 1
            else:
                # current timestamp is not consecutive with the previous one
                consecutive_timestamps_cnt = (
                    ptr - last_consecutive_timestatmp_idx
                )
                if consecutive_timestamps_cnt == 1:
                    # we have only one timestamp, so we don't need to merge
                    new_sentences.append(
                        sentences[last_consecutive_timestatmp_idx]
                    )
                    new_timestamps.append(
                        timestamps[last_consecutive_timestatmp_idx]
                    )
                    new_scores.append(scores[last_consecutive_timestatmp_idx])
                else:
                    # we have more than one timestamps, so we need to merge
                    # For MR, we only need to add one new timestamp that covers all the consecutive timestamps
                    sub_new_timestamp = [
                        timestamps[last_consecutive_timestatmp_idx][0],
                        timestamps[ptr - 1][1],
                    ]

                    if add_sub:
                        new_timestamps.extend(
                            timestamps[last_consecutive_timestatmp_idx:ptr]
                        )
                        new_sentences.extend(
                            sentences[last_consecutive_timestatmp_idx:ptr]
                        )
                        new_scores.extend(
                            scores[last_consecutive_timestatmp_idx:ptr]
                        )

                    new_timestamps.append(sub_new_timestamp)
                    new_sentences.append(
                        sentences[last_consecutive_timestatmp_idx]
                    )
                    new_scores.append(
                        min(
                            max(scores[last_consecutive_timestatmp_idx:ptr])
                            + 1e-2,
                            1.0,
                        )
                    )

                # update the pointer to the new start of consecutive timestamps
                last_consecutive_timestatmp_idx = ptr
                ptr += 1

        if last_consecutive_timestatmp_idx == ptr - 1:
            # we have only one timestamp, so we don't need to merge
            new_sentences.append(sentences[last_consecutive_timestatmp_idx])
            new_timestamps.append(timestamps[last_consecutive_timestatmp_idx])
            new_scores.append(scores[last_consecutive_timestatmp_idx])

        else:
            # For MR, we only need to add one new timestamp that covers all the consecutive timestamps
            sub_new_timestamp = [
                timestamps[last_consecutive_timestatmp_idx][0],
                timestamps[ptr - 1][1],
            ]

            if add_sub:
                new_timestamps.extend(
                    timestamps[last_consecutive_timestatmp_idx:ptr]
                )
                new_sentences.extend(
                    sentences[last_consecutive_timestatmp_idx:ptr]
                )
                new_scores.extend(scores[last_consecutive_timestatmp_idx:ptr])

            new_timestamps.append(sub_new_timestamp)
            new_sentences.append(sentences[last_consecutive_timestatmp_idx])
            new_scores.append(
                min(
                    max(scores[last_consecutive_timestatmp_idx:ptr]) + 1e-2, 1.0
                )
            )
        return new_sentences, new_timestamps, new_scores


    def restore_sentence_dvp(self, ids, duration, token_score=None):
        sentences = []
        timestamps = []
        scores = []
        # print(self.decode(ids))
        cur_ptr = 0
        while cur_ptr < len(ids) - 3:
            cur_token = self.convert_ids_to_tokens(ids[cur_ptr])
            next_token = self.convert_ids_to_tokens(ids[cur_ptr + 1])
            if cur_token in self.time_tokens and next_token in self.time_tokens:
                # start to decode a event
                time_end_ptr = cur_ptr + 1
                sen_end_ptr = time_end_ptr + 1

                while (
                    self.convert_ids_to_tokens(ids[sen_end_ptr])
                    not in self.special_tokens
                ):  # at least we will stop at </s>
                    sen_end_ptr += 1

                # find event successfully
                timestamp = self.tokens2timestamps(
                    ids[time_end_ptr - 1], ids[time_end_ptr], duration
                )
                # print(time_end_ptr, sen_end_ptr)
                sentence = self.decode(ids[time_end_ptr + 1 : sen_end_ptr])
                sentences.append(sentence.strip())

                if token_score is not None:
                    if self.score_type == "category":
                        this_score = (
                            token_score[time_end_ptr + 1 : sen_end_ptr]
                            .mean()
                            .tolist()
                        )
                    elif self.score_type == "segment":
                        this_score = (
                            token_score[time_end_ptr - 1 : time_end_ptr + 1]
                            .mean()
                            .tolist()
                        )
                    else:
                        raise NotImplementedError

                    scores.append(this_score)
                else:
                    scores.append(1.0)

                timestamps.append(timestamp)
                cur_ptr = sen_end_ptr

            else:
                cur_ptr += 1

        # filter out the empty sentences
        new_sentences = []
        new_timestamps = []
        new_scores = []

        for sentence, timestamp, score in zip(sentences, timestamps, scores):
            if sentence.strip() != "":
                new_sentences.append(sentence)
                new_timestamps.append(timestamp)
                new_scores.append(score)

        sentences = new_sentences
        timestamps = new_timestamps
        scores = new_scores

        if len(sentences) == 0:
            return [""], [[0, duration]], [0.1]
        else:
            return sentences, timestamps, scores

    def restore_sentence_gebd(self, ids, duration, token_score=None):
        sentences = []
        timestamps = []
        scores = []

        # simply find all the time tokens
        for i, id in enumerate(ids):
            if self.convert_ids_to_tokens(id) in self.time_tokens:
                timepoint = self.token2timepoint(id, duration)
                sen_end_ptr = i + 1
                while (
                    self.convert_ids_to_tokens(ids[sen_end_ptr])
                    not in self.special_tokens
                ):  # at least we will stop at </s>
                    sen_end_ptr += 1

                if sen_end_ptr == i + 1:
                    sentences.append("")
                else:
                    sentence = self.decode(ids[i + 1 : sen_end_ptr])
                    sentences.append(sentence.strip())

                timestamps.append(timepoint)
                scores.append(1.0)

            else:
                continue

        # print(self.convert_ids_to_tokens(ids), sentences, timestamps, scores)
        return sentences, timestamps, scores

    def restore_sentence_tal(self, ids, duration):
        sentences = []
        timestamps = []
        scores = []

        cur_ptr = 0
        while cur_ptr < len(ids) - 4:
            cur_token = self.convert_ids_to_tokens(ids[cur_ptr])
            next_token = self.convert_ids_to_tokens(ids[cur_ptr + 1])
            next_next_token = self.convert_ids_to_tokens(ids[cur_ptr + 2])
            if (
                cur_token in self.time_tokens
                and next_token in self.time_tokens
                and next_next_token in self.iou_tokens
            ):
                # start to decode a event
                time_end_ptr = cur_ptr + 1
                iou_ptr = cur_ptr + 2
                sen_end_ptr = iou_ptr + 1

                score = self.convert_ids_to_tokens(ids[iou_ptr]).split("_")[-1][
                    :-1
                ]
                score = float(score) / (self.num_iou_bins - 1)

                while (
                    self.convert_ids_to_tokens(ids[sen_end_ptr])
                    not in self.special_tokens
                ):  # at least we will stop at </s>
                    sen_end_ptr += 1

                # find event successfully
                timestamp = self.tokens2timestamps(
                    ids[time_end_ptr - 1], ids[time_end_ptr], duration
                )
                # print(time_end_ptr, sen_end_ptr)
                sentence = self.decode(ids[time_end_ptr + 1 : sen_end_ptr])

                if self.convert_ids_to_tokens(ids[sen_end_ptr]) == "</s>":
                    sen_end_ptr += 1

                sentences.append(sentence.strip())
                timestamps.append(timestamp)
                scores.append(score)

                cur_ptr = sen_end_ptr

            else:
                cur_ptr += 1

        # print(self.convert_ids_to_tokens(ids), sentences, timestamps, scores)
        return sentences, timestamps, scores

    def iou(self, start1, end1, start2, end2):
        try:
            start1 = float(start1.split("_")[-1][:-1])
            end1 = float(end1.split("_")[-1][:-1])

            start2 = float(start2.split("_")[-1][:-1])
            end2 = float(end2.split("_")[-1][:-1])
        except:
            return 0.0

        if start1 >= end1 or start2 >= end2:
            return 0.0

        intersection = min(end1, end2) - max(start1, start2)
        union = max(end1, end2) - min(start1, start2)
        # print(start1, end1, start2, end2, intersection / union)
        return max(intersection / union, 0.0)

    def padd_iou_tokens(self, gt_ids, pred_ids):
        if self.num_iou_bins == 0:
            return torch.LongTensor(gt_ids)

        padded_gt_ids = []

        gt_ids[gt_ids == -100] = self.convert_tokens_to_ids("<pad>")
        pred_ids[pred_ids == -100] = self.convert_tokens_to_ids("<pad>")

        for gt_id, pred_id in zip(gt_ids, pred_ids):  # loop the batch
            padded_gt_id = []
            gt_tokens = self.convert_ids_to_tokens(gt_id.tolist())
            pred_tokens = self.convert_ids_to_tokens(pred_id.tolist())

            iou_bin = None
            for i, gt_t in enumerate(gt_tokens):
                if gt_t == "<event>":
                    padded_gt_id.append(self.convert_tokens_to_ids(gt_t))
                    start, end = gt_tokens[i + 1], gt_tokens[i + 2]
                    pred_start, pred_end = (
                        pred_tokens[i + 1],
                        pred_tokens[i + 2],
                    )

                    # print(start, end, pred_start, pred_end)
                    iou = self.iou(start, end, pred_start, pred_end)
                    iou_bin = f"<iou_bin_{int(iou * (self.num_iou_bins - 1))}>"

                    padded_gt_id.append(
                        self.convert_tokens_to_ids(gt_tokens[i + 1])
                    )
                    padded_gt_id.append(
                        self.convert_tokens_to_ids(gt_tokens[i + 2])
                    )
                    padded_gt_id.append(self.convert_tokens_to_ids(iou_bin))

                elif gt_t in self.time_tokens:
                    continue

                else:
                    padded_gt_id.append(self.convert_tokens_to_ids(gt_t))

            padded_gt_id = padded_gt_id[: self.context_length]
            padded_gt_ids.append(padded_gt_id)

        padded_gt_ids = torch.LongTensor(padded_gt_ids)

        return padded_gt_ids

    def restore_sentence_cc(self, ids):
        event_starts = []
        event_ends = []
        last_end = True
        for i, id in enumerate(ids):
            if id == self.convert_tokens_to_ids("<event>"):
                if last_end:
                    event_starts.append(i)
                    last_end = False
            elif id == self.convert_tokens_to_ids("</event>"):
                if not last_end:
                    event_ends.append(i)
                    last_end = True

        assert (
            len(event_starts) == len(event_ends)
            or len(event_starts) == len(event_ends) + 1
        ), self.convert_ids_to_tokens(ids)

        if len(event_starts) == len(event_ends) + 1:
            event_ends.append(len(ids))

        event_starts = event_starts[0:1]
        event_ends = event_ends[0:1]

        sentences = []
        # timestamps = []
        for start, end in zip(event_starts, event_ends):
            sentence = self.decode(ids[start + 1 : end])
            sentences.append(sentence.strip())
            # timestamps.append(timestamp)

        # print(sentences)
        return sentences  # , timestamps

    def restore_sentence_mr(self, ids, duration, source):
        # print(self.convert_ids_to_tokens(ids))
        timestamps = []
        cur_ptr = 0
        while cur_ptr < len(ids) - 2:
            cur_token = self.convert_ids_to_tokens(ids[cur_ptr])
            next_token = self.convert_ids_to_tokens(ids[cur_ptr + 1])
            if cur_token in self.time_tokens and next_token in self.time_tokens:
                # start to decode a event
                time_end_ptr = cur_ptr + 1

                # find event successfully
                timestamp = self.tokens2timestamps(
                    ids[time_end_ptr - 1], ids[time_end_ptr], duration
                )

                timestamps.append(timestamp)
                cur_ptr = time_end_ptr + 1

            else:
                cur_ptr += 1

        if source == "Charades":
            if len(timestamps) == 0:
                return [[0.0, duration]]
            else:
                if self.time_triplet == "start_end_cat":
                    return timestamps[0:1]
                else:
                    center_dur = timestamps[0:1][0]
                    start = max(0, center_dur[0] - center_dur[1] / 2)
                    end = min(center_dur[0] + center_dur[1] / 2, duration)
                    return [[start, end]]

        elif source == "qvhighlights":
            return timestamps

    def combine_sentences(
        self, sentences, timestamps, duration, segment_masks, is_gebd, ious
    ):
        input = ""

        if self.use_event_counter:
            input += f"There are <counter_{len(timestamps)}> events in the given video. "
        
        # debug the moment retrieval
        time_tokens = []
        if duration is not None and duration != -1.0:
            # Dense captioning, or temporal action localization
            if segment_masks is not None:
                segment_masks = segment_masks.tolist()
            else:
                segment_masks = [1] * len(sentences)

            if ious is None:
                ious = [None] * len(sentences)

            for i, (sentence, timestamp, seg_m, iou) in enumerate(zip(
                sentences, timestamps, segment_masks, ious)
            ):
                sentence = sentence.strip()
                start_time, end_time = timestamp

                this_caption, this_caption2 = None, None
                if is_gebd:
                    if start_time == end_time:
                        timepoint = start_time
                        time_token = self.timepoint2token(timepoint, duration)
                        this_caption = f"{time_token}{sentence}"
                    else:
                        if self.timepoint_type == "start":
                            timepoint = start_time
                            time_token = self.timepoint2token(
                                timepoint, duration
                            )
                            this_caption = f"{time_token}{sentence}"
                        elif self.timepoint_type == "end":
                            timepoint = end_time
                            time_token = self.timepoint2token(
                                timepoint, duration
                            )
                            this_caption = f"{time_token}{sentence}"
                        elif self.timepoint_type == "both":
                            timepoint = start_time
                            timepoint2 = end_time
                            time_token = self.timepoint2token(
                                timepoint, duration
                            )
                            time_token2 = self.timepoint2token(
                                timepoint2, duration
                            )
                            this_caption = f"{time_token}{sentence}"
                            this_caption2 = f"{time_token2}{sentence}"

                else:
                    if (
                        start_time >= end_time
                        and self.time_triplet == "start_end_cat"
                    ):
                        continue

                    time_token = self.timestamps2tokens(
                        start_time, end_time, duration
                    )

                    if seg_m == 1:
                        this_caption = f"{time_token}{sentence}"
                    else:
                        this_caption = (
                            f"<noise_time_bin><noise_time_bin>{sentence}"
                        )

                    if iou is not None:
                        iou_bin = (
                            int(iou * (self.num_iou_bins - 1))
                            if self.num_iou_bins > 0
                            else 0
                        )
                        iou_token = f"<iou_bin_{iou_bin}>"
                        this_caption += iou_token

                if self.use_event_counter:
                    input += f"<counter_{i}>"
                
                input += "<event>" + this_caption + "</event>"
                if this_caption2 is not None:
                    input += "<event>" + this_caption2 + "</event>"
                time_tokens.append(time_token)
        else:
            if len(sentences) == 1:
                input += "<event>" + sentences[0] + "</event>"
            else:  # for MSR-VTT only
                assert len(sentences) == 20
                input += "<event>" + random.choice(sentences) + "</event>"

        return input.strip(), time_tokens

    def combine_differences(self, sentences, differences, clip_indicators):
        input = ""
        # debug the moment retrieval
        time_tokens = []
        for sentence, diff, clip_indicator in zip(
            sentences, differences, clip_indicators
        ):
            sentence = sentence.strip()
            start_diff, end_diff = diff

            # print(clip_indicator, (clip_indicator[0] ^ clip_indicator[1]))
            if self.only_inside_segments and (
                not (clip_indicator[0] ^ clip_indicator[1])
            ):
                input += "<event><noevent></event>"
                time_tokens.append("<noevent>")
                continue

            # False means frm_center_timestamp >= closest_timestamp[0]
            # True means  frm_center_timestamp < closest_timestamp[0]
            start_indicator_bin = (
                "<before>" if clip_indicator[0] == 1 else "<after>"
            )
            end_indicator_bin = (
                "<before>" if clip_indicator[1] == 1 else "<after>"
            )

            time_token = f"{start_indicator_bin}<time_bin_{start_diff}>{end_indicator_bin}<time_bin_{end_diff}>"
            this_caption = f"{time_token}{sentence}"

            input += "<event>" + this_caption + "</event>"
            time_tokens.append(time_token)

        return input.strip(), time_tokens

    def __call__(
        self,
        sentences,
        timestamps,
        duration,
        tokenize_type="caption",
        segment_masks=None,
        clip_indicators=None,
        is_gebd=False,
        ious=None,
    ):
        if tokenize_type == "caption":
            input, time_tokens = self.combine_sentences(
                sentences, timestamps, duration, segment_masks, is_gebd, ious
            )
            max_length = self.context_length
        elif tokenize_type == "prompt":
            input = sentences[0]
            max_length = self.context_length_prompt
            time_tokens = None

        elif tokenize_type == "frame_captions":
            input, time_tokens = self.combine_sentences(
                sentences, timestamps, duration, segment_masks
            )
            max_length = self.context_length * 4

        elif tokenize_type == "dense":
            input, time_tokens = self.combine_differences(
                sentences, timestamps, clip_indicators
            )
            # print(duration, len(sentences), len(timestamps), len(clip_indicators))
            max_length = self.context_length

        text = self.tokenizer(
            input,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        target_token_weights = torch.ones_like(text["input_ids"]).float()
        if tokenize_type == "caption" and segment_masks is not None:
            noise_bin_id = self.convert_tokens_to_ids("<noise_time_bin>")
            noise_bin_ids = torch.LongTensor(
                [noise_bin_id] * len(text["input_ids"])
            ).unsqueeze(0)
            target_token_weights = (
                (~(text["input_ids"] == noise_bin_ids)).float().squeeze(0)
            )

        """if tokenize_type == "caption" and segment_masks is not None:
            print(
                f"Target Sequence: => {self.tokenize(input)} with length {len(self.tokenize(input))}"
            )
        elif tokenize_type == "caption":
            # print(sentences, timestamps)
            print(
                f"Input Sequence: => {self.tokenize(input)} with length {len(self.tokenize(input))}"
            )
        elif tokenize_type == "prompt":
            print(
                f"Prompt: => {self.tokenize(input)} with length {len(self.tokenize(input))}"
            )"""

        return text, input, target_token_weights


class TokenizerwithBoxtoken(object):
    def __init__(self, config):
        if "bart" in config["tokenizer"]:
            tokenizer = BartTokenizerFast.from_pretrained(config["tokenizer"])
        elif "t5" in config["tokenizer"]:
            tokenizer = T5TokenizerFast.from_pretrained(config["tokenizer"])
        else:
            raise NotImplementedError

        # tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
        num_bins = config.get("num_box_bins", 1000)  # follow Pix2Seq-v2
        self.num_bins = num_bins
        self.box_tokens = [f"<box_{i}>" for i in range(num_bins)]

        self.pretrained_tokenizer_length = len(tokenizer)

        addition_tokens = self.box_tokens + ["<obj>", "</obj>"]

        num_iou_bins = config.get("num_iou_bins", 0)
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

    def boxes2tokens(self, box, height, width):
        x1, y1, x2, y2 = box

        if height != None and width != None:
            x1 = max(min(x1, width), 0)
            y1 = max(min(y1, height), 0)
            x2 = max(min(x2, width), 0)
            y2 = max(min(y2, height), 0)

            x1_bin = int(x1 / width * (self.num_bins - 1))
            y1_bin = int(y1 / height * (self.num_bins - 1))
            x2_bin = int(x2 / width * (self.num_bins - 1))
            y2_bin = int(y2 / height * (self.num_bins - 1))

        else:  # the boxes are normalized before
            x1_bin = int(x1 * (self.num_bins - 1))
            y1_bin = int(y1 * (self.num_bins - 1))
            x2_bin = int(x2 * (self.num_bins - 1))
            y2_bin = int(y2 * (self.num_bins - 1))

        return f"<box_{x1_bin}><box_{y1_bin}><box_{x2_bin}><box_{y2_bin}>"

    def tokens2boxes(self, x1_id, y1_id, x2_id, y2_id, height, width):
        try:
            x1 = int(self.convert_ids_to_tokens(x1_id).split("_")[-1][:-1])
            y1 = int(self.convert_ids_to_tokens(y1_id).split("_")[-1][:-1])
            x2 = int(self.convert_ids_to_tokens(x2_id).split("_")[-1][:-1])
            y2 = int(self.convert_ids_to_tokens(y2_id).split("_")[-1][:-1])

        except:
            return None

        if x1 == x2:
            x1 = max(0, x1 - 1)

        if y1 == y2:
            y1 = max(0, y1 - 1)

        x1 = x1 / (self.num_bins - 1) * width
        y1 = y1 / (self.num_bins - 1) * height
        x2 = x2 / (self.num_bins - 1) * width
        y2 = y2 / (self.num_bins - 1) * height

        return [x1, y1, x2, y2]

    def find_special_tokens(self, tokens):
        # print(tokens, self.special_tokens)

        pos = -1
        for spe_token in self.special_tokens:
            if tokens.find(spe_token) != -1:
                pos = tokens.find(spe_token)
                return pos

        for box_token in self.box_tokens:
            if tokens.find(box_token) != -1:
                pos = tokens.find(box_token)
                return pos

        return pos

    def padd_iou_tokens(self, gt_ids, pred_ids):
        if self.num_iou_bins == 0:
            return torch.LongTensor(gt_ids)

        padded_gt_ids = []

        gt_ids[gt_ids == -100] = self.convert_tokens_to_ids("<pad>")
        pred_ids[pred_ids == -100] = self.convert_tokens_to_ids("<pad>")

        for gt_id, pred_id in zip(gt_ids, pred_ids):  # loop the batch
            padded_gt_id = []
            gt_tokens = self.convert_ids_to_tokens(gt_id.tolist())
            pred_tokens = self.convert_ids_to_tokens(pred_id.tolist())

            iou_bin = None
            for i, gt_t in enumerate(gt_tokens):
                if gt_t == "<obj>":
                    padded_gt_id.append(self.convert_tokens_to_ids(gt_t))
                    x1, y1, x2, y2 = (
                        gt_tokens[i + 1],
                        gt_tokens[i + 2],
                        gt_tokens[i + 3],
                        gt_tokens[i + 4],
                    )
                    pred_x1, pred_y1, pred_x2, pred_y2 = (
                        pred_tokens[i + 1],
                        pred_tokens[i + 2],
                        pred_tokens[i + 3],
                        pred_tokens[i + 4],
                    )

                    try:
                        box_gt = torch.FloatTensor([x1, y1, x2, y2]).unsqueeze(
                            0
                        )
                        box_pred = torch.FloatTensor(
                            [pred_x1, pred_y1, pred_x2, pred_y2]
                        ).unsqueeze(0)
                        iou = torchvision.ops.box_iou(
                            box_gt, box_pred
                        ).tolist()[0][0]

                    except:
                        iou = 0.0

                    iou_bin = f"<iou_bin_{int(iou * (self.num_iou_bins - 1))}>"

                    padded_gt_id.append(
                        self.convert_tokens_to_ids(gt_tokens[i + 1])
                    )
                    padded_gt_id.append(
                        self.convert_tokens_to_ids(gt_tokens[i + 2])
                    )
                    padded_gt_id.append(self.convert_tokens_to_ids(iou_bin))

                elif gt_t in self.box_tokens:
                    continue

                else:
                    padded_gt_id.append(self.convert_tokens_to_ids(gt_t))

            padded_gt_id = padded_gt_id[: self.context_length]
            padded_gt_ids.append(padded_gt_id)

        padded_gt_ids = torch.LongTensor(padded_gt_ids)

        return padded_gt_ids

    def restore_box(self, ids, height, width, token_score=None):
        bboxes = []
        categories = []
        scores = []

        ids = ids[1:]

        cur_ptr = 0
        while cur_ptr < len(ids) - 6:
            cur_token = self.convert_ids_to_tokens(ids[cur_ptr])
            next_token = self.convert_ids_to_tokens(ids[cur_ptr + 1])
            next_next_token = self.convert_ids_to_tokens(ids[cur_ptr + 2])
            next_next_next_token = self.convert_ids_to_tokens(ids[cur_ptr + 3])

            if (
                cur_token in self.box_tokens
                and next_token in self.box_tokens
                and next_next_token in self.box_tokens
                and next_next_next_token in self.box_tokens
            ):
                # start to decode a box
                box1_ptr = cur_ptr
                box2_ptr = cur_ptr + 1
                box3_ptr = cur_ptr + 2
                box4_ptr = cur_ptr + 3

                if (
                    self.convert_ids_to_tokens(ids[cur_ptr + 4])
                    in self.box_tokens
                ):
                    box1_ptr = cur_ptr + 1
                    box2_ptr = cur_ptr + 2
                    box3_ptr = cur_ptr + 3
                    box4_ptr = cur_ptr + 4

                sen_start_ptr = box4_ptr + 1
                sen_end_ptr = box4_ptr + 1

                while (
                    self.convert_ids_to_tokens(ids[sen_end_ptr])
                    not in self.special_tokens
                ):  # at least we will stop at </s>
                    sen_end_ptr += 1

                # find event successfully
                box = self.tokens2boxes(
                    ids[box1_ptr],
                    ids[box2_ptr],
                    ids[box3_ptr],
                    ids[box4_ptr],
                    height,
                    width,
                )

                category = self.decode(ids[sen_start_ptr:sen_end_ptr])
                if token_score is not None:
                    if self.score_type == "category":
                        this_score = (
                            token_score[sen_start_ptr:sen_end_ptr]
                            .mean()
                            .tolist()
                        )
                    elif self.score_type == "segment":
                        this_score = (
                            token_score[box1_ptr : box4_ptr + 1].mean().tolist()
                        )

                    else:
                        raise NotImplementedError

                    scores.append(this_score)
                else:
                    scores.append(1.0)

                categories.append(category.strip())
                bboxes.append(box)

                cur_ptr = sen_end_ptr

            else:
                cur_ptr += 1

        # print(self.convert_ids_to_tokens(ids), token_score, scores)
        return bboxes, categories, scores

    def combine_sentences(self, category_names, boxes, height, width):
        input = ""

        for cat_name, box in zip(category_names, boxes):
            # print(cat_name, box)
            box_tokens = self.boxes2tokens(box, height, width)
            this_caption = f"{box_tokens}{cat_name}"
            input += "<obj>" + this_caption + "</obj>"

        return input.strip()

    def __call__(
        self, category_names, boxes, height, width, tokenize_prompt=False
    ):
        if not tokenize_prompt:
            # category_names, boxes = shuffle(category_names, boxes)
            input = self.combine_sentences(category_names, boxes, height, width)
            max_length = self.context_length
        elif boxes is None:
            input = (
                category_names[0]
                + ". Where does it locate in the current frame?"
            )
            max_length = self.context_length_prompt
        else:
            input = category_names[0]
            box_tokens = self.boxes2tokens(boxes[0], height, width)
            input += f". The target object in the reference frame locates at <obj>{box_tokens}</obj>, where does it locate in the current frame?"
            max_length = self.context_length_prompt

        text = self.tokenizer(
            input,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        """if tokenize_prompt:
            print(
                f"Prompt:  {input} => {self.tokenize(input)} with length {len(self.encode(input))}"
            )
        else:
            print(
                f"Caption: {input} => {self.tokenize(input)} with length {len(self.encode(input))}"
            )"""

        return text, input
