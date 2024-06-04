"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import csv
import json
import pickle
import os
import numpy as np
from tqdm import tqdm
import clip
import torch
import torch.distributed as dist
from lavis.common.dist_utils import (
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    main_process,
)
import collections

from lavis.common.registry import registry
from lavis.datasets.build_tokenizer import (
    TokenizerwithTimetoken,
    TokenizerwithBoxtoken,
)
from lavis.datasets.build_tokenizer2 import TokenizerwithIoUtoken
from lavis.tasks.base_task import BaseTask
from lavis.tasks.eval_utils import visualize_video
from lavis.tasks.eval_utils import evaluate as evaluate_sot
from lavis.tasks.eval_utils import iou as calculate_iou

from lavis.datasets.datasets.coco_eval import CocoEvaluator
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

from densevid_eval3.eval_soda import eval_soda
from densevid_eval3.eval_para import eval_para
from densevid_eval3.eval_dvc import eval_dvc

from .kalman_filter import KalmanFilter


def load_results_from_json(filename):
    assert os.path.isfile(filename)
    with open(filename, "r") as f:
        results = json.load(f)
    # for activity net external classification scores
    if "results" in results:
        results = results["results"]
    return results


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(
        self,
        evaluate_type,
        fps,
        num_frms_per_clip,
        num_beams,
        max_len,
        min_len,
        tokenizer,
        dvp_anet_ann_path,
        ar_k700_ann_path,
        cc_msrvtt_ann_path,
        sot_trackingnet_ann_path,
        update_interval,
        update_threshold,
        confidence_threshold,
        compensate,
    ):
        super().__init__()

        self.evaluate_type = evaluate_type
        self.fps = fps
        self.num_frms_per_clip = num_frms_per_clip
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.tokenizer = tokenizer
        self.report_metric = True

        self.dvp_anet_ann_path = dvp_anet_ann_path
        self.ar_k700_ann_path = ar_k700_ann_path
        self.cc_msrvtt_ann_path = cc_msrvtt_ann_path
        self.sot_trackingnet_ann_path = sot_trackingnet_ann_path

        self.update_interval = update_interval
        self.update_threshold = update_threshold
        self.confidence_threshold = confidence_threshold
        self.compensate = compensate

        if self.evaluate_type == "ar_k700" or self.evaluate_type == "ar_k400":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, _ = clip.load("ViT-B-32.pt", device=device)
            print("Loading clip model successfully!")
            self.cats = []
            feats = []
            with open(ar_k700_ann_path) as f:
                reader = csv.reader(f)
                headers = next(reader)
                for row in tqdm(reader):
                    with torch.no_grad():
                        text = clip.tokenize([row[1]]).to(device)
                        clip_feat = clip_model.encode_text(text)
                        self.cats.append(row[1])
                        feats.append(clip_feat)

            cat_feats = torch.cat(feats)
            cat_feats /= cat_feats.norm(dim=-1, keepdim=True)
            self.cat_feats = cat_feats
            self.clip = clip_model
            self.clip.eval()
            self.device = device

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        evaluate_type = run_cfg.evaluate_type
        # use_nucleus_sampling = run_cfg.get("use_nucleus_sampling, False")
        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len

        if "sot" in run_cfg.evaluate_type:
            tokenizer = TokenizerwithBoxtoken(cfg.datasets_cfg)
        elif run_cfg.evaluation_type in ["ar_k400", "qa_msrvtt", "cc_msrvtt", ]:
            tokenizer = TokenizerwithIoUtoken(cfg.datasets_cfg)
        elif run_cfg.cfg.evaluation_type == "dvp_anet":
            tokenizer = TokenizerwithTimetoken(cfg.datasets_cfg)
        else:
            raise NotImplementedError
        
        dvp_anet_ann_path = run_cfg.dvp_anet_ann_path
        ar_k700_ann_path = run_cfg.ar_k700_ann_path
        cc_msrvtt_ann_path = run_cfg.cc_msrvtt_ann_path
        sot_trackingnet_ann_root = cfg.datasets_cfg.vis_root_val[0]
        sot_trackingnet_split = (
            os.path.basename(cfg.datasets_cfg.ann_paths_val[0][0])
            .split(".")[0]
            .split("_")[-1]
            .upper()
        )
        if "TRAIN" in sot_trackingnet_split:
            split = sot_trackingnet_split[:5] + "_" + sot_trackingnet_split[5:]
        elif "TEST" in sot_trackingnet_split:
            split = sot_trackingnet_split
        else:
            # deal with the blip2query case
            # "/mnt/data/TrackingNet/omni_trackingnet_train11_blip2query.json" ->
            sot_trackingnet_split = (
                os.path.basename(cfg.datasets_cfg.ann_paths_val[0][0])
                .split(".")[0]
                .split("_")[-2]
                .upper()
            )
            if "TRAIN" in sot_trackingnet_split:
                split = (
                    sot_trackingnet_split[:5] + "_" + sot_trackingnet_split[5:]
                )
            elif "TEST" in sot_trackingnet_split:
                split = sot_trackingnet_split
            else:
                split = "none"

        sot_trackingnet_ann_path = os.path.join(sot_trackingnet_ann_root, split)

        update_interval = cfg.run_cfg.get("update_interval", 1)
        update_threshold = cfg.run_cfg.get("update_threshold", 0.9)
        confidence_threshold = cfg.run_cfg.get("confidence_threshold", -1e4)
        compensate = cfg.run_cfg.get("compensate", "kf")

        fps = cfg.datasets_cfg["fps"]
        num_frms_per_clip = (
            cfg.datasets_cfg["num_frms_per_clip"]
            if cfg.model_cfg.use_video_qformer
            else 1
        )

        return cls(
            evaluate_type=evaluate_type,
            fps=fps,
            num_frms_per_clip=num_frms_per_clip,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            tokenizer=tokenizer,
            dvp_anet_ann_path=dvp_anet_ann_path,
            ar_k700_ann_path=ar_k700_ann_path,
            cc_msrvtt_ann_path=cc_msrvtt_ann_path,
            sot_trackingnet_ann_path=sot_trackingnet_ann_path,
            update_interval=update_interval,
            update_threshold=update_threshold,
            confidence_threshold=confidence_threshold,
            compensate=compensate,
        )


    def valid_step(self, model, samples):
        task = samples["task"][0].strip()

        if task == "sot":
            assert samples["video"].shape[0] == 1, "Only support batch size 1"
            assert (
                samples["image_sizes"].cpu().tolist()[0][0]
                == samples["image_sizes"].cpu().tolist()[0][1]
            ), "During inference, the height and width of the input image must be the same"
            size = samples["image_sizes"].cpu().tolist()[0][0]  # B, T, 2

            path = samples["file_names"][0]
            height = samples["height"][0]
            width = samples["width"][0]
            # length = samples["length"][0]

            kf = KalmanFilter()
            aug_height, aug_width = size[0], size[1]
            (
                preds,
                pred_bboxes,
                pred_scores,
                kf_bboxes,
                iou_scores,
            ) = model.online_generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=1,
                max_length=self.max_len,
                min_length=self.min_len,
                repetition_penalty=1.0,
                length_penalty=1.0,
                output_scores=True,
                height=aug_height,
                width=aug_width,
                update_interval=self.update_interval,
                update_threshold=self.update_threshold,
                confidence_threshold=self.confidence_threshold,
                compensate=self.compensate,
                tokenizer=self.tokenizer,
                kf=kf,
            )

        elif task == "dense_video_captioning":
            preds = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                repetition_penalty=1.0,
                length_penalty=1.0,
                output_scores=True,
            )

        elif task in [
            "action_recognition",
            "clip_captioning",
            "clip_qa"
        ]:
            preds, _, _ = model.online_caption(
                samples,
                use_nucleus_sampling=False,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                repetition_penalty=1.0,
                length_penalty=1.0,
                output_scores=(self.num_beams == 1),
                tokenizer=self.tokenizer,
            )

        else:
            raise NotImplementedError

        if isinstance(preds, dict):
            preds, token_scores = preds["sequences"], preds["scores"]

        else:
            token_scores = None

        if task not in ["object_detection", "sot"]:
            names = samples["name"]
            gt_durations = samples["duration"]

        results = []
        if task == "dense_video_captioning":
            gt_timestamps = samples["timestamps"]
            gt_sentences = samples["text"]

            for video_name, pred, dur, gt_time, gt_sen in zip(names, preds, gt_durations, gt_timestamps, gt_sentences):
                pred = pred.detach().cpu().numpy().tolist()
                video_name = os.path.basename(video_name).split(".")[0]

                if not video_name.startswith("v_"):
                    video_name = "v_" + video_name
                
                item = {
                    "name": video_name,
                    "duration": dur,
                    "gt_timestamps": gt_time,
                    "gt_sentences": gt_sen
                }
                
                sentences = []
                timestamps = []
                scores = []
                if self.supervise_with_clipwise_sequence:
                    recons_pred = self.tokenizer.restore_sentence_clip(
                        pred, dur, add_sub=True
                    )

                    for sent, time, score in zip(
                        recons_pred[0], recons_pred[1], recons_pred[2]
                    ):
                        sentences.append(sent)
                        timestamps.append(time)
                        scores.append(score)

                else:
                    recons_pred = self.tokenizer.restore_sentence_dvp(pred, dur)

                    for sent, time, sco in zip(
                        recons_pred[0], recons_pred[1], recons_pred[2]
                    ):
                        """if (time[1] - time[0]) < 0.05:
                            continue"""

                        sentences.append(sent)
                        timestamps.append(time)
                        scores.append(sco)

                item["sentences"] = sentences
                item["timestamps"] = timestamps
                item["scores"] = scores
                results.append(item)
        

        elif task == "clip_captioning":
            img_ids = samples["video_id"]

            for pred, img_id in zip(preds, img_ids):
                pred = pred.detach().cpu().numpy().tolist()
                recons_pred = self.tokenizer.restore_sentence_cc(pred)
                if len(recons_pred) == 0:
                    recons_pred = [""]

                # print({"image_id": img_id, "caption": recons_pred[0]})
                results.append({"image_id": img_id, "caption": recons_pred[0]})

        elif task == "clip_qa":
            gt_answers = samples["text"]
            questions = samples["raw_prompt"]
            for question, pred, gt_answer in zip(questions, preds, gt_answers):
                pred = pred.detach().cpu().numpy().tolist()
                try:
                    recons_pred = self.tokenizer.restore_sentence_cc(pred)
                    answer = recons_pred[0]

                    if answer.find("?") != -1:
                        answer = answer[answer.find("?") + 1 :]

                    results.append(
                        {
                            "gt": gt_answer[0].split("Answer:")[1].strip(),
                            "ret_pred": answer,
                            "question": question,
                        }
                    )
                except:
                    results.append(
                        {
                            "gt": gt_answer[0].split("Answer:")[1].strip(),
                            "ret_pred": "",
                            "question": question,
                        }
                    )

        elif task == "action_recognition":
            gt_texts = samples["text"]
            if token_scores is not None and self.num_beams == 1:
                batched_token_scores = []
                # travase each position
                for token_score in token_scores:
                    token_score = torch.nn.functional.softmax(
                        token_score, dim=-1
                    )
                    this_pos = token_score.max(dim=1)[0]
                    batched_token_scores.append(this_pos)

                batched_token_scores = (
                    torch.stack(batched_token_scores, dim=0)
                    .transpose(0, 1)
                    .detach()
                    .cpu()
                )  # .tolist()

            else:
                batched_token_scores = [None] * preds.shape[0]

            for video_name, pred, score, gt_cat in zip(names, preds, batched_token_scores, gt_texts):
                pred = pred.detach().cpu().numpy().tolist()
                try:
                    recons_pred, confidence = self.tokenizer.restore_sentence_cc(pred, score)
                    # print(recons_pred, confidence)
                    raw_pred = recons_pred[0]
                    if raw_pred.find("<event>") != -1:
                        raw_pred = raw_pred[: raw_pred.find("<event>")]

                    text = clip.tokenize([raw_pred]).to(self.device)
                    clip_feat = self.clip.encode_text(text)
                    clip_feat /= clip_feat.norm(dim=-1, keepdim=True)

                    logits = (100 * clip_feat @ self.cat_feats.T).softmax(
                        dim=-1
                    )  # (1, 512) @ (512, 700)
                    _, indices = logits[0].topk(1)
                    ind = indices.tolist()[0]
                    topk_cats = self.cats[ind]

                    results.append(
                        {
                            "video_name": video_name,
                            "gt": gt_cat[0],
                            "raw_pred": raw_pred,
                            "ret_pred": topk_cats,
                            "confidence": confidence
                        }
                    )

                except:
                    results.append(
                        {
                            "video_name": video_name,
                            "gt": gt_cat[0],
                            "raw_pred": "",
                            "ret_pred": "",
                            "confidence": 0.0
                        }
                    )


        elif task == "sot":
            video_name = os.path.basename(os.path.dirname(path[0]))
            scale_x = width / aug_width
            scale_y = height / aug_height

            pred_bboxes[:, 0::2] *= scale_x
            pred_bboxes[:, 1::2] *= scale_y

            kf_bboxes[:, 0::2] *= scale_x
            kf_bboxes[:, 1::2] *= scale_y

            pred_scores = pred_scores.cpu().tolist()
            iou_scores = iou_scores.cpu().tolist()

            # convert boxes from x1y1x2y2 to x1y1wh
            pred_bboxes[:, 2] = pred_bboxes[:, 2] - pred_bboxes[:, 0]
            pred_bboxes[:, 3] = pred_bboxes[:, 3] - pred_bboxes[:, 1]

            results.append(
                {
                    "name": video_name,
                    "boxes": pred_bboxes.tolist(),
                }
            )

        return results

    @staticmethod
    def save_result_dvp(result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)
        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            # combine results from all processes
            merged_result = []
            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                merged_result += res

            merged_dict = {}
            for re in merged_result:
                merged_dict[re["name"]] = []
                for sent, time in zip(
                    re["sentences"], re["timestamps"]
                ):
                    merged_dict[re["name"]].append(
                        {"sentence": sent, "timestamp": time}
                    )

            final_result = {
                "version": "ActivityNet-v1.3",
                "results": merged_dict,
                "external_data": {"used": False},
            }
            json.dump(final_result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file

    @staticmethod
    def save_result_clipqa(result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            acc = 0.0
            num_samples = 0
            for item in result:
                gt_cat = item["gt"]
                topk_cats = item["pred"]
                acc += float(topk_cats == gt_cat)
                num_samples += 1

            json.dump(
                {"accuracy": acc / num_samples},
                open(final_result_file, "w"),
            )

            print("result file saved to %s" % final_result_file)

        return final_result_file

    @staticmethod
    def save_result_ar(result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            top1 = 0.0
            num_samples = 0
            top1_cls = 0.0
            for item in result:
                gt_cat = item["gt"]
                topk_cats = item["ret_pred"]
                top1_ = float(topk_cats == gt_cat)
                top1 += top1_

                if "classifier_acc" in item:
                    top1_cls += item["classifier_acc"]

                num_samples += 1

            json.dump(
                {
                    "top1_gen": top1 / num_samples,
                    "top1_cls": top1_cls / num_samples,
                },
                open(final_result_file, "w"),
            )

            print("result file saved to %s" % final_result_file)

        return final_result_file


    @staticmethod
    def save_result_sot(result, result_dir, filename, anno_root):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        result_dir_trackingnet = os.path.join(
            registry.get_path("result_dir"), "sot_eval"
        )
        os.makedirs(result_dir_trackingnet, exist_ok=True)

        if is_main_process():
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            for res in result:
                name = res["name"]
                boxes = res["boxes"]

                save_path = os.path.join(result_dir_trackingnet, name + ".txt")

                with open(save_path, "w") as f:
                    for box in boxes:
                        f.write(
                            "%d,%d,%d,%d\n" % (box[0], box[1], box[2], box[3])
                        )
                print("save to %s" % save_path)

            _, s, p, np, per_vid_re = evaluate_sot(
                anno_root, result_dir_trackingnet
            )
            metrics = {
                "Success": s,
                "Precision": p,
                "Normalized_Precision": np,
                "Per_vid_re": per_vid_re,
            }
            json.dump(
                metrics,
                open(final_result_file, "w"),
            )

            print("result file saved to %s" % final_result_file)

        return final_result_file

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        metrics = {}
        # os.makedirs(val_result, exist_ok=True)

        if self.evaluate_type == "dvp_anet":
            eval_result_file = self.save_result_dvp(
                result=val_result,
                result_dir=registry.get_path("result_dir"),
                filename="{}_epoch{}".format(split_name, epoch),
            )

            if is_dist_avail_and_initialized():
                dist.barrier()

            dvc_eval_version = "2018"
            score = collections.defaultdict(lambda: -1)
            dvc_score = eval_dvc(
                json_path=eval_result_file,
                reference=self.dvp_anet_ann_path,
                version=dvc_eval_version,
            )
            dvc_score = {k: sum(v) / len(v) for k, v in dvc_score.items()}
            dvc_score.update(
                eval_soda(eval_result_file, ref_list=self.dvp_anet_ann_path)
            )
            # dvc_score.update(eval_para(dvc_filename, referneces=para_gt_filenames))
            score.update(dvc_score)

            agg_metrics = 0.0
            # Print the averages
            for metric, value in score.items():
                metrics[metric] = round(100 * value, 2)
                agg_metrics += round(100 * value, 2)

            metrics["agg_metrics"] = agg_metrics


        elif self.evaluate_type == "cc_msrvtt":
            eval_result_file = self.save_result(
                result=val_result,
                result_dir=registry.get_path("result_dir"),
                filename="{}_epoch{}".format(split_name, epoch),
                remove_duplicate="image_id",
            )

            if is_dist_avail_and_initialized():
                dist.barrier()

            metrics = self._report_metrics(eval_result_file, "test")

        elif self.evaluate_type == "ar_k700" or self.evaluate_type == "ar_k400":
            eval_result_file = self.save_result_ar(
                result=val_result,
                result_dir=registry.get_path("result_dir"),
                filename="{}_epoch{}".format(split_name, epoch),
            )

            if is_dist_avail_and_initialized():
                dist.barrier()

            re = json.load(open(eval_result_file))
            metrics = re
            metrics["agg_metrics"] = re["top1_gen"] + re["top1_cls"]

        elif self.evaluate_type == "qa_msrvtt":
            eval_result_file = self.save_result_ar(
                result=val_result,
                result_dir=registry.get_path("result_dir"),
                filename="{}_epoch{}".format(split_name, epoch),
            )

            if is_dist_avail_and_initialized():
                dist.barrier()

            re = json.load(open(eval_result_file))
            metrics = re
            metrics["agg_metrics"] = re["top1_gen"]


        elif self.evaluate_type == "sot_trackingnet":
            eval_result_file = self.save_result_sot(
                result=val_result,
                result_dir=registry.get_path("result_dir"),
                filename="{}_epoch{}".format(split_name, epoch),
                anno_root=self.sot_trackingnet_ann_path,
            )

            if is_dist_avail_and_initialized():
                dist.barrier()

            metrics = json.load(open(eval_result_file))

            agg_metrics = (
                metrics["Success"]
                + metrics["Precision"]
                + metrics["Normalized_Precision"]
            )
            metrics["agg_metrics"] = agg_metrics

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        # create coco object and coco_result object
        coco = COCO(self.cc_msrvtt_ann_path)
        coco_result = coco.loadRes(eval_result_file)

        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(coco, coco_result)

        # evaluate on a subset of images by setting
        # coco_eval.params['image_id'] = coco_result.getImgIds()
        # please remove this line when evaluating the full validation set
        # coco_eval.params['image_id'] = coco_result.getImgIds()

        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        coco_eval.evaluate()

        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print(f"{metric}: {score:.3f}")

        agg_metrics = coco_eval.eval["CIDEr"] + coco_eval.eval["METEOR"]
        """log_stats = {split_name: {k: v for k, v in coco_eval.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")"""

        coco_res = {k: v for k, v in coco_eval.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res
