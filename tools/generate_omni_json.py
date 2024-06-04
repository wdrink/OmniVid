import json
import os
import numpy as np
from tqdm import tqdm
from decord import VideoReader
import cv2

def anet_dvp(raw_json_path, split):
    json_root = os.path.dirname(raw_json_path)
    new_json_path = os.path.join(json_root, "omni_anet_dvp_" + split + ".json")

    videos = json.load(open(raw_json_path))
    new_videos = []
    for k, v in tqdm(videos.items()):
        ann = v.copy()

        ann["video_name"] = os.path.join("videos", k + ".mp4")

        abs_path = os.path.join(
            "/home/v-junkewang/data/activitynet/videos", k + ".mp4"
        )
        if not os.path.exists(abs_path):
            print(abs_path, "not exists")
            continue

        vr = VideoReader(abs_path)
        fps = vr.get_avg_fps()
        ann["fps"] = fps

        ann["captions"] = ann["sentences"]
        ann.pop("sentences")

        timestamps = ann["timestamps"]
        for ts in timestamps:
            if ts[0] >= ts[1]:
                print(k, ts)
        # ann = ann

        ann["task"] = "dense_video_captioning"
        ann["source"] = "activitynet"
        # print(ann)
        new_videos.append(ann)

    # json_str = json.dumps(new_videos)
    with open(new_json_path, "w") as f:
        json.dump(new_videos, f)

    return


# anet_dvp("/home/v-junkewang/data/activitynet/train.json", "train")
# anet_dvp("/home/v-junkewang/data/activitynet/val_1.json", "val1")
# anet_dvp("/home/v-junkewang/data/activitynet/val_2.json", "val2")


def msrvtt(raw_json_path, split):
    json_root = os.path.dirname(raw_json_path)
    new_json_path = os.path.join(json_root, "omni_msrvtt_" + split + ".json")

    videos = json.load(open(raw_json_path))["videos"]
    sentences = json.load(open(raw_json_path))["sentences"]

    new_videos = []
    for vid in videos:
        if vid["split"] != split:
            continue

        vid_name = vid["video_id"]
        # id = vid["id"]

        if split in ["train", "validate"]:
            vid_path = os.path.join("TrainValVideo", vid_name + ".mp4")
        else:
            vid_path = os.path.join("TestVideo", vid_name + ".mp4")

        abs_path = os.path.join("/home/v-junkewang/data/MSR-VTT", vid_path)
        if not os.path.exists(abs_path):
            print(abs_path, "not exists")
            continue

        ann = {
            "video_name": vid_path,
            "id": vid["id"],
            "duration": -1.0,
            "timestamps": [],
            "captions": [],
        }

        for sen in sentences:
            if sen["video_id"] == vid_name:
                ann["timestamps"].append([-1.0, -1.0])
                ann["captions"].append(sen["caption"])

        ann["task"] = "clip_captioning"
        ann["source"] = "MSR-VTT"
        new_videos.append(ann)

    with open(new_json_path, "w") as f:
        json.dump(new_videos, f)

    return


# msrvtt("/home/v-junkewang/data/MSR-VTT/train_val_videodatainfo.json", "train")
# msrvtt("/home/v-junkewang/data/MSR-VTT/train_val_videodatainfo.json", "validate")
# msrvtt("/home/v-junkewang/data/MSR-VTT/test_videodatainfo.json", "test")

def get_duration(video_path):
    # print(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    return round(duration, 2), fps


def find_cat_id(cat_name, cat_id_names):
    for item in cat_id_names:
        if item[1] == cat_name:
            return int(item[0])
    return -1


def k700(raw_json_path, split):
    json_root = os.path.dirname(raw_json_path)
    new_json_path = os.path.join(json_root, "omni_k400_" + split + ".json")

    videos = open(raw_json_path).readlines()

    cat_id_names = open(
        "/home/v-junkewang/data/kinetics400_256/kinetics_400_labels.csv"
    ).readlines()
    cat_id_names = [item.strip("\n").split(",") for item in cat_id_names]

    new_videos = []

    non_exists = 0
    for video in tqdm(videos):
        video = video.strip("\n").split(" ")[0]
        video_name = video.split("/")[1]
        video_cat = video.split("/")[0]  # .replace("_", " ")

        abs_path = os.path.join(
            "/home/v-junkewang/data/kinetics400_256",
            split,
            video_cat,
            video_name,
        )
        if not os.path.exists(abs_path):
            print(abs_path, " not exists.")
            non_exists += 1
            continue

        k = video_name.split(".")[0]
        # duration = get_duration(os.path.join(json_root, "videos", video_name + ".mp4"))

        ann = {
            "video_name": os.path.join(split, video_cat, video_name),
            "duration": -1.0,
            "timestamps": [[-1.0, -1.0]],
            "captions": [video_cat.replace("_", " ")],
            "class_id": find_cat_id(video_cat.replace("_", " "), cat_id_names),
        }

        print(ann)
        ann["task"] = "action_recognition"
        ann["source"] = "k400"
        new_videos.append(ann)

    print("{} videos are not available.".format(non_exists))
    with open(new_json_path, "w") as f:
        json.dump(new_videos, f)

    return


# k700("/home/v-junkewang/data/kinetics400_256/train_256.txt", "train_256")
# k700("/home/v-junkewang/data/kinetics400_256/val_256.txt", "val_256")


def msrvtt_qa(raw_json_path, split):
    json_root = os.path.dirname(raw_json_path)
    new_json_path = os.path.join(json_root, "omni_msrvttqa_" + split + ".json")

    videos = json.load(open(raw_json_path))  # ["videos"]
    # sentences = json.load(open(raw_json_path))["sentences"]

    new_videos = []
    for vid in videos:
        # if vid["split"] != split:
        #    continue

        vid_name = vid["video"]
        # id = vid["id"]

        if split in ["train", "val"]:
            vid_path = os.path.join("TrainValVideo", vid_name)
        else:
            vid_path = os.path.join("TestVideo", vid_name)

        abs_path = os.path.join("/home/v-junkewang/data/MSR-VTT", vid_path)
        if not os.path.exists(abs_path):
            print(abs_path, "not exists")
            continue

        ann = {
            "video_name": vid_path,
            "id": vid_name.split(".")[0],
            "duration": -1.0,
            "timestamps": [[-1.0, -1.0]],
            "captions": [
                "Question: " + vid["question"] + "Answer: " + vid["answer"]
            ],
            "question_id": vid["question_id"],
        }

        ann["task"] = "clip_qa"
        ann["source"] = "MSR-VTT"
        new_videos.append(ann)

    with open(new_json_path, "w") as f:
        json.dump(new_videos, f)

    return


# msrvtt_qa("../../data/anno_downstream/msrvtt_qa_train.json", "train")
# msrvtt_qa("../../data/anno_downstream/msrvtt_qa_val.json", "val")
# msrvtt_qa("../../data/anno_downstream/msrvtt_qa_test.json", "test")



def isvalid(bbox, width, height):
    """x1, y1, x2, y2 = bbox
    if x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height and x1 < x2 and y1 < y2:
        return True
    else:
        return False"""

    """cx, cy, w, h = bbox
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    if (
        x1 >= 0
        and y1 >= 0
        and x2 <= width
        and y2 <= height
        and x1 < x2
        and y1 < y2
    ):
        return True
    else:
        return False"""

    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    if (
        x1 >= 0
        and y1 >= 0
        and x2 <= width
        and y2 <= height
        and x1 < x2
        and y1 < y2
    ):
        return True
    else:
        return False


def trackingnet(raw_json_path, split):
    json_root = os.path.dirname(raw_json_path)
    video_root = json_root  # os.path.dirname(json_root)
    new_json_path = os.path.join(
        json_root, "omni_trackingnet_" + split + ".json"
    )

    videos = json.load(open(raw_json_path))["videos"]
    anns = json.load(open(raw_json_path))["annotations"]

    anns_list2dict = {ann["video_id"]: ann for ann in anns}

    new_videos = []

    for video in tqdm(videos):
        file_names = video["file_names"]
        video_name = os.path.dirname(file_names[0])
        video_id = video["id"]
        height = video["height"]
        width = video["width"]
        length = video["length"]

        ann = anns_list2dict[video_id]

        if not os.path.exists(os.path.join(video_root, video_name)):
            print(os.path.join(video_root, video_name), "not exists")
            continue

        if "train" in split:
            category = ann["category_id"]  # int
            bboxes = ann["bboxes"]  # [[x1, y1, x2, y2], ...]]
            areas = ann["areas"]  # [area1, area2, ...]

            assert len(file_names) == len(bboxes) == len(areas) == length

            new_bboxes = []
            for i in range(length):
                bbox_i = bboxes[i]
                bbox_i_x1, bbox_i_y1, bbox_i_w, bbox_i_h = bbox_i
                bbox_i_x2, bbox_i_y2 = (
                    bbox_i_x1 + bbox_i_w,
                    bbox_i_y1 + bbox_i_h,
                )

                # clamp the bbox
                bbox_i_x1 = max(0, bbox_i_x1)
                bbox_i_y1 = max(0, bbox_i_y1)
                bbox_i_x2 = min(width, bbox_i_x2)
                bbox_i_y2 = min(height, bbox_i_y2)

                new_bboxes.append([bbox_i_x1, bbox_i_y1, bbox_i_x2, bbox_i_y2])

            bboxes = new_bboxes
            
            assert len(bboxes) == len(areas) == length == len(file_names)
            ann = {
                "video_id": video_id,
                "video_name": video_name,
                "file_names": file_names,
                "duration": -1,
                "height": height,
                "width": width,
                "length": length,
                "category": category,
                "bboxes": bboxes,
                "areas": areas,
            }

        else:
            category = ann["category_id"]  # int
            bboxes = ann["bboxes"]  # [[x1, y1, x2, y2], ...]]
            areas = ann["areas"]  # [area1, area2, ...]

            ann = {
                "video_id": video_id,
                "video_name": video_name,
                "file_names": file_names,
                "duration": -1,
                "height": height,
                "width": width,
                "length": length,
                "category": category,
                "bboxes": bboxes[0:1],
                "areas": areas[0:1],
            }

        ann["task"] = "sot"
        ann["source"] = "trackingnet"
        new_videos.append(ann)

    print(new_json_path, len(videos), len(new_videos))

    with open(new_json_path, "w") as f:
        json.dump(new_videos, f)

    return


"""trackingnet("/data1/v-junkewang/trackingnet/TRAIN_0.json", "train0")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_1.json", "train1")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_2.json", "train2")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_3.json", "train3")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_4.json", "train4")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_5.json", "train5")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_6.json", "train6")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_7.json", "train7")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_8.json", "train8")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_9.json", "train9")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_10.json", "train10")
trackingnet("/data1/v-junkewang/trackingnet/TRAIN_11.json", "train11")
trackingnet("/data1/v-junkewang/trackingnet/TEST.json", "test")
"""



def lasot(raw_json_path, split):
    json_root = os.path.dirname(raw_json_path)
    video_root = json_root  # os.path.dirname(json_root)
    new_json_path = os.path.join(json_root, "omni_lasot_" + split + ".json")

    videos = json.load(open(raw_json_path))["videos"]
    anns = json.load(open(raw_json_path))["annotations"]

    anns_list2dict = {ann["video_id"]: ann for ann in anns}

    new_videos = []

    for video in tqdm(videos):
        file_names = video["file_names"]
        video_name = os.path.dirname(file_names[0])
        video_id = video["id"]
        height = video["height"]
        width = video["width"]
        length = video["length"]

        ann = anns_list2dict[video_id]

        if not os.path.exists(os.path.join(video_root, video_name)):
            print(os.path.join(video_root, video_name), "not exists")
            continue

        if "train" in split:
            category = ann["category_id"]  # int
            bboxes = ann["bboxes"]  # [[x1, y1, x2, y2], ...]]
            areas = ann["areas"]  # [area1, area2, ...]

            assert len(file_names) == len(bboxes) == len(areas) == length

            new_bboxes = []
            for i in range(length):
                bbox_i = bboxes[i]
                bbox_i_x1, bbox_i_y1, bbox_i_w, bbox_i_h = bbox_i
                bbox_i_x2, bbox_i_y2 = (
                    bbox_i_x1 + bbox_i_w,
                    bbox_i_y1 + bbox_i_h,
                )

                # clamp the bbox
                bbox_i_x1 = max(0, bbox_i_x1)
                bbox_i_y1 = max(0, bbox_i_y1)
                bbox_i_x2 = min(width, bbox_i_x2)
                bbox_i_y2 = min(height, bbox_i_y2)

                new_bboxes.append([bbox_i_x1, bbox_i_y1, bbox_i_x2, bbox_i_y2])

            bboxes = new_bboxes

            assert len(bboxes) == len(areas) == length == len(file_names)
            ann = {
                "video_id": video_id,
                "video_name": video_name,
                "file_names": file_names,
                "duration": -1,
                "height": height,
                "width": width,
                "length": length,
                "category": category,
                "bboxes": bboxes,
                "areas": areas,
            }

        else:
            category = ann["category_id"]  # int
            bboxes = ann["bboxes"]  # [[x1, y1, x2, y2], ...]]
            areas = ann["areas"]  # [area1, area2, ...]
            assert len(file_names) == len(bboxes) == len(areas) == length

            new_bboxes = []
            for i in range(length):
                bbox_i = bboxes[i]
                bbox_i_x1, bbox_i_y1, bbox_i_w, bbox_i_h = bbox_i
                bbox_i_x2, bbox_i_y2 = (
                    bbox_i_x1 + bbox_i_w,
                    bbox_i_y1 + bbox_i_h,
                )

                # clamp the bbox
                bbox_i_x1 = max(0, bbox_i_x1)
                bbox_i_y1 = max(0, bbox_i_y1)
                bbox_i_x2 = min(width, bbox_i_x2)
                bbox_i_y2 = min(height, bbox_i_y2)

                new_bboxes.append([bbox_i_x1, bbox_i_y1, bbox_i_x2, bbox_i_y2])

            bboxes = new_bboxes

            ann = {
                "video_id": video_id,
                "video_name": video_name,
                "file_names": file_names,
                "duration": -1,
                "height": height,
                "width": width,
                "length": length,
                "category": category,
                "bboxes": bboxes,
                "areas": areas,
            }

        ann["task"] = "sot"
        ann["source"] = "lasot"
        new_videos.append(ann)

    print(new_json_path, len(videos), len(new_videos))

    with open(new_json_path, "w") as f:
        json.dump(new_videos, f)

    return


# lasot("/data1/v-junkewang/LaSOT/train.json", "train")
# lasot("/data1/v-junkewang/LaSOT/test.json", "test")


def got10k(raw_json_path, split):
    json_root = os.path.dirname(raw_json_path)
    video_root = json_root  # os.path.dirname(json_root)
    new_json_path = os.path.join(json_root, "omni_got10k_" + split + ".json")

    videos = json.load(open(raw_json_path))["videos"]
    anns = json.load(open(raw_json_path))["annotations"]

    anns_list2dict = {ann["video_id"]: ann for ann in anns}

    new_videos = []

    for video in tqdm(videos):
        file_names = video["file_names"]
        # print(file_names)

        if split == "train":
            file_names = [os.path.join("train", f) for f in file_names]
        else:
            file_names = [os.path.join("test", f) for f in file_names]

        video_name = os.path.dirname(file_names[0])
        video_id = video["id"]
        height = video["height"]
        width = video["width"]
        length = video["length"]

        ann = anns_list2dict[video_id]

        if not os.path.exists(os.path.join(video_root, video_name)):
            print(os.path.join(video_root, video_name), "not exists")
            continue

        if "train" in split:
            category = ann["category_id"]  # int
            bboxes = ann["bboxes"]  # [[x1, y1, x2, y2], ...]]
            areas = ann["areas"]  # [area1, area2, ...]

            new_bboxes = []
            for i in range(length):
                bbox_i = bboxes[i]
                bbox_i_x1, bbox_i_y1, bbox_i_w, bbox_i_h = bbox_i
                bbox_i_x2, bbox_i_y2 = (
                    bbox_i_x1 + bbox_i_w,
                    bbox_i_y1 + bbox_i_h,
                )

                # clamp the bbox
                bbox_i_x1 = max(0, bbox_i_x1)
                bbox_i_y1 = max(0, bbox_i_y1)
                bbox_i_x2 = min(width, bbox_i_x2)
                bbox_i_y2 = min(height, bbox_i_y2)

                new_bboxes.append([bbox_i_x1, bbox_i_y1, bbox_i_x2, bbox_i_y2])

            bboxes = new_bboxes

            assert len(bboxes) == len(areas) == length == len(file_names)
            ann = {
                "video_id": video_id,
                "video_name": video_name,
                "file_names": file_names,
                "duration": -1,
                "height": height,
                "width": width,
                "length": length,
                "category": category,
                "bboxes": bboxes,
                "areas": areas,
            }

        else:
            category = ann["category_id"]  # int
            bboxes = ann["bboxes"]  # [[x1, y1, x2, y2], ...]]
            areas = ann["areas"]  # [area1, area2, ...]
            assert len(file_names) == len(bboxes) == len(areas) == length

            new_bboxes = []
            for i in range(length):
                bbox_i = bboxes[i]
                bbox_i_x1, bbox_i_y1, bbox_i_w, bbox_i_h = bbox_i
                bbox_i_x2, bbox_i_y2 = (
                    bbox_i_x1 + bbox_i_w,
                    bbox_i_y1 + bbox_i_h,
                )

                # clamp the bbox
                bbox_i_x1 = max(0, bbox_i_x1)
                bbox_i_y1 = max(0, bbox_i_y1)
                bbox_i_x2 = min(width, bbox_i_x2)
                bbox_i_y2 = min(height, bbox_i_y2)

                new_bboxes.append([bbox_i_x1, bbox_i_y1, bbox_i_x2, bbox_i_y2])

            bboxes = new_bboxes


            ann = {
                "video_id": video_id,
                "video_name": video_name,
                "file_names": file_names,
                "duration": -1,
                "height": height,
                "width": width,
                "length": length,
                "category": category,
                "bboxes": bboxes,
                "areas": areas,
            }

        ann["task"] = "sot"
        ann["source"] = "got10k"
        new_videos.append(ann)

    print(new_json_path, len(videos), len(new_videos))

    with open(new_json_path, "w") as f:
        json.dump(new_videos, f)

    return


# got10k("/data1/v-junkewang/GOT10K/train.json", "train")
# got10k("/data1/v-junkewang/GOT10K/test.json", "test")


