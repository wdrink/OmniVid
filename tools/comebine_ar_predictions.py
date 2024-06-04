import os
import json
import argparse

def load_sing(results_dir):
    jsons = os.listdir(os.path.join(results_dir, "result"))
    jsons = [j for j in jsons if "rank" in j]
    results = {}
    for j in jsons:
        json_path = os.path.join(results_dir, "result", j)
        data = json.load(open(json_path))
        
        for d in data:
            # "video_name": "val_256/abseiling/0wR5jVB-WPk.mp4", "gt": "abseiling", "raw_pred": "abseiling", "ret_pred": "abseiling", "confidence": [0.9959315061569214]
            results[d["video_name"]] = d
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()
    result_dir = args.results_dir

    results = os.listdir(result_dir)
    results = [os.path.join(result_dir, r) for r in results]

    best_pred_dict = load_sing(results[0])

    for r in results[1:]:
        pred_dict = load_sing(r)

        for k, v in pred_dict.items():
            conf1 = v["confidence"]
            conf2 = best_pred_dict[k]["confidence"]

            if conf1 > conf2:
                best_pred_dict[k] = v
    

    acc_raw = 0.
    acc_ret = 0.
    videos = 0
    for k, v in best_pred_dict.items():
        if v["gt"] == v["raw_pred"]:
            acc_raw += 1
        
        if v["gt"] == v["ret_pred"]:
            acc_ret += 1
        
        videos += 1
    
    print(f"Raw acc is {acc_raw / videos}, retreival acc is {acc_ret / videos}.")
    



