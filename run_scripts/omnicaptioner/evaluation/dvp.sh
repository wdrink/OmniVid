python3 -m torch.distributed.run --nproc_per_node=6 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_dvp.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./dvp_trainval_ep18_beam5_8-256" run.batch_size_eval 1 run.num_beams 5 run.min_len 8 run.max_len 256