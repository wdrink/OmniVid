# MSVD
python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42931 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_cc_msvd.yaml --options model.pretrained "path_to_checkpoint" run.output_dir "./cc_msvd_beam3_3-256" run.batch_size_eval 4 run.num_beams 3 run.min_len 3 run.max_len 256

# MSR-VTT:
python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42935 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_cc_msrvtt.yaml --options model.pretrained "path_to_checkpoint" run.output_dir "./cc_msrvtt_beam3_3-50" run.batch_size_eval 4 run.num_beams 3 run.min_len 3 run.max_len 50
