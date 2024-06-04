
# 83.2
python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_cen_1" run.batch_size_eval 4 datasets.spatial_crop "center" datasets.temporal_crop "1/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_cen_2" run.batch_size_eval 4 datasets.spatial_crop "center" datasets.temporal_crop "2/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_cen_3" run.batch_size_eval 4 datasets.spatial_crop "center" datasets.temporal_crop "3/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_cen_4" run.batch_size_eval 4 datasets.spatial_crop "center" datasets.temporal_crop "4/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_tl_1" run.batch_size_eval 4 datasets.spatial_crop "top_left" datasets.temporal_crop "1/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_tl_2" run.batch_size_eval 4 datasets.spatial_crop "top_left" datasets.temporal_crop "2/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_tl_3" run.batch_size_eval 4 datasets.spatial_crop "top_left" datasets.temporal_crop "3/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_tl_4" run.batch_size_eval 4 datasets.spatial_crop "top_left" datasets.temporal_crop "4/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_br_1" run.batch_size_eval 4 datasets.spatial_crop "bottom_right" datasets.temporal_crop "1/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_br_2" run.batch_size_eval 4 datasets.spatial_crop "bottom_right" datasets.temporal_crop "2/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_br_3" run.batch_size_eval 4 datasets.spatial_crop "bottom_right" datasets.temporal_crop "3/4" run.num_beams 1 run.min_len 3 run.max_len 30

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42933 evaluate.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_ar.yaml --options model.load_pretrained True model.pretrained "path_to_checkpoint" run.output_dir "./k400_br_4" run.batch_size_eval 4 datasets.spatial_crop "bottom_right" datasets.temporal_crop "4/4" run.num_beams 1 run.min_len 3 run.max_len 30

