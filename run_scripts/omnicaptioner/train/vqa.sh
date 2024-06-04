
# training 
python3 -m torch.distributed.run --nnodes 1 \
	--node_rank 0 \
	--master_port 41989 \
	--nproc_per_node 8 \
	train.py --cfg-path ./lavis/projects/omnicaptioner/local/videoblip2_bart_qa.yaml --auto_resume
