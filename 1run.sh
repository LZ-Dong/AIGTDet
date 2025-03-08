# gcn train
CUDA_VISIBLE_DEVICES=1 python AIGTDet/script_zh/run_gcn.py \
--model_type gcn \
--dataset AIGTDet/data_zh/hyper_qwen \
--output_dir AIGTDet/experiments_zh/hyper_qwen_gcn_only