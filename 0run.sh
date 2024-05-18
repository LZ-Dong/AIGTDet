# # construct_graph
# /home/donglingzhong/miniconda3/envs/spacy_env/bin/python construct_graph_coref.py --gpu_id 0 --raw_dir data/GROVER100/train.jsonl
# /home/donglingzhong/miniconda3/envs/spacy_env/bin/python construct_graph_coref.py --gpu_id 0 --raw_dir data/GROVER100/test.jsonl
# # baseline
# CUDA_VISIBLE_DEVICES=0 /home/donglingzhong/miniconda3/envs/AIGC_env/bin/python main.py \
# --train_data data/GROVER100/train_coref.jsonl \
# --valid_data data/GROVER100/test_coref.jsonl \
# --output_dir experiments/grover100_baseline
# # mean
# CUDA_VISIBLE_DEVICES=0 /home/donglingzhong/miniconda3/envs/AIGC_env/bin/python main.py \
# --type mean \
# --epoch 4 \
# --train_data data/GROVER100/train_coref.jsonl \
# --valid_data data/GROVER100/test_coref.jsonl \
# --output_dir experiments/grover100_mean

# CUDA_VISIBLE_DEVICES=0 /home/donglingzhong/miniconda3/envs/AIGC_env/bin/python main.py \
# --type mean \
# --epoch 4 \
# --train_data data/GROVER500/train_coref.jsonl \
# --valid_data data/GROVER500/test_coref.jsonl \
# --output_dir experiments/grover500_mean_200

CUDA_VISIBLE_DEVICES=0 /home/donglingzhong/miniconda3/envs/AIGC_env/bin/python main.py \
--type baseline \
--model roberta-large \
--epoch 4 \
--train_data data/GROVER300/train_coref.jsonl \
--valid_data data/GROVER300/test_coref.jsonl \
--output_dir experiments/grover300_roberta-large