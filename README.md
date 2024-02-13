# AIGTDet
### coco_env
conda create -n coco_env python=3.9
conda activate coco_env
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install scikit-learn==1.1.1
pip install wandb==0.16.1
pip install ray==2.0.1

conda install packaging
conda install -c nvidia cuda-nvcc
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --no-build-isolation .

pip install easydl
pip install sentence-transformers==2.2.2
pip install pandas
pip install ray[tune]

python run_detector.py --output_dir ./my_output --model_type bert --do_train 1 --train_file ./data/gpt2/gpt2_500_train.jsonl --dev_file ./data/gpt2/gpt2_test.jsonl

### allennlp_env
conda create -n allennlp_env python=3.8
conda activate allennlp_env
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -i https://download.pytorch.org/whl/cu113
pip install allennlp==2.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install allennlp-models==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
(for windows)
pip install jsonnetbin
python .\extract_keywords.py --raw_dir "C:\Users\Dong\Desktop\donglz\Coh-MGT-Detection\data\gpt2\gpt2_test.jsonl"
python .\construct_graph.py --kw_file_dir "C:\Users\Dong\Desktop\donglz\Coh-MGT-Detection\data\gpt2\gpt2_test_kws.jsonl"
(for linux)
python extract_keywords.py --raw_dir "/root/autodl-tmp/Coh-MGT-Detection/data/gpt3.5-davinci3/gpt3.5-Mixed-davinci3/gpt3.5_mixed_1000_train.jsonl"
python construct_graph.py --kw_file_dir "/root/autodl-tmp/Coh-MGT-Detection/data/gpt3.5-davinci3/gpt3.5-Mixed-davinci3/gpt3.5_mixed_1000_train_kws.jsonl"
