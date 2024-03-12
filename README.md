# AIGTDet

NOTE: cuda >= 11.3

## spacy_env

conda create -n spacy_env python=3.9

conda activate spacy_env

### Install torch+cu113

https://pytorch.org/get-started/previous-versions/

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

### Install spaCy 3.4.4 GPU

https://spacy.io/usage

https://github.com/explosion/spaCy/discussions/11436

pip install spacy==3.4.4

pip install cupy-cuda11x

### Install spacy-experimental

[End-to-end neural coref in spaCy](https://github.com/explosion/spaCy/discussions/11585)

pip install spacy-experimental==0.6.2

pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1-py3-none-any.whl

## coco_env

conda create -n coco_env python=3.9

conda activate coco_env

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -i https://download.pytorch.org/whl/cu113

pip install transformers==4.20.1

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

## allennlp_env

conda create -n allennlp_env python=3.8

conda activate allennlp_env

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -i https://download.pytorch.org/whl/cu113

(for windows)

pip install jsonnetbin

pip install allennlp==2.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install allennlp-models==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

python .\extract_keywords.py --raw_dir "C:\Users\Dong\Desktop\donglz\Coh-MGT-Detection\data\gpt2\gpt2_test.jsonl"

python .\construct_graph.py --kw_file_dir "C:\Users\Dong\Desktop\donglz\Coh-MGT-Detection\data\gpt2\gpt2_test_kws.jsonl"

(for linux)

conda install -c conda-forge jsonnet

pip install allennlp==2.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install allennlp-models==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

python extract_keywords.py --raw_dir "/root/autodl-tmp/Coh-MGT-Detection/data/gpt3.5-davinci3/gpt3.5-Mixed-davinci3/gpt3.5_mixed_1000_train.jsonl"

python construct_graph.py --kw_file_dir "/root/autodl-tmp/Coh-MGT-Detection/data/gpt3.5-davinci3/gpt3.5-Mixed-davinci3/gpt3.5_mixed_1000_train_kws.jsonl"
