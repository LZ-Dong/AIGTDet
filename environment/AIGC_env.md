AIGC_env

cuda=12.2

https://github.com/huggingface/transformers/blob/main/README_zh-hans.md

conda create -n AIGC_env python=3.9

conda activate AIGC_env

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

pip install transformers
