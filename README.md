# About 
This repo contains the source code for my project outside course scope at Pioneer Centre for Artificial Intelligence, Denmark.  
Project title: Meme-text retrieval: a new dataset and a cross-model embedder  
Main supervisor: Serge Belongie   
Co-supervisor: Peter Ebert Christensen  

# Models

## [LlaVA](https://github.com/haotian-liu/LLaVA)

# Environment

## Install

If you are not using Linux, do *NOT* proceed.

1. Clone this repository and navigate to meme_text_retrieval_p1 folder
```bash
git clone https://github.com/Seefreem/meme_text_retrieval_p1.git
cd meme_text_retrieval_p1
```

2. Install Package
```Shell
conda create -n meme_text python=3.10 -y
conda activate meme_text
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```
## Install CLIP
```shell
conda install --yes -c pytorch cudatoolkit=11.0
pip install git+https://github.com/openai/CLIP.git
```
# Quick Start
## CLI inference for data annotation

```Shell
python data_annotation.llava_v1.6_7b.py
```

## GPT-4o for data annotation
```shell
python -m  data_annotation.gpt_4o --start-id 1  --file-length 500 
```

## CLI Inference

Chat about images using LLaVA without the need of Gradio interface. It also supports multiple GPUs, 4-bit and 8-bit quantized inference. With 4-bit quantization, for our LLaVA-1.5-7B, it uses less than 8GB VRAM on a single GPU.

```Shell
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
```



