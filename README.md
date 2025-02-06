# About 
This repo contains the source code for my project outside course scope at Pioneer Centre for Artificial Intelligence, Denmark.  
Project title: Meme-text retrieval: a new dataset and a cross-model embedder  
Main supervisor: Serge Belongie   
Co-supervisor: Peter Ebert Christensen  
Paper: [Large Vision-Language Models for Knowledge-Grounded Data Annotation of Memes](https://doi.org/10.48550/arXiv.2501.13851)

# Dataset
The proposed dataset is split into **training_set.json** and **validation_set.json**. There is a link towards each meme. 

# Models
We utilized **CLIP** and **LlaVA-1.6** for our experiments. Please refer to their original repositories for details.  
# Environment

## Install

The following instructions are for Linux users.

1. Clone this repository and navigate to meme_text_retrieval_p1 folder
```bash
git clone https://github.com/Seefreem/meme_text_retrieval_p1.git
cd meme_text_retrieval_p1
```

2. Install Packages
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

# Quick Start
## Data Annotation
Run the following command for data annotation:
```Shell
conda activate meme_text
cd data_annotation
python gpt_4o.py --start-id 0  --dataset meme_text_retrieval --prompt-type gpt-4o-all-data
```
When you have the responses from GPT-4o, you may use **post_processing.ipynb** to extract features and check the validity.   
Usually, there will be some missing information. We recommend you filter them out and do annotation again.


## Get Templatic Memes from Figmemes and MemeCap
Run the following command for filtering out templatic memes:
```Shell
cd evaluation
python get_template_based_memes.py --dataset figmemes 
```
After filtering, the code will generate a HTML file for visualizing the paired templates and instances.

## Fine-tuning CLIP
Run the following command to fine-tune CLIP, without hyperparameter searching (you may set "sweep" as True to enable hyperparameter optimization):
```Shell
cd fine_tune
python fine_tune_clip.py --epochs 20 --warmup-epochs 1 --sweep False 
```
Run the following command to test fine-tuned CLIP on your target dataset:
```Shell
python retrieval_test.py --test-data "the json file of your dataset" --root 'root directory of images' --text_type meme_captions  
```

