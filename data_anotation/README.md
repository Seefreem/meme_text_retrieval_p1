# LlaVA v1.6 7B
For file llava_v1.6_7b.py, it is the script for interacting with quantified LlaVA v1.6 7B. This script is used for generating data which are further used for comparing with the performance of GPT-4o and unquantified LlaVA v1.6 34B.     

To run the script, the following steps should be followed:  
1. Clone this repository and navigate to LLaVA folder
```sh
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```sh
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```sh
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

4. Upgrade to latest code base
```sh
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

5. Copy the "memes" folder and the file "llava_v1.6_7b.py" into the root directory of LLaVA.

6. Run the script
```sh
python llava_v1.6_7b.py

```