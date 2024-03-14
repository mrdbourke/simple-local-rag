# Simple Local RAG Tutorial

TK image of what we're going to build

Designed to run locally on a NVIDIA GPU.

All the way from PDF ingestion to "chat with PDF" style features.

All using open-source tools.

PDF source: https://pressbooks.oer.hawaii.edu/humannutrition2/ 

You can also run notebook `00-simple-local-rag.ipynb` directly in [Google Colab](https://colab.research.google.com/github/mrdbourke/simple-local-rag/blob/main/00-simple-local-rag.ipynb). 

## Setup

Note: Tested in Python 3.11, running on Windows 11 with a NVIDIA RTX 4090 with CUDA 12.1.

### Clone repo

```
https://github.com/mrdbourke/simple-local-rag.git
```

### Create environment

```
python -m venv venv
```

### Activate environment

Linux/macOS:
```
source venv/bin/activate
```

Windows: 
```
.\venv\Scripts\activate
```

### Install requirements

```
pip install requirements.txt
```

**Note:** You may have to install `torch` manually (`torch` 2.1.1+ is required for newer versions of attention for faster inference) with CUDA, see: https://pytorch.org/get-started/locally/

On Windows I used:

```
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Launch notebook

VS Code:

```
code .
```

Jupyter Notebook

```
jupyter notebook
```

**Setup notes:** 
* If you run into any install/setup troubles, please leave an issue.
* To get access to the Gemma LLM models, you will have to [agree to the terms & conditions](https://huggingface.co/google/gemma-7b-it) on the Gemma model page on Hugging Face. You will then have to authorize your local machine via the [Hugging Face CLI/Hugging Face Hub `login()` function](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication). Once you've done this, you'll be able to download the models. If you're using Google Colab, you can add a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) to the "Secrets" tab.
* Installing and compiling Flash Attention 2 (faster attention implementation) can take ~5-45 minutes depending on your system setup. See the [Flash Attention 2 GitHub](https://github.com/Dao-AILab/flash-attention/tree/main) for more. In particular, if you're running on Windows, see this [GitHub issue thread](https://github.com/Dao-AILab/flash-attention/issues/595).


TODO:
- Finish setup instructions
- Make header image of workflow 
- Add intro to RAG info in README?
- Add extensions to README 
