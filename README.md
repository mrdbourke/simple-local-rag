# Simple Local RAG Tutorial

Local RAG pipeline we're going to build:

!["This is a flowchart describing a simple local retrieval-augmented generation (RAG) workflow for document processing and embedding creation, followed by search and answer functionality. The process begins with a collection of documents, such as PDFs or a 1200-page nutrition textbook, which are preprocessed into smaller chunks, for example, groups of 10 sentences each. These chunks are used as context for the Large Language Model (LLM). A cool person (potentially the user) asks a query such as "What are the macronutrients? And what do they do?" This query is then transformed by an embedding model into a numerical representation using sentence transformers or other options from Hugging Face, which are stored in a torch.tensor format for efficiency, especially with large numbers of embeddings (around 100k+). For extremely large datasets, a vector database/index may be used. The numerical query and relevant document passages are processed on a local GPU, specifically an RTX 4090. The LLM generates output based on the context related to the query, which can be interacted with through an optional chat web app interface. All of this processing happens on a local GPU. The flowchart includes icons for documents, processing steps, and hardware, with arrows indicating the flow from document collection to user interaction with the generated text and resources."](images/simple-local-rag-workflow-flowchart.png)

All designed to run locally on a NVIDIA GPU.

All the way from PDF ingestion to "chat with PDF" style features.

All using open-source tools.

In our specific example, we'll build NutriChat, a RAG workflow that allows a person to query a 1200 page PDF version of a Nutrition Textbook and have an LLM generate responses back to the query based on passages of text from the textbook.

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
- [ ] Finish setup instructions 
- [x] Make header image of workflow 
- [ ] Add intro to RAG info in README?
- [ ] Add extensions to README 
