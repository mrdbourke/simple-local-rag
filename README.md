# Simple Local RAG Tutorial

TK image of what we're going to build

Designed to run locally on a NVIDIA GPU.

All the way from PDF ingestion to "chat with PDF" style features.

All using open-source tools.

PDF source: https://pressbooks.oer.hawaii.edu/humannutrition2/ 

## Setup

Note: Tested in Python 3.11

Create environment:
```
python -m venv venv
```

Linux/macOS:
```
source venv/bin/activate
```

Windows: 
```
.\venv\Scripts\activate
```

Next:
- Finish setup instructions
- Make header image of workflow 


**Note:** Installing and compiling Flash Attention 2 (faster attention implementation) can take ~5-45 minutes depending on your system setup. See the [Flash Attention 2 GitHub](https://github.com/Dao-AILab/flash-attention/tree/main) for more. In particular, if you're running on Windows, see this [GitHub issue thread](https://github.com/Dao-AILab/flash-attention/issues/595).