# Mitigating Action-Relation Hallucinations in LVLMs via Relation-aware Visual Enhancement

## Setup

1) Recommended Python: Python 3.10

2) Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3) Default model and data paths (override via CLI if different):
- Model: `./Models/llava-v1.5-7b`
- Images: `./datasets/VG_100K`


## Usage

### 1) Attention-head ARS computation (statistics + heatmap)

```bash
python attention_statistics_analyzer.py 
```


### 2) Relation hallucination evaluation (RVE)

```bash
python Rel_hallucination_llava_MMRel.py 
```
## Datasets

We follow the relation understanding benchmark MMRel for data construction and evaluation. Please refer to the official repository for downloads and details: MMRel: A Relation Understanding Benchmark in the MLLM Era — https://github.com/niejiahao1998/MMRel
