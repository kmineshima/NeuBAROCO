# BlackboxNLP 2025 Dataset

Datasets and scripts for the following paper(s):

- "Normative Reasoning in Large Language Models: A Comparative Benchmark from Logical and Modal Perspectives" (accepted to BlackboxNLP 2025)

## Contents

- [Datasets](#datasets)
  - [Deontic Logic Task Format](#deontic-logic-task-format)
  - [Syllogistic Task Format](#syllogistic-task-format)
- [Running scripts](#running-scripts)
  - [Setup](#setup)
  - [Set API keys](#set-api-keys)
  - [Evaluation](#evaluation)
- [Citation](#citation)

## Datasets

### Deontic Logic Task Format

#### Files

- [`deontic_single.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/blackboxnlp2025/deontic_single.tsv) - Normative problems
- [`epistemic_single.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/blackboxnlp2025/epistemic_single.tsv) - Epistemic problems

#### Description

| Column Name | Description |
| ---- | ---- |
| ID | problem ID |
| premises_en | one premise in English |
| hypothesis_en | one hypothesis in English |
| gold | correct answer, the relationship of the hypothesis to the premise (*entailment*, *non-entailment*) |
| content-type | classification based on belief congruency (*congruent*, *incongruent*, *nonsense*) |
| inference-pattern | type of logical inferences (*NotMu-MiNot*, *NotMi-MuNot*, *MiNot-NotMu*, *Mu-Mi*, *NotMi-NotMu*, *NotMu-NotMi*, *MiNot-MuNot*, *Mi-Mu*, *FC-Or-Elim*, *FC-Or-Intro*, *Ross-Or-Intro*) |
| modal | modality of the premises and hypothesis (*deontic*, *epistemic*) |

- See [our paper](#citation) for details on content-type, inference-pattern, and modal.

### Syllogistic Task Format

#### Files

- [`deontic_multiple.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/blackboxnlp2025/deontic_multiple.tsv) - Normative problems
-  [`epistemic_multiple.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/blackboxnlp2025/epistemic_multiple.tsv) - Epistemic problems

#### Description

| Column Name | Description |
| ---- | ---- |
| ID | problem ID |
| premises_en | two premises in English |
| hypothesis_en | one hypothesis in English |
| gold | correct answer, the relationship of the hypothesis to the premises (*entailment*, *non-entailment*) |
| content-type | classification based on belief congruency (*congruent*, *incongruent*, *nonsense*) |
| inference-pattern | type of logical inferences (*Hyp-MP*, *Hyp-MT*, *Cat-MP*, *Cat-MT*, *Hyp-AC*, *Hyp-DA*, *Cat-AC*, *Cat-DA*) |
| modal | modality of the premises and hypothesis (*deontic*, *epistemic*) |

- See [our paper](#citation) for details on content-type, inference-pattern, and modal.

## Running scripts

### Setup

```bash
git clone https://github.com/kmineshima/NeuBAROCO
cd NeuBAROCO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Set API keys

```bash
export OPENAI_API_KEY=<YOUR_KEY>  # For OpenAI API
export HUGGINGFACE_API_KEY=<YOUR_KEY>  # For HuggingFace Inference Endpoints API
```

### Evaluation

#### Basic usage

```bash
python -m scripts.experiments.deontic --help
```

#### Deontic Logic Task

Example(s):

```bash
# Normative domain, zero-shot setting
python -m scripts.experiments.deontic deontic-base --test_n=all --model gpt-4o-mini gpt-4o

# All domains and settings
python -m scripts.experiments.deontic deontic-base deontic-kshot deontic-cot epistemic-base epistemic-kshot epistemic-cot --test_n=all --model gpt-4o-mini gpt-4o
```

#### Syllogistic Task

Example(s):

```bash
# Normative domain, zero-shot setting
python -m scripts.experiments.deontic deontic-mp-base --test_n=all --model gpt-4o-mini gpt-4o

# All domains and settings
python -m scripts.experiments.deontic deontic-mp-base deontic-mp-kshot deontic-mp-cot epistemic-mp-base epistemic-mp-kshot epistemic-mp-cot --test_n=all --model gpt-4o-mini gpt-4o
```

## Citation

TBA
