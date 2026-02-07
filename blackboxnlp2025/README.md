# BlackboxNLP 2025 Dataset

Datasets and scripts for the following paper:

- ["Normative Reasoning in Large Language Models: A Comparative Benchmark from Logical and Modal Perspectives"](https://aclanthology.org/2025.blackboxnlp-1.17/) (BlackboxNLP 2025) ([arXiv](https://arxiv.org/abs/2510.26606))

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

```
@inproceedings{ozeki-etal-2025-normative,
    title = "Normative Reasoning in Large Language Models: A Comparative Benchmark from Logical and Modal Perspectives",
    author = "Ozeki, Kentaro  and
      Ando, Risako  and
      Morishita, Takanobu  and
      Abe, Hirohiko  and
      Mineshima, Koji  and
      Okada, Mitsuhiro",
    editor = "Belinkov, Yonatan  and
      Mueller, Aaron  and
      Kim, Najoung  and
      Mohebbi, Hosein  and
      Chen, Hanjie  and
      Arad, Dana  and
      Sarti, Gabriele",
    booktitle = "Proceedings of the 8th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.blackboxnlp-1.17/",
    doi = "10.18653/v1/2025.blackboxnlp-1.17",
    pages = "276--294",
    ISBN = "979-8-89176-346-3",
    abstract = "Normative reasoning is a type of reasoning that involves normative or deontic modality, such as obligation and permission. While large language models (LLMs) have demonstrated remarkable performance across various reasoning tasks, their ability to handle normative reasoning remains underexplored. In this paper, we systematically evaluate LLMs' reasoning capabilities in the normative domain from both logical and modal perspectives. Specifically, to assess how well LLMs reason with normative modals, we make a comparison between their reasoning with normative modals and their reasoning with epistemic modals, which share a common formal structure. To this end, we introduce a new dataset covering a wide range of formal patterns of reasoning in both normative and epistemic domains, while also incorporating non-formal cognitive factors that influence human reasoning. Our results indicate that, although LLMs generally adhere to valid reasoning patterns, they exhibit notable inconsistencies in specific types of normative reasoning and display cognitive biases similar to those observed in psychological studies of human reasoning. These findings highlight challenges in achieving logical consistency in LLMs' normative reasoning and provide insights for enhancing their reliability. All data and code are released publicly at https://github.com/kmineshima/NeuBAROCO."
}
```

## License

The datasets are licensed under Creative Commons Attribution 4.0 International.

[![CC4](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
