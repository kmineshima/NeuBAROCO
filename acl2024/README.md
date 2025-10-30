# NALOMA2023 & ACL2024 NeuBAROCO Datasets

Datasets and scripts for the following paper(s):

- ["Exploring Reasoning Biases in Large Language Models Through Syllogism: Insights from the NeuBAROCO Dataset"](https://aclanthology.org/2024.findings-acl.950/) (ACL2024 Findings) ([arXiv](https://arxiv.org/abs/2408.04403v1))
- ["Evaluating Large Language Models with NeuBAROCO: Syllogistic Reasoning Ability and Human-like Biases"](https://aclanthology.org/2023.naloma-1.1/) (NALOMA 2023) ([arXiv](https://arxiv.org/abs/2306.12567))

## Contents

- [Datasets](#datasets)
  - [NLI (Natural Language Inference) Task Format](#nli-natural-language-inference-task-format)
  - [Multiple-Choice Task Format](#multiple-choice-task-format)
  - [Data used in the NALOMA2023 experiments](#data-used-in-the-naloma2023-experiments)
- [Running scripts](#running-scripts)
  - [Setup](#setup)
  - [Set API keys](#set-api-keys)
  - [Evaluation](#evaluation)
- [Citation](#citation)

## Datasets

### NLI (Natural Language Inference) Task Format

#### File

[`NeuBAROCO_NLI.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/acl2024/NeuBAROCO_NLI.tsv)

#### Description

| Column Name | Description |
| ---- | ---- |
| ID | problem ID |
| ORIGINAL_ID | (INTERNAL) original problem ID |
| premises_ja | two premises in Japanese |
| hypothesis_ja | one hypothesis in Japanese |
| premises_en | two premises in English |
| hypothesis_en | one hypothesis in English |
| gold | correct answer, the relationship of the hypothesis to the premises (*entailment*, *contradiction*, *neutral*) |
| mood | the form of each premise and conclusion (three letters composed of A, E, I and O) |
| inference-type | type of logical inferences (*syllogism*, *propositional*) |
| content-type | classification based on belief congruency (*symbolic*, *congruent*, *incongruent*) |
| conversion | associated with conversion error (*yes*, *no*) |
| atmosphere | associated with atmosphere effect (*yes*, *no*) |

- See [our paper](#citation) for details on content-type, inference-type, conversion, and atmosphere.


### Multiple-Choice Task Format

#### File

[`NeuBAROCO_MC.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/acl2024/NeuBAROCO_MC.tsv)

#### Description

| Column Name | Description |
| ---- | ---- |
| ID | problem ID |
| premises_ja | two premises in Japanese |
| hypothesis_ja_1 | hypothesis 1 in Japanese |
| hypothesis_ja_2 | hypothesis 2 in Japanese |
| hypothesis_ja_3 | hypothesis 3 in Japanese |
| hypothesis_ja_4 | hypothesis 4 in Japanese |
| hypothesis_ja_5 | hypothesis 5 in Japanese |
| premises_en1 | two premises in English |
| hypothesis_en_1 | hypothesis 1 in English |
| hypothesis_en_2 | hypothesis 2 in English |
| hypothesis_en_3 | hypothesis 3 in English |
| hypothesis_en_4 | hypothesis 4 in English |
| hypothesis_en_5 | hypothesis 5 in English |
| gold | correct answer (1-5) |
| content-type | classification based on belief congruency (*symbolic*, *contentual*, *congruent*, *incongruent*) |
| mood | the form of each premise and conclusion (three letters composed of A, E, I and O) |
| figure | code for the order in which each term appears (1-4) |

- **NOTE:** One of the five hypotheses is "none of them".

### Data used in the NALOMA2023 experiments

#### File

[`NeuBAROCO_NALOMA.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/acl2024/NeuBAROCO_NALOMA.tsv)

- [Evaluating Large Language Models with NeuBAROCO: Syllogistic Reasoning Ability and Human-like Biases](https://aclanthology.org/2023.naloma-1.1) (Ando et al., NALOMA-WS 2023)

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

#### ACL2024 experiments

#### Basic usage

```bash
python -m scripts.experiments.acl2024 --help
```

#### NLI Task

Example:

```bash
python -m scripts.experiments.acl2024 nli --test_n=all --lang en ja --model gpt-3.5-turbo-1106 gpt-4-0613
```

#### Multiple-Choice Task

Example:

```bash
python -m scripts.experiments.acl2024 choice5 --test_n=all --lang en ja --model gpt-3.5-turbo-1106 gpt-4-0613
```

## Citation

If you use this data in any published research, please cite the following:

```
@inproceedings{ozeki-etal-2024-exploring,
    title = "Exploring Reasoning Biases in Large Language Models Through Syllogism: Insights from the {N}eu{BAROCO} Dataset",
    author = "Ozeki, Kentaro  and
      Ando, Risako  and
      Morishita, Takanobu  and
      Abe, Hirohiko  and
      Mineshima, Koji  and
      Okada, Mitsuhiro",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.950",
    pages = "16063--16077",
}
```

For the NALOMA2023 dataset:

```
@inproceedings{ando-etal-2023-evaluating,
    title = "Evaluating Large Language Models with {N}eu{BAROCO}: Syllogistic Reasoning Ability and Human-like Biases",
    author = "Ando, Risako  and
      Morishita, Takanobu  and
      Abe, Hirohiko  and
      Mineshima, Koji  and
      Okada, Mitsuhiro",
    editor = "Chatzikyriakidis, Stergios  and
      de Paiva, Valeria",
    booktitle = "Proceedings of the 4th Natural Logic Meets Machine Learning Workshop",
    month = jun,
    year = "2023",
    address = "Nancy, France",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.naloma-1.1/",
    pages = "1--11",
    abstract = "This paper investigates whether current large language models exhibit biases in logical reasoning, similar to humans. Specifically, we focus on syllogistic reasoning, a well-studied form of inference in the cognitive science of human deduction. To facilitate our analysis, we introduce a dataset called NeuBAROCO, originally designed for psychological experiments that assess human logical abilities in syllogistic reasoning. The dataset consists of syllogistic inferences in both English and Japanese. We examine three types of biases observed in human syllogistic reasoning: belief biases, conversion errors, and atmosphere effects. Our findings demonstrate that current large language models struggle more with problems involving these three types of biases."
}
```

## License

The datasets are licensed under Creative Commons Attribution 4.0 International.

[![CC4](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
