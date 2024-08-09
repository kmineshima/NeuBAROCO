# NeuBAROCO

Datasets and scripts for the ACL2024 Findings paper: "Exploring Reasoning Biases in Large Language Models Through Syllogism: Insights from the NeuBAROCO Dataset".

## Contents

- [Datasets](#datasets)
  - [NLI (Natural Language Inference) Task Format](#nli-natural-language-inference-task-format)
  - [Multiple-Choice Task Format](#multiple-choice-task-format)
  - [Data used in the NALOMA2023 experiments](#data-used-in-the-naloma2023-experiments)
- [Running scripts](#running-scripts)
- [Citation](#citation)

## Datasets

### NLI (Natural Language Inference) Task Format

#### File

[`data/NeuBAROCO_NLI.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/data/NeuBAROCO_NLI.tsv)

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

[`data/NeuBAROCO_MC.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/data/NeuBAROCO_MC.tsv)

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

[`data/naloma2023/NeuBAROCO_NALOMA.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/data/naloma2023/NeuBAROCO_NALOMA.tsv)

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

- ACL Anthology: TBA
- [arXiv preprint](https://arxiv.org/abs/2408.04403v1)

```
@article{ozeki2024exploring,
  title={Exploring Reasoning Biases in Large Language Models Through Syllogism: Insights from the NeuBAROCO Dataset},
  author={Kentaro Ozeki and Risako Ando and Takanobu Morishita and Hirohiko Abe and Koji Mineshima and Mitsuhiro Okada},
  journal={arXiv preprint arXiv:2408.04403},
  year={2024}
}
```
