# EACL 2026 Dataset

Dataset for the following paper:

- "Evaluation of Deontic Conditional Reasoning in Large Language Models: The Case of Wason's Selection Task" (accepted to EACL2026 Main)

## Contents

- [Dataset](#dataset)
  - [Wason Selection Task Format](#wason-selection-task-format)
- [Citation](#citation)

## Dataset

### Wason Selection Task Format

#### Files

- [`wason.tsv`](https://github.com/kmineshima/NeuBAROCO/blob/main/eacl2026/wason.tsv) - Deontic and epistemic Wason selection task problems

#### Description

| Column Name | Description |
| ---- | ---- |
| ID | problem ID |
| modal | modality of the rule (*deontic*, *epistemic*) |
| form | polarity pattern of the rule (*pos-pos*, *pos-neg*, *neg-pos*, *neg-neg*) |
| rule | conditional rule in English |
| card-1 | first card (front or back) in English |
| card-2 | second card (front or back) in English |
| card-3 | third card (front or back) in English |
| card-4 | fourth card (front or back) in English |
| gold1 | correct answer 1, one of the four cards that should be turned over (*1, 2, 3, 4*) |
| gold2 | correct answer 2, another one of the four cards that should be turned over (*1, 2, 3, 4*) |

- See [our paper](#citation) for details on the dataset.

## Citation

TBA

## License

The datasets are licensed under Creative Commons Attribution 4.0 International.

[![CC4](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
