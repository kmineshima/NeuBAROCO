import argparse
import os
from itertools import product
from textwrap import dedent

import openai
import pandas as pd

from scripts.experiments.deontic.base import run_experiment_base
from scripts.lib.v2 import SinglePremiseExperiment as BaseExperiment
from scripts.lib.v2.utils import sample_balance

TASKS = {
    "deontic-base": "deontic_single_base",
    "epistemic-base": "epistemic_single_base",
    "deontic-kshot": "deontic_single_kshot",
    "epistemic-kshot": "epistemic_single_kshot",
    "deontic-cot": "deontic_single_cot",
    "epistemic-cot": "epistemic_single_cot",
    "deontic-mp-base": "deontic_multiple_base",
    "epistemic-mp-base": "epistemic_multiple_base",
    "deontic-mp-cot": "deontic_multiple_cot",
    "epistemic-mp-cot": "epistemic_multiple_cot",
    "deontic-mp-kshot": "deontic_multiple_kshot",
    "epistemic-mp-kshot": "epistemic_multiple_kshot",
}


def choice2_single(
    types: list[str] = [],
    test_n: int | str = "all",
    lang: list[str] = ["en"],
    model: list[str] = ["gpt-4o-mini", "gpt-4o"],
    *,
    rootdir: str = "./",
    dry_run=False,
):
    # Load datasets
    df_deontic_single = pd.read_csv(
        "./blackboxnlp2025/deontic_single.tsv", delimiter="\t", dtype=str
    )
    df_epistemic_single = pd.read_csv(
        "./blackboxnlp2025/epistemic_single.tsv", delimiter="\t", dtype=str
    )

    df_baroco = pd.concat([df_deontic_single, df_epistemic_single], ignore_index=True)

    # Prepare test data and data for few-shot examples
    if test_n == "all":
        df_test_all = df_baroco
    else:
        df_test_all = sample_balance(df_baroco, "content-type", 3, test_n)

    # Set up experiment
    exp = BaseExperiment(df_test=df_test_all, basedir="modal_sp", rootdir=rootdir)

    for type in types:
        modal, problem_type, prompt_type = type.split("_")
        # Filter only the rows with modal == type
        df_test = df_test_all[df_test_all["modal"] == modal]
        exp.df_test = df_test

        for l, m in product(lang, model):
            exp.interval = None
            exp.response_max_tokens = 10

            if not isinstance(m, str):
                model_name = m["model_name"]
                if m["model"].startswith("huggingface/"):
                    exp.interval = 0.01
            else:
                model_name = m

            if dry_run:
                m = f"mock.{model_name}"

            if prompt_type == "base":
                # NOTE: ベースラインv2
                prompt = {
                    "en": dedent(
                        """\
                        Determine whether the hypothesis follows from the premise(s).
                        - Answer 'entailment' if the hypothesis follows from the premise(s).
                        - Otherwise, answer 'non-entailment'.
                        Respond only with 'entailment' or 'non-entailment', and nothing else.
                        """
                    ),
                }

                run_experiment_base(
                    exp,
                    m,
                    output_dir=f"{model_name}.{modal}.base.{l}",
                    prompt=prompt[l],
                    lang=l,
                )

            elif prompt_type == "kshot":
                # NOTE: few-shot
                examples_deontic = [
                    # I-1. NotMu-MiNot
                    dedent(
                        """\
                        Premise: You are not required to finish homework by Friday.
                        Hypothesis: It is permissible not to finish homework by Friday.
                        Answer: entailment
                        """
                    ),
                    # I-2. NotMi-MuNot
                    dedent(
                        """\
                        Premise: It is not permissible to finish homework.
                        Hypothesis: You are required not to finish homework.
                        Answer: entailment
                        """
                    ),
                    # I-3. MiNot-NotMu
                    dedent(
                        """\
                        Premise: It is permissible not to finish homework by Friday.
                        Hypothesis: You are not required to finish homework by Friday.
                        Answer: entailment
                        """
                    ),
                    # I-4. Mu-Mi
                    dedent(
                        """\
                        Premise: You are required to finish homework.
                        Hypothesis: It is permissible to finish homework.
                        Answer: entailment
                        """
                    ),
                    # I-5. NotMi-NotMu
                    dedent(
                        """\
                        Premise: It is not permissible to finish homework.
                        Hypothesis: You are not required to finish homework.
                        Answer: entailment
                        """
                    ),
                    # II-1. NotMu-NotMi
                    dedent(
                        """\
                        Premise: You are not required to finish homework.
                        Hypothesis: It is not permissible to finish homework.
                        Answer: non-entailment
                        """
                    ),
                    # II-2. MiNot-MuNot
                    dedent(
                        """\
                        Premise: It is permissible not to finish homework.
                        Hypothesis: You are required not to finish homework.
                        Answer: non-entailment
                        """
                    ),
                    # II-3. Mi-Mu
                    dedent(
                        """\
                        Premise: It is permissible to finish homework.
                        Hypothesis: You are required to finish homework.
                        Answer: non-entailment
                        """
                    ),
                    # FC-Or-Elim
                    dedent(
                        """\
                        Premise: You may eat an apple or a pear.
                        Hypothesis: You may eat an apple.
                        Answer: entailment
                        """
                    ),
                    # FC-Or-Intro
                    dedent(
                        """\
                        Premise: You may eat an apple.
                        Hypothesis: You may eat an apple or a pear.
                        Answer: non-entailment
                        """
                    ),
                    # Ross-Or-Intro
                    dedent(
                        """\
                        Premise: It is obligatory to mail a letter.
                        Hypothesis: It is obligatory to mail a letter or to burn it.
                        Answer: non-entailment
                        """
                    ),
                ]

                examples_epistemic = [
                    # I-1. NotMu-MiNot
                    dedent(
                        """\
                        Premise: It is not certain that our team will win the game.
                        Hypothesis: It is possible that our team will not win the game.
                        Answer: entailment
                        """
                    ),
                    # I-2. NotMi-MuNot
                    dedent(
                        """\
                        Premise: It is not possible that our team will win the game.
                        Hypothesis: It is necessarily the case that our team will not win the game.
                        Answer: entailment
                        """
                    ),
                    # I-3. MiNot-NotMu
                    dedent(
                        """\
                        Premise: It is possible that our team will not win the game.
                        Hypothesis: It is not certain that our team will win the game.
                        Answer: entailment
                        """
                    ),
                    # I-4. Mu-Mi
                    dedent(
                        """\
                        Premise: It is certain that our team will win the game.
                        Hypothesis: It is possible that our team will win the game.
                        Answer: entailment
                        """
                    ),
                    # I-5. NotMi-NotMu
                    dedent(
                        """\
                        Premise: It is not possible that our team will win the game.
                        Hypothesis: It is not certain that our team will win the game.
                        Answer: entailment
                        """
                    ),
                    # II-1. NotMu-NotMi
                    dedent(
                        """\
                        Premise: It is not certain that our team will win the game.
                        Hypothesis: It is not possible that our team will win the game.
                        Answer: non-entailment
                        """
                    ),
                    # II-2. MiNot-MuNot
                    dedent(
                        """\
                        Premise: It is possible that our team will not win the game.
                        Hypothesis: It is certain that our team will not win the game.
                        Answer: non-entailment
                        """
                    ),
                    # II-3. Mi-Mu
                    dedent(
                        """\
                        Premise: It is possible that our team will win the game.
                        Hypothesis: It is certain that our team will win the game.
                        Answer: non-entailment
                        """
                    ),
                ]

                examples_deontic_str = "\n".join(examples_deontic)
                examples_epistemic_str = "\n".join(examples_epistemic)

                prompt = {
                    "en_deontic": dedent(
                        """\
                        Determine whether the hypothesis follows from the premise(s).
                        - Answer 'entailment' if the hypothesis follows from the premise(s).
                        - Otherwise, answer 'non-entailment'.
                        Respond only with 'entailment' or 'non-entailment', and nothing else.

                        """
                    )
                    + examples_deontic_str,
                    "en_epistemic": dedent(
                        """\
                        Determine whether the hypothesis follows from the premise(s).
                        - Answer 'entailment' if the hypothesis follows from the premise(s).
                        - Otherwise, answer 'non-entailment'.
                        Respond only with 'entailment' or 'non-entailment', and nothing else.
                        """
                    )
                    + examples_epistemic_str,
                }

                run_experiment_base(
                    exp,
                    m,
                    output_dir=f"{model_name}.{modal}.kshot.{l}",
                    prompt=prompt[l + "_" + modal],
                    lang=l,
                )

            elif prompt_type == "cot":
                prompt = {
                    "en": dedent(
                        """\
                        Determine whether the hypothesis follows from the premise(s).
                        - Answer 'entailment' if the hypothesis follows from the premise(s).
                        - Otherwise, answer 'non-entailment'.
                        Let's think step by step. Then output only one word, 'entailment' or 'non-entailment' on the last line, immediately after a line break.
                        """
                    ),
                }

                # exp.response_max_tokens = 512
                exp.response_max_tokens = max(exp.response_max_tokens, 1024)
                run_experiment_base(
                    exp,
                    m,
                    output_dir=f"{model_name}.{modal}.cot.{l}",
                    prompt=prompt[l],
                    lang=l,
                )

    # NOTE: Output comparison report
    exp.create_report_overall()


def choice2_multiple(
    types: list[str] = [],
    test_n: int | str = "all",
    lang: list[str] = ["en"],
    model: list[str] = ["gpt-4o-mini", "gpt-4o"],
    *,
    rootdir: str = "./",
    dry_run=False,
):
    # Load datasets
    df_deontic = pd.read_csv(
        "./blackboxnlp2025/deontic_multiple.tsv", delimiter="\t", dtype=str
    )
    df_epistemic = pd.read_csv(
        "./blackboxnlp2025/epistemic_multiple.tsv", delimiter="\t", dtype=str
    )

    df_baroco = pd.concat([df_deontic, df_epistemic], ignore_index=True)

    # Prepare test data and data for few-shot examples
    if test_n == "all":
        df_test_all = df_baroco
    else:
        df_test_all = sample_balance(df_baroco, "content-type", 3, test_n)

    # Set up experiment
    exp = BaseExperiment(df_test=df_test_all, basedir="modal_mp", rootdir=rootdir)

    for type in types:
        modal, problem_type, prompt_type = type.split("_")
        # Filter only the rows with modal == type
        df_test = df_test_all[df_test_all["modal"] == modal]
        exp.df_test = df_test

        for l, m in product(lang, model):
            exp.interval = None

            if not isinstance(m, str):
                model_name = m["model_name"]
                if m["model"].startswith("huggingface/"):
                    exp.interval = 0.01
            else:
                model_name = m

            if dry_run:
                m = f"mock.{model_name}"

            if prompt_type == "base":
                # NOTE: ベースラインv2
                prompt = {
                    "en": dedent(
                        """\
                        Determine whether the hypothesis follows from the premise(s).
                        - Answer 'entailment' if the hypothesis follows from the premise(s).
                        - Otherwise, answer 'non-entailment'.
                        Respond only with 'entailment' or 'non-entailment', and nothing else.
                        """
                    ),
                }

                run_experiment_base(
                    exp,
                    m,
                    output_dir=f"{model_name}.{modal}.base.{l}",
                    prompt=prompt[l],
                    lang=l,
                )

            elif prompt_type == "cot":
                prompt = {
                    "en": dedent(
                        """\
                        Determine whether the hypothesis follows from the premise(s).
                        - Answer 'entailment' if the hypothesis follows from the premise(s).
                        - Otherwise, answer 'non-entailment'.
                        Let's think step by step. Then output only one word, 'entailment' or 'non-entailment' on the last line, immediately after a line break.
                        """
                    ),
                }

                # exp.response_max_tokens = 512
                exp.response_max_tokens = max(exp.response_max_tokens, 1024)
                run_experiment_base(
                    exp,
                    m,
                    output_dir=f"{model_name}.{modal}.cot.{l}",
                    prompt=prompt[l],
                    lang=l,
                )

            elif prompt_type == "kshot":
                # NOTE: few-shot
                examples_deontic = [
                    # III-1. Hyp-MP
                    dedent(
                        """\
                        Premise: If the traffic light is red, then Taro must stop the car. The traffic light is red.
                        Hypothesis: Taro must stop the car.
                        Answer: entailment
                        """
                    ),
                    # III-2. Hyp-MT
                    dedent(
                        """\
                        Premise: If the traffic light is red, then Taro must stop the car. It is not the case that Taro must stop the car.
                        Hypothesis: The traffic light is not red.
                        Answer: entailment
                        """
                    ),
                    # III-3. Cat-MP
                    dedent(
                        """\
                        Premise: All first-year students at this university must submit a paper. John is a first-year student at this university.
                        Hypothesis: John must submit a paper.
                        Answer: entailment
                        """
                    ),
                    # III-4. Cat-MT
                    dedent(
                        """\
                        Premise: All first-year students at this university must submit a paper. Bob is not required to submit a paper.
                        Hypothesis: Bob is not a first-year student at this university.
                        Answer: entailment
                        """
                    ),
                    # IV-1. Hyp-AC
                    dedent(
                        """\
                        Premise: If the traffic light is red, then Taro must stop the car. Taro must stop the car.
                        Hypothesis: The traffic light is red.
                        Answer: non-entailment
                        """
                    ),
                    # IV-2. Hyp-DA
                    dedent(
                        """\
                        Premise: If the traffic light is red, then Taro must stop the car. The traffic light is not red.
                        Hypothesis: It is not the case that Taro must stop the car.
                        Answer: non-entailment
                        """
                    ),
                    # IV-3. Cat-AC
                    dedent(
                        """\
                        Premise: All first-year students at this university must submit a paper. John must submit a paper.
                        Hypothesis: John is a first-year student at this university.
                        Answer: non-entailment
                        """
                    ),
                    # IV-4. Cat-DA
                    dedent(
                        """\
                        Premise: All first-year students at this university must submit a paper. Bob is not a first-year student at this university.
                        Hypothesis: Bob is not required to submit a paper.
                        Answer: non-entailment
                        """
                    ),
                ]

                examples_epistemic = [
                    # III-1. Hyp-MP
                    dedent(
                        """\
                        Premise: If the artifact is made of gold, then it is certain that it will be valuable. The artifact is made of gold.
                        Hypothesis: The artifact must have been valuable.
                        Answer: entailment
                        """
                    ),
                    # III-2. Hyp-MT
                    dedent(
                        """\
                        Premise: If the artifact is made of gold, then it is certain that it will be valuable. There is a possibility that the artifact is not valuable.
                        Hypothesis: The artifact is not made of gold.
                        Answer: entailment
                        """
                    ),
                    # III-3. Cat-MP
                    dedent(
                        """\
                        Premise: All first-year students at this university must have been in the classroom. John is a first-year student at this university.
                        Hypothesis: John must have been in the classroom.
                        Answer: entailment
                        """
                    ),
                    # III-4. Cat-MT
                    dedent(
                        """\
                        Premise: All first-year students at this university must have been in the classroom. It is not the case that John must have been in the classroom.
                        Hypothesis: John is not a first-year student at this university.
                        Answer: entailment
                        """
                    ),
                    # IV-1. Hyp-AC
                    dedent(
                        """\
                        Premise: All first-year students at this university must have been in the classroom. John is not a first-year student at this university.
                        Hypothesis: It is not the case that John must have been in the classroom.
                        Answer: non-entailment
                        """
                    ),
                    # IV-2. Hyp-DA
                    dedent(
                        """\
                        Premise: If the artifact is made of gold, then it is certain that it will be valuable. The artifact is not made of gold.
                        Hypothesis: It is not certain that the artifact will be valuable.
                        Answer: non-entailment
                        """
                    ),
                    # IV-3. Cat-AC
                    dedent(
                        """\
                        Premise: All first-year students at this university must have been in the classroom. John must have been in the classroom.
                        Hypothesis: John is a first-year student at this university.
                        Answer: non-entailment
                        """
                    ),
                    # IV-4. Cat-DA
                    dedent(
                        """\
                        Premise: All first-year students at this university must have been in the classroom. John is not a first-year student at this university.
                        Hypothesis: It is not the case that John must have been in the classroom.
                        Answer: non-entailment
                        """
                    ),
                ]

                examples_deontic_str = "\n".join(examples_deontic)
                examples_epistemic_str = "\n".join(examples_epistemic)

                prompt = {
                    "en_deontic": dedent(
                        """\
                        Determine whether the hypothesis follows from the premise(s).
                        - Answer 'entailment' if the hypothesis follows from the premise(s).
                        - Otherwise, answer 'non-entailment'.
                        Respond only with 'entailment' or 'non-entailment', and nothing else.

                        """
                    )
                    + examples_deontic_str,
                    "en_epistemic": dedent(
                        """\
                        Determine whether the hypothesis follows from the premise(s).
                        - Answer 'entailment' if the hypothesis follows from the premise(s).
                        - Otherwise, answer 'non-entailment'.
                        Respond only with 'entailment' or 'non-entailment', and nothing else.

                        """
                    )
                    + examples_epistemic_str,
                }

                run_experiment_base(
                    exp,
                    m,
                    output_dir=f"{model_name}.{modal}.kshot.{l}",
                    prompt=prompt[l + "_" + modal],
                    lang=l,
                )

    # NOTE: Output comparison report
    exp.create_report_overall()


def main(
    tasks=list(TASKS.keys()),
    lang=["en"],
    test_n: int | str = "all",
    model=["gpt-4o"],
    dry_run=False,
    rootdir="./",
):
    openai.api_key = os.getenv("API_KEY")

    single_tasks = [TASKS[task] for task in tasks if "mp-" not in task]
    if single_tasks:
        choice2_single(
            single_tasks,
            test_n,
            lang,
            model,
            dry_run=dry_run,
            rootdir=rootdir,
        )

    multiple_tasks = [TASKS[task] for task in tasks if "mp-" in task]

    if multiple_tasks:
        choice2_multiple(
            multiple_tasks,
            test_n,
            lang,
            model,
            dry_run=dry_run,
            rootdir=rootdir,
        )


if __name__ == "__main__":
    # Use argparse to get arguments from command line.
    parser = argparse.ArgumentParser()

    # optional positional argument: tasks (separated by space)
    parser.add_argument(
        "tasks",
        nargs="*",
        default=None,
        choices=list(TASKS.keys()) + [[]],
        help="tasks to run",
    )

    # option: lang (default: "en")
    parser.add_argument(
        "--lang",
        nargs="*",
        default=["en"],
        choices=["en"],
        help="languages to run",
    )

    # option: test_n (default: 12)
    parser.add_argument(
        "--test_n",
        default=12,
        help="'all' or number of test data to use (default: 12)",
    )

    # option: model (default "gpt-4o")
    parser.add_argument(
        "--model",
        nargs="*",
        default=["gpt-4o"],
        choices=[
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "o1-preview",
            "o1-mini",
            "gpt-3.5-turbo-1106",
            "gpt-4-0613",
            "llama-2-13b-hf",
            "llama-2-70b-hf",
            "llama-3-1-8b-instruct",
            "llama-3-1-70b-instruct",
            "llama-3-3-70b-instruct",
            "Swallow-13b-hf",
            "Swallow-70b-hf",
            "phi-3-mini-4k-instruct",
            "phi-3-small-8k-instruct",
            "phi-3-medium-4k-instruct",
            "phi-4",
            "qwen2-5-72b-instruct",
            "gpt-oss-20b",
            "gpt-oss-120b",
        ],
        help="models to run",
    )

    # option: api-base (default: None)
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for the model",
    )

    # option: rootdir (default: "./")
    parser.add_argument(
        "--rootdir",
        default="./",
        help="root directory for experiment directories",
    )

    # option: dry-run (default: False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="dry run",
    )

    args = parser.parse_args()

    tasks = args.tasks if args.tasks else list(TASKS.keys())
    test_n = args.test_n if args.test_n == "all" else int(args.test_n)

    models = []

    for model in args.model:
        if model.startswith("Swallow-"):
            model = {
                "model_name": model,
                "model": f"huggingface/tokyotech-llm/{model}",
                "api_base": args.api_base,
            }
        elif model.startswith("llama-"):
            model = {
                "model_name": model,
                "model": f"huggingface/meta-llama/{model}",
                "api_base": args.api_base,
            }
        elif model.startswith("qwen"):
            model = {
                "model_name": model,
                "model": f"huggingface/qwen/{model}",
                "api_base": args.api_base,
            }
        elif model.startswith("deepseek-"):
            model = {
                "model_name": model,
                "model": f"huggingface/deepseek-ai/{model}",
                "api_base": args.api_base,
            }
        elif model.startswith("gpt-oss"):
            model = {
                "model_name": model,
                "model": f"huggingface/openai/{model}",
                "api_base": args.api_base,
            }
        elif model.startswith("phi-"):
            model = {
                "model_name": model,
                "model": f"huggingface/microsoft/{model}",
                "api_base": args.api_base,
            }

        models.append(model)

    main(
        tasks,
        args.lang,
        test_n,
        models,
        dry_run=args.dry_run,
        rootdir=args.rootdir,
    )
