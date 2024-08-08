import argparse
import os
from itertools import product

import openai
import pandas as pd

from scripts.experiments.base import run_experiment_base
from scripts.experiments.choice5 import run_choice5
from scripts.experiments.kshot import run_3shot_symbol
from scripts.experiments.translate_and_reason import Experiment as TRExperiment
from scripts.experiments.translate_and_reason import (
    run_base,
    run_base_reasoning,
    run_fol,
    run_set,
)
from scripts.lib import Choice5Experiment
from scripts.lib import Experiment as BaseExperiment
from scripts.lib.utils import sample_balance


def translate_and_reason(
    test_n: int | str = "all",
    lang: list[str] = ["en"],
    model: list[str | dict] = ["gpt-4"],
    rootdir: str = "./",
    dry_run=False,
):
    # Load dataset
    df_baroco = pd.read_csv(
        "./data/acl2024/NeuBAROCO_TE90.tsv", delimiter="\t", dtype=str
    )

    # Prepare test data and data for few-shot examples
    if test_n == "all":
        df_test = df_baroco
    else:
        df_test = sample_balance(df_baroco, "content-type", 3, test_n)

    exp = TRExperiment(
        df_test=df_test, basedir="acl2024.translate_and_reason", rootdir=rootdir
    )

    for l, m in product(lang, model):
        if not isinstance(m, str):
            model_name = m["model_name"]
        else:
            model_name = m

        if dry_run:
            m = f"mock.{model_name}"

        run_base(exp, m, output_dir=f"{model_name}.base.{l}", lang=l)
        run_base_reasoning(
            exp, m, output_dir=f"{model_name}.base_reasoning.{l}", lang=l
        )
        run_fol(exp, m, output_dir=f"{model_name}.fol.{l}", lang=l)
        run_set(exp, m, output_dir=f"{model_name}.set.{l}", lang=l)

    # NOTE: 比較レポート出力
    exp.create_report_overall()

    # NOTE: コスト見積もり
    exp.estimate_cost(model="gpt-3.5-turbo")
    exp.estimate_cost(model="gpt-4")


def choice3(
    test_n: int | str = "all",
    lang: list[str] = ["en", "ja"],
    model: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    rootdir: str = "./",
    dry_run=False,
):
    # Load dataset
    df_baroco = pd.read_csv("./data/NeuBAROCO_NLI.tsv", delimiter="\t", dtype=str)

    # Prepare test data and data for few-shot examples
    if test_n == "all":
        df_test = df_baroco
    else:
        df_test = sample_balance(df_baroco, "gold", 3, test_n)

    # Set up experiment
    exp = BaseExperiment(df_test=df_test, basedir="acl2024.nli", rootdir=rootdir)

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

        # NOTE: ベースライン
        run_experiment_base(exp, m, output_dir=f"{model_name}.base.{l}", lang=l)

        # NOTE: k-shot
        run_3shot_symbol(exp, m, output_dir=f"{model_name}.3shot.{l}", lang=l)

    # NOTE: 比較レポート出力
    exp.create_report_overall()

    # NOTE: コスト見積もり
    exp.estimate_cost(model="gpt-3.5-turbo")
    exp.estimate_cost(model="gpt-4")


def choice5(
    test_n: int | str = "all",
    lang: list[str] = ["en", "ja"],
    model: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    rootdir="./",
    dry_run: bool = False,
):
    # Load dataset
    df_baroco = pd.read_csv(
        "./data/NeuBAROCO_MC.tsv", delimiter="\t", dtype=str
    )

    # Prepare test data
    if test_n == "all":
        df_test = df_baroco
    elif isinstance(test_n, int):
        df_test = df_baroco.sample(n=test_n)
    else:
        raise ValueError("test_n must be 'all' or int")

    # Set up experiment
    exp = Choice5Experiment(df_test=df_test, basedir="acl2024.choice5", rootdir=rootdir)

    for l, m in product(lang, model):
        if not isinstance(m, str):
            model_name = m["model_name"]
        else:
            model_name = m

        if dry_run:
            m = f"mock.{model_name}"

        run_choice5(exp, m, output_dir=f"{model_name}.choice5.{l}", lang=l)

    # Output comparison report
    exp.create_report_overall()

    # Estimate cost
    exp.estimate_cost(model="gpt-3.5-turbo")
    exp.estimate_cost(model="gpt-4")


def main(
    tasks=["translate_and_reason"],
    lang=["en"],
    test_n: int | str = "all",
    model=["gpt-4"],
    dry_run=False,
    rootdir="./",
):
    if "translate_and_reason" in tasks:
        translate_and_reason(test_n, lang, model, dry_run=dry_run, rootdir=rootdir)
    if "choice5" in tasks:
        choice5(test_n, lang, model, dry_run=dry_run, rootdir=rootdir)
    if "choice3" in tasks or "nli" in tasks:
        choice3(test_n, lang, model, dry_run=dry_run, rootdir=rootdir)


if __name__ == "__main__":
    # Use argparse to get arguments from command line.
    parser = argparse.ArgumentParser()

    # optional positional argument: tasks (choices: "choice3", separated by space)
    parser.add_argument(
        "tasks",
        nargs="*",
        default=None,
        choices=["choice3", "nli", "choice5", "translate_and_reason", []],
        help="tasks to run",
    )

    # option: lang (default: "en") (choices: "en", "ja", separated by space)
    parser.add_argument(
        "--lang",
        nargs="*",
        default=["en"],
        choices=["en", "ja"],
        help="languages to run",
    )

    # option: test_n (default: 12)
    parser.add_argument(
        "--test_n",
        default=12,
        help="'all' or number of test data to use (default: 12)",
    )

    # option: model (default: "gpt-3.5-turbo" and "gpt-4")
    parser.add_argument(
        "--model",
        nargs="*",
        default=["gpt-4"],
        choices=[
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-3.5-turbo-1106",
            "gpt-4-0613",
            "llama-2-13b-hf",
            "llama-2-70b-hf",
            "Swallow-13b-hf",
            "Swallow-70b-hf",
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

    tasks = args.tasks if args.tasks else ["translate_and_reason"]
    test_n = args.test_n if args.test_n == "all" else int(args.test_n)

    models = []

    for model in args.model:
        if model.startswith("Swallow-"):
            model = {
                "model_name": model,
                "model": "huggingface/tokyotech-llm/{model}",
                "api_base": args.api_base,
            }
        elif model.startswith("llama-"):
            model = {
                "model_name": model,
                "model": "huggingface/meta-llama/{model}",
                "api_base": args.api_base,
            }

        models.append(model)

    main(tasks, args.lang, test_n, models, dry_run=args.dry_run, rootdir=args.rootdir)
