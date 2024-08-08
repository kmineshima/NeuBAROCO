import os
from textwrap import dedent

import openai
import pandas as pd

from scripts.lib import Experiment
from scripts.lib.utils import sample_balance

# # openai-python v1: Disable httpx logging in openai-python:
# import logging
# logging.getLogger("httpx").setLevel(logging.WARNING)


def run_experiment_base(
    exp: Experiment,
    model: str,
    prompt: str | None = None,
    output_dir: str = "base",
    lang: str = "en",
):
    default_prompt = {
        "en": dedent(
            """\
            Determine the correct logical relationship between the given premises and the hypothesis.
            - Answer "entailment" if the hypothesis follows logically from the premises.
            - Answer "contradiction" if the premises and the hypothesis are logically incompatible with each other.
            - Answer "neither" if the relationship is neither "entailment" nor "contradiction".
            Your answer must be one word: "entailment", "contradiction", or "neither".
            """
        ),
        "ja": dedent(
            """\
            与えられた前提と仮説の正しい論理的関係を判定しなさい。
            - 仮説が前提から論理的に導かれる場合は「含意」と答えなさい。
            - 前提と仮説が論理的に両立しない場合は「矛盾」と答えなさい。
            - その関係が「含意」でも「矛盾」でもない場合は「どちらでもない」と答えなさい。
            「含意」「矛盾」「どちらでもない」のいずれか一語で回答しなさい。
            """
        ),
    }

    prompt = prompt or default_prompt[lang]

    # NOTE: 評価実行・レポート出力
    exp.eval_report(
        prompt,
        output_dir=output_dir,
        model=model,
        lang=lang,
    )


def main():
    MODEL = "mock"
    # MODEL = "gpt-4"
    # MODEL = "gpt-4-1106-preview"  # gpt-4-turbo
    # MODEL = "gpt-3.5-turbo"
    # MODEL = {"model": "ollama/phi", "api_base": "http://localhost:11434"}
    # MODEL = {
    #     "model": "huggingface/tokyotech-llm/Swallow-13b-hf",
    #     "api_base": "https://ID.us-east-1.aws.endpoints.huggingface.cloud",
    # }

    df_baroco = pd.read_csv("./data/NeuBAROCO_NLI.tsv", delimiter="\t", dtype=str)
    # Exclude rows with empty "gold" values:
    df_baroco = df_baroco[df_baroco["gold"].notna()]

    df_test = sample_balance(df_baroco, "gold", 3, 30)

    exp = Experiment(df_test=df_test, basedir="base")

    run_experiment_base(exp, MODEL)

    # NOTE: 比較レポート出力
    exp.create_report_overall()

    # NOTE: コスト見積もり
    exp.estimate_cost(model="gpt-3.5-turbo")
    exp.estimate_cost(model="gpt-4")
    # exp.estimate_cost(model="gpt-4-1106-preview")


if __name__ == "__main__":
    main()
