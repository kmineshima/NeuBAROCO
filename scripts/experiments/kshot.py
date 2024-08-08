import os
from textwrap import dedent

import openai
import pandas as pd

from scripts.experiments.base import run_experiment_base
from scripts.lib import Experiment

# # openai-python v1: Disable httpx logging in openai-python:
# import logging
# logging.getLogger("httpx").setLevel(logging.WARNING)

FEW_SHOT_DEFAULT_PROMPT = {
    "en": dedent(
        """\
        Determine the correct logical relationship between the given premises and the hypothesis.
        - Answer "entailment" if the hypothesis follows logically from the premises.
        - Answer "contradiction" if the premises and the hypothesis are logically incompatible with each other.
        - Answer "neither" if the relationship is neither "entailment" nor "contradiction".
        Your answer must be one word: "entailment", "contradiction", or "neither".

        {few_shot_examples}"""
    ),
    "ja": dedent(
        """\
        与えられた前提と仮説の正しい論理的関係を判定しなさい。
        - 仮説が前提から論理的に導かれる場合は「含意」と答えなさい。
        - 前提と仮説が論理的に両立しない場合は「矛盾」と答えなさい。
        - その関係が「含意」でも「矛盾」でもない場合は「どちらでもない」と答えなさい。
        「含意」「矛盾」「どちらでもない」のいずれか一語で回答しなさい。

        {few_shot_examples}"""
    ),
}


def generate_few_shot_examples_en(df_inst):
    answers = {
        "entailment": "entailment",
        "neutral": "neither",
        "contradiction": "contradiction",
    }
    examples_text = ""

    for i, row in df_inst.iterrows():
        premises = row["premises_en"].split(". ")
        # print(premises)
        text = (
            f"Premise 1: {premises[0]}.\n"
            f"Premise 2: {premises[1]}\n"
            f"Hypothesis: {row['hypothesis_en']}\n"
            f"The answer is: {answers[row['gold']]}\n\n"
        )
        examples_text += text

    # Remove trailing line break
    examples_text = examples_text[:-1]

    return examples_text


def generate_few_shot_examples_ja(df_inst):
    answers = {
        "entailment": "含意",
        "neutral": "どちらでもない",
        "contradiction": "矛盾",
    }
    examples_text = ""

    for i, row in df_inst.iterrows():
        premises = row["premises_ja"].split("。")
        # print(premises)
        text = (
            f"前提1: {premises[0]}。\n"
            f"前提2: {premises[1]}。\n"
            f"仮説: {row['hypothesis_ja']}\n"
            f"答えは: {answers[row['gold']]}\n\n"
        )
        examples_text += text

    # Remove trailing line break
    examples_text = examples_text[:-1]

    return examples_text


def generate_few_shot_examples(df_inst, lang="en"):
    if lang == "en":
        return generate_few_shot_examples_en(df_inst)
    elif lang == "ja":
        return generate_few_shot_examples_ja(df_inst)


def run_3shot_symbol(
    exp: Experiment,
    model: str,
    prompt: str | None = None,
    output_dir: str = "3shot",
    lang: str = "en",
):
    prompt = prompt or FEW_SHOT_DEFAULT_PROMPT[lang]

    # 例示生成 (k-shot)
    if lang == "en":
        examples = dedent(
            """\
            Premise 1: Some X are Y.
            Premise 2: All Y are Z.
            Hypothesis: All X are Z.
            The answer is: neither

            Premise 1: Some X are Y.
            Premise 2: All Y are Z.
            Hypothesis: Some X are Z.
            The answer is: entailment

            Premise 1: Some X are Y.
            Premise 2: All Y are Z.
            Hypothesis: No X are Z.
            The answer is: contradiction
            """
        )
    elif lang == "ja":
        examples = dedent(
            """\
            前提1: あるXはYである。
            前提2: すべてのYはZである。
            仮説: すべてのXはZである。
            答えは: どちらでもない

            前提1: あるXはYである。
            前提2: すべてのYはZである。
            仮説: あるXはZである。
            答えは: 含意

            前提1: あるXはYである。
            前提2: すべてのYはZである。
            仮説: どのXもZでない。
            答えは: 矛盾
            """
        )

    # NOTE: 評価実行・レポート出力
    exp.eval_report(
        prompt,
        bind_vars={
            "few_shot_examples": examples,
        },
        output_dir=output_dir,
        model=model,
        lang=lang,
    )


def run_kshot(
    exp: Experiment,
    model: str,
    df_examples: pd.DataFrame,
    prompt: str | None = None,
    output_dir: str = "kshot",
    lang: str = "en",
):
    prompt = prompt or FEW_SHOT_DEFAULT_PROMPT[lang]

    # 例示生成 (k-shot)
    examples = generate_few_shot_examples(df_examples, lang=lang)

    # NOTE: 評価実行・レポート出力
    exp.eval_report(
        prompt,
        bind_vars={
            "few_shot_examples": examples,
        },
        output_dir=output_dir,
        model=model,
        lang=lang,
    )


def main():
    MODEL = "mock"
    # MODEL = "gpt-4"
    # MODEL = "gpt-4-1106-preview"  # gpt-4-turbo
    # MODEL = "gpt-3.5-turbo"

    df_baroco = pd.read_csv("./data/NeuBAROCO_NLI.tsv", delimiter="\t", dtype=str)
    # Exclude rows with empty "gold" values:
    df_baroco = df_baroco[df_baroco["gold"].notna()]

    df_test = df_baroco.iloc[3:].sample(n=30)
    df_inst = df_baroco.iloc[:3]

    exp = Experiment(df_test=df_test, basedir="kshot_en")

    run_experiment_base(exp, MODEL)
    run_kshot(exp, MODEL, df_inst, output_dir="3shot")

    # NOTE: 比較レポート出力
    exp.create_report_overall()

    # NOTE: コスト見積もり
    exp.estimate_cost(model="gpt-3.5-turbo")
    exp.estimate_cost(model="gpt-4")
    # exp.estimate_cost(model="gpt-4-1106-preview")


if __name__ == "__main__":
    main()
