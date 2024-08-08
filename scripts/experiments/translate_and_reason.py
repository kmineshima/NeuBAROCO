import os
from textwrap import dedent
from typing import Dict

import openai
import pandas as pd

from scripts.lib import Experiment as BaseExperiment
from scripts.lib.utils import sample_balance

# # openai-python v1: Disable httpx logging in openai-python:
# import logging
# logging.getLogger("httpx").setLevel(logging.WARNING)


class Experiment(BaseExperiment):
    _mock_response = "..."
    _response_max_tokens = 2048

    def make_prompt(
        self,
        prompt: str,
        pr1: str,
        pr2: str,
        con: str,
        bind_vars: Dict[str, str] = {},
        lang: str = "en",
    ):
        prompt = prompt.format(**bind_vars, problem="{problem}")

        if lang == "en":
            problem = (
                f"## Input\nPremise 1: {pr1}.\n"
                + f"Premise 2: {pr2}.\n"
                + f"Hypothesis: {con}"
            )
        elif lang == "ja":
            problem = (
                f"## 入力\n前提1: {pr1}。\n" + f"前提2: {pr2}。\n" + f"仮説: {con}"
            )

        # NOTE: Use "{problem}" as a special placeholder for the problem
        res = prompt.format(problem=problem)

        return res

    def put_answer(self, exp):
        assert isinstance(exp, str)

        ans = exp.strip().split("\n")[-1].lower()
        ans = ans.split("。")[0]

        if ans in self.ents + self.ents_ja:
            res = "entailment"
        elif ans in self.conts + self.conts_ja:
            res = "contradiction"
        elif ans in self.neutrals + self.neutrals_ja:
            res = "neutral"
        else:
            res = "check"

        return res


def _run_cot(
    exp: Experiment,
    model: str,
    examples: str,
    output_dir: str,
    response_start_heading: str,
    lang: str = "en",
):
    PROMPT = {
        "en": dedent(
            """\
        ## Task
        Determine the correct logical relationship between the given premises and the hypothesis.
        - Answer "entailment" if the hypothesis follows logically from the premises.
        - Answer "contradiction" if the premises and the hypothesis are logically incompatible with each other.
        - Answer "neither" if the relationship is neither "entailment" nor "contradiction".

        {few_shot_examples}

        {problem}

        ## {response_start_heading}"""
        ),
        "ja": dedent(
            """\
        与えられた前提と仮説の正しい論理的関係を判定しなさい。
        - 仮説が前提から論理的に導かれる場合は「含意」と答えなさい。
        - 前提と仮説が論理的に両立しない場合は「矛盾」と答えなさい。
        - その関係が「含意」でも「矛盾」でもない場合は「どちらでもない」と答えなさい。

        {few_shot_examples}

        {problem}

        ## {response_start_heading}"""
        ),
    }

    prompt = PROMPT[lang]

    # NOTE: 評価実行・レポート出力
    exp.eval_report(
        prompt,
        bind_vars={
            "few_shot_examples": examples,
            "response_start_heading": response_start_heading,
        },
        output_dir=output_dir,
        model=model,
        lang=lang,
    )


def run_base_0shot(
    exp: Experiment,
    model: str,
    output_dir: str,
    lang: str = "en",
):
    PROMPT = {
        "en": dedent(
            """\
        ## Task
        Determine the correct logical relationship between the given premises and the hypothesis.
        - Answer "entailment" if the hypothesis follows logically from the premises.
        - Answer "contradiction" if the premises and the hypothesis are logically incompatible with each other.
        - Answer "neither" if the relationship is neither "entailment" nor "contradiction".

        {problem}

        ## Answer"""
        ),
        "ja": dedent(
            """\
        与えられた前提と仮説の正しい論理的関係を判定しなさい。
        - 仮説が前提から論理的に導かれる場合は「含意」と答えなさい。
        - 前提と仮説が論理的に両立しない場合は「矛盾」と答えなさい。
        - その関係が「含意」でも「矛盾」でもない場合は「どちらでもない」と答えなさい。

        {problem}

        ## 答え"""
        ),
    }

    prompt = PROMPT[lang]

    # NOTE: 評価実行・レポート出力
    exp.eval_report(
        prompt,
        bind_vars={},
        output_dir=output_dir,
        model=model,
        lang=lang,
    )


def run_base(
    exp: Experiment,
    model: str,
    output_dir: str,
    lang: str = "en",
):
    if lang == "en":
        examples = dedent(
            """\
            ## Input
            Premise 1: Some A are B.
            Premise 2: All B are C.
            Hypothesis: All A are C.

            ## Answer
            [Your answer must be one word: "entailment", "contradiction", or "neither"]"""
        )
        response_start_heading = "Answer"
    elif lang == "ja":
        examples = dedent(
            """\
            ## 入力
            前提1: あるAはBである。
            前提2: すべてのBはCである。
            仮説: すべてのAはCである。

            ## 答え
            [「含意」「矛盾」「どちらでもない」のいずれか一語で回答しなさい]"""
        )
        response_start_heading = "答え"

    print(
        exp,
        model,
        examples,
        output_dir,
        lang,
    )

    _run_cot(
        exp,
        model,
        examples,
        output_dir,
        response_start_heading=response_start_heading,
        lang=lang,
    )


def run_base_reasoning(
    exp: Experiment,
    model: str,
    output_dir: str = "base_reasoning",
    lang: str = "en",
):
    if lang == "en":
        examples = dedent(
            """\
            ## Input
            Premise 1: Some A are B.
            Premise 2: All B are C.
            Hypothesis: All A are C.

            ## Reasoning
            [Explain your reasoning for the answer]

            ## Answer
            [Your answer must be one word: "entailment", "contradiction", or "neither"]"""
        )
        response_start_heading = "Reasoning"
    elif lang == "ja":
        examples = dedent(
            """\
            ## 入力
            前提1: あるAはBである。
            前提2: すべてのBはCである。
            仮説: すべてのAはCである。

            ## 論証
            [答えを導く論証を説明しなさい]

            ## 答え
            [「含意」「矛盾」「どちらでもない」のいずれか一語で回答しなさい]"""
        )
        response_start_heading = "論証"

    _run_cot(
        exp,
        model,
        examples,
        output_dir,
        response_start_heading=response_start_heading,
        lang=lang,
    )


def run_fol(
    exp: Experiment,
    model: str,
    output_dir: str = "fol",
    lang: str = "en",
):
    if lang == "en":
        examples = dedent(
            """\
            ## Input
            Premise 1: Some A are B.
            Premise 2: All B are C.
            Hypothesis: All A are C.

            ## Translation to predicate logic
            Premise 1: ∃x(Ax∧Bx)
            Premise 2: ∀x(Bx→Cx)
            Hypothesis: ∀x(Ax→Cx)

            ## Reasoning
            [Explain your reasoning for the answer]

            ## Answer
            [Your answer must be one word: "entailment", "contradiction", or "neither"]"""
        )
        response_start_heading = "Translation to predicate logic"
    elif lang == "ja":
        examples = dedent(
            """\
            ## 入力
            前提1: あるAはBである。
            前提2: すべてのBはCである。
            仮説: すべてのAはCである。

            ## 述語論理への翻訳
            前提1: ∃x(Ax∧Bx)
            前提2: ∀x(Bx→Cx)
            仮説: ∀x(Ax→Cx)

            ## 論証
            [答えを導く論証を説明しなさい]

            ## 答え
            [「含意」「矛盾」「どちらでもない」のいずれか一語で回答しなさい]"""
        )
        response_start_heading = "述語論理への翻訳"

    _run_cot(
        exp,
        model,
        examples,
        output_dir,
        response_start_heading=response_start_heading,
        lang=lang,
    )


def run_set(
    exp: Experiment,
    model: str,
    output_dir: str = "set",
    lang: str = "en",
):
    if lang == "en":
        examples = dedent(
            """\
            ## Input
            Premise 1: Some A are B.
            Premise 2: All B are C.
            Hypothesis: All A are C.

            ## Translation to set theory
            Premise 1: The set of As overlaps with the set of Bs.
            Premise 2: The set of Bs is a subset of the set of Cs.
            Hypothesis: The set of As is a subset of the set of Cs.

            ## Reasoning
            [Explain your reasoning for the answer]

            ## Answer
            [Your answer must be one word: "entailment", "contradiction", or "neither"]"""
        )
        response_start_heading = "Translation to set theory"
    elif lang == "ja":
        examples = dedent(
            """\
            ## 入力
            前提1: あるAはBである。
            前提2: すべてのBはCである。
            仮説: すべてのAはCである。

            ## 集合論への翻訳
            前提1: Aの集合とBの集合は共通部分を持つ。
            前提2: Bの集合はCの集合の部分集合である。
            仮説: Aの集合はCの集合の部分集合である。

            ## 論証
            [答えを導く論証を説明しなさい]

            ## 答え
            [「含意」「矛盾」「どちらでもない」のいずれか一語で回答しなさい]"""
        )
        response_start_heading = "集合論への翻訳"

    _run_cot(
        exp,
        model,
        examples,
        output_dir,
        response_start_heading=response_start_heading,
        lang=lang,
    )


def run_fol_plus_set(
    exp: Experiment,
    model: str,
    output_dir: str = "fol+set",
    lang: str = "en",
):
    if lang == "en":
        examples = dedent(
            """\
            ## Input
            Premise 1: Some A are B.
            Premise 2: All B are C.
            Hypothesis: All A are C.

            ## Translation to predicate logic
            Premise 1: ∃x(Ax∧Bx)
            Premise 2: ∀x(Bx→Cx)
            Hypothesis: ∀x(Ax→Cx)

            ## Translation to set theory
            Premise 1: The set of As overlaps with the set of Bs.
            Premise 2: The set of Bs is a subset of the set of Cs.
            Hypothesis: The set of As is a subset of the set of Cs.

            ## Reasoning
            [Explain your reasoning for the answer]

            ## Answer
            [Your answer must be one word: "entailment", "contradiction", or "neither"]"""
        )
        response_start_heading = "Translation to predicate logic"
    elif lang == "ja":
        examples = dedent(
            """\
            ## 入力
            前提1: あるAはBである。
            前提2: すべてのBはCである。
            仮説: すべてのAはCである。

            ## 述語論理への翻訳
            前提1: ∃x(Ax∧Bx)
            前提2: ∀x(Bx→Cx)
            仮説: ∀x(Ax→Cx)

            ## 集合論への翻訳
            前提1: Aの集合とBの集合は共通部分を持つ。
            前提2: Bの集合はCの集合の部分集合である。
            仮説: Aの集合はCの集合の部分集合である。

            ## 論証
            [答えを導く論証を説明しなさい]

            ## 答え
            [「含意」「矛盾」「どちらでもない」のいずれか一語で回答しなさい]"""
        )
        response_start_heading = "述語論理への翻訳"

    _run_cot(
        exp,
        model,
        examples,
        output_dir,
        response_start_heading=response_start_heading,
        lang=lang,
    )


def main():
    # MODEL = "mock"
    MODEL = "gpt-4"
    # MODEL = "gpt-4-1106-preview"  # gpt-4-turbo
    # MODEL = "gpt-3.5-turbo"

    df_baroco = pd.read_csv("./data/NeuBAROCO_NLI.tsv", delimiter="\t", dtype=str)
    # Exclude rows with empty "gold" values:
    df_baroco = df_baroco[df_baroco["gold"].notna()]
    # Exclude rows whose content-type values are "others":
    df_baroco = df_baroco[df_baroco["content-type"] != "others"]

    df_test = sample_balance(
        df_baroco[df_baroco["source"] == "all"], "content-type", 3, 30
    )

    # exp_base = BaseExperiment(df_test=df_test, basedir="translate_and_reason")
    exp = Experiment(df_test=df_test, basedir="translate_and_reason")

    # run_experiment_base(exp_base, MODEL)
    run_base(exp, MODEL, output_dir="base")
    run_base_reasoning(exp, MODEL, output_dir="base_reasoning")
    run_fol(exp, MODEL, output_dir="fol")
    run_set(exp, MODEL, output_dir="set")
    # run_fol_plus_set(exp, MODEL, output_dir="fol+set")

    # NOTE: 比較レポート出力
    exp.create_report_overall()

    # NOTE: コスト見積もり
    exp.estimate_cost(model="gpt-3.5-turbo")
    exp.estimate_cost(model="gpt-4")
    # exp.estimate_cost(model="gpt-4-1106-preview")


if __name__ == "__main__":
    main()
