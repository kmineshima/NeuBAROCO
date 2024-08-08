import os
from textwrap import dedent

import openai
import pandas as pd

from scripts.lib import Choice5Experiment


def run_choice5(exp, model, prompt: str | None = None, output_dir="choice5", lang="en"):
    default_prompt = {
        "en": dedent(
            """\
            Select one statement from the five options provided that logically follows as a conclusion from the two premises presented in each problem. Answer by providing the number of your choice.
            """
        ),
        "ja": dedent(
            """\
            各問題にある2つの前提の結論として成り立つ文を、5つの選択肢の中から1つだけ選んでください。番号で回答してください。
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
    # MODEL = "gpt-3.5-turbo"
    # MODEL = "gpt-4-1106-preview"  # gpt-4-turbo
    # MODEL = "gpt-4"
    MODEL = "mock"  # APIを使わず、常にentailmentを返す (テスト用)

    df_baroco = pd.read_csv("./data/NeuBAROCO_2.2.tsv", delimiter="\t")

    # 評価用データ
    df_sample = df_baroco.sample(n=30)

    exp = Choice5Experiment(df_test=df_sample, basedir="exp_choice5")

    run_choice5(exp, MODEL)

    # NOTE: 比較レポート出力
    exp.create_report_overall()

    # NOTE: コスト見積もり
    exp.estimate_cost(model="gpt-3.5-turbo")
    exp.estimate_cost(model="gpt-4")


if __name__ == "__main__":
    main()
