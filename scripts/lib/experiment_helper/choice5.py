from typing import Dict

import pandas as pd

from scripts.lib.experiment_helper.base import Experiment


class Choice5Experiment(Experiment):
    def __init__(self, df_test, **kwargs):
        super().__init__(df_test, **kwargs)
        self.mock_response = "1"

    def predict(
        self,
        df: pd.DataFrame,
        prompt: str,
        bind_vars: Dict[str, str] = {},
        lang: str = "en",
        model: str | dict = "gpt-3.5-turbo",
        include_logprobs: bool = True,
    ):
        if lang == "en":
            premises = df["premises_en"]
            conclusion1 = df["hypothesis_en_1"]
            conclusion2 = df["hypothesis_en_2"]
            conclusion3 = df["hypothesis_en_3"]
            conclusion4 = df["hypothesis_en_4"]
            conclusion5 = df["hypothesis_en_5"]
        elif lang == "ja":
            premises = df["premises_ja"]
            conclusion1 = df["hypothesis_ja_1"]
            conclusion2 = df["hypothesis_ja_2"]
            conclusion3 = df["hypothesis_ja_3"]
            conclusion4 = df["hypothesis_ja_4"]
            conclusion5 = df["hypothesis_ja_5"]
        else:
            raise ValueError("Invalid language specified.")

        res_ans = []
        res_orig = []
        res_logprobs = []

        count = 0

        for ps, h1, h2, h3, h4, h5 in zip(
            premises, conclusion1, conclusion2, conclusion3, conclusion4, conclusion5
        ):
            p1, p2 = ps.split(self.sentence_finals.get(lang, "."))[:2]
            input_str = self.make_prompt(
                prompt,
                p1.strip(),
                p2.strip(),
                h1.strip(),
                h2.strip(),
                h3.strip(),
                h4.strip(),
                h5.strip(),
                bind_vars=bind_vars,
                lang=lang,
            )

            count += 1

            self.log(f"=== ID {count} ===")
            self.log(input_str)

            if model == "mock" or (
                isinstance(model, str) and model.startswith("mock.")
            ):
                pred, logprobs = self.mock_ask_llm(input_str)
            else:
                pred, logprobs = self.ask_llm(
                    input_str, model=model, include_logprobs=include_logprobs
                )

            self.log(f"{pred}\n")
            res_orig.append(pred)
            ans = self.put_answer(pred)
            res_ans.append(ans)

            res_logprobs.append(logprobs)

        return {
            "predictions": res_ans,
            "predictions_orig": res_orig,
            "predictions_logprobs": res_logprobs,
        }

    def put_answer(self, exp):
        assert isinstance(exp, str)

        exp = exp.replace("答えは: ", "")
        exp = exp.replace("The answer is: ", "")
        line = exp.strip().split("\n")[-1]
        ans = line.strip().split()[0].lower()
        ans = ans.split(".")[0]
        return ans

    def make_prompt(
        self,
        prompt: str,
        pr1: str,
        pr2: str,
        con1: str,
        con2: str,
        con3: str,
        con4: str,
        con5: str,
        bind_vars: Dict[str, str] = {},
        lang: str = "en",
    ):
        if lang == "en":
            res = (
                prompt.format(**bind_vars)
                + "\n"
                + f"Premise 1: {pr1}.\n"
                + f"Premise 2: {pr2}.\n"
                + f"1. {con1}\n"
                + f"2. {con2}\n"
                + f"3. {con3}\n"
                + f"4. {con4}\n"
                + f"5. {con5}\n\n"
                + "The answer is:"
            )
        elif lang == "ja":
            res = (
                prompt.format(**bind_vars)
                + "\n"
                + f"前提 1: {pr1}。\n"
                + f"前提 2: {pr2}。\n"
                + f"1. {con1}\n"
                + f"2. {con2}\n"
                + f"3. {con3}\n"
                + f"4. {con4}\n"
                + f"5. {con5}\n\n"
                + "答えは:"
            )

        return res

    def get_dataset_stats(self):
        """Get dataset statistics as dict"""

        stats = {}

        stats["total_count"] = len(self.df_test)

        columns = [
            # "source",
            # "category",
            "gold",
            "content-type",
        ]

        for col in columns:
            stats[col] = self.df_test[col].value_counts().to_dict()

        return stats
