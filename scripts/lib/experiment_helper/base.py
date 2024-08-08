import json
import logging
import math
from base64 import b64encode
from datetime import datetime
from os import PathLike
from pathlib import Path
from time import sleep
from typing import Dict

import litellm
import openai
import pandas as pd
from rich.logging import RichHandler
from rich.progress import track
from tenacity import RetryError, Retrying, stop_after_attempt, wait_random_exponential

from scripts.lib.reporter import create_report, create_report_overall
from scripts.lib.tokens import cost_from_messages

# # openai-python v1: Disable httpx logging in openai-python:
logging.getLogger("httpx").setLevel(logging.WARNING)

# # LiteLLM: Disable LiteLLM logging:
# logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Drop unsupported parameters outside OpenAI API
litellm.drop_params = True


class Base64Image:
    def __init__(self, image_path):
        self.image_path = image_path

    def __str__(self):
        return self.encode_image()

    def encode_image(self):
        with open(self.image_path, "rb") as f:
            return b64encode(f.read()).decode("utf-8")


class Experiment:
    """Experiment helper class."""

    # Define language-specific sentence finals
    sentence_finals = {"en": ".", "ja": "。"}

    # Define answer categories
    ents = ["entailment", "entailment."]
    conts = ["contradiction", "contradiction."]
    neutrals = ["neither", "neither."]
    ents_ja = ["含意", "含意する"]
    conts_ja = ["矛盾", "矛盾する", "矛盾している"]
    neutrals_ja = ["どちらでもない", "含意しない"]

    # Default values
    _mock_response = "entailment"
    _response_max_tokens = 10

    interval: float | None

    def __init__(
        self,
        df_test,
        basedir=None,
        rootdir="./",
        append_timestamp: bool = True,
        interval: float | None = None,
    ):
        self.messages_history = []
        self.report_files = []
        self.result_files = []
        self.output_dirs = []

        self.mock_response = self._mock_response
        self.response_max_tokens = self._response_max_tokens

        # Set the test data
        self.df_test = df_test
        self.interval = interval

        self.rootdir = rootdir

        self.timestamp = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

        # Set the base directory of the experiment
        if basedir:
            if append_timestamp:
                basedir = Path(rootdir, f"{basedir}.{self.timestamp}")
            else:
                pass
            Path(rootdir, basedir).mkdir(exist_ok=False)
        else:
            basedir = "."
            Path(rootdir, basedir).mkdir(exist_ok=True)

        self.basedir = basedir

        # Create a logger that logs stdout outputs to a file
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                # logging.FileHandler(Path(basedir, log_file)),
                # logging.StreamHandler(sys.stdout),
                RichHandler(
                    show_level=False, omit_repeated_times=False, show_path=False
                ),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.log = self.logger.info

    def make_prompt(
        self,
        prompt: str,
        pr1: str,
        pr2: str,
        con: str,
        bind_vars: Dict[str, str] = {},
        lang: str = "en",
    ):
        if lang == "en":
            prompt = prompt.format(**bind_vars)

            problem = (
                f"Premise 1: {pr1}.\n" + f"Premise 2: {pr2}.\n" + f"Hypothesis: {con}"
            )

            # NOTE: Use "{problem}" as a special placeholder for the problem
            if "{problem}" in prompt:
                res = prompt.format(problem=problem)
            else:
                res = prompt + "\n" + f"{problem}\n" + "The answer is: "
        elif lang == "ja":
            res = (
                prompt.format(**bind_vars)
                + "\n"
                + f"前提1: {pr1}。\n"
                + f"前提2: {pr2}。\n"
                + f"仮説: {con}\n"
                + "答えは:"
            )
        else:
            raise ValueError("Invalid language specified.")

        return res

    # Choose the model to use
    def ask_llm(
        self,
        text,
        image_path: str | PathLike | None = None,
        model: str | dict = "gpt-3.5-turbo",
        include_logprobs=True,
        **kwargs,
    ):
        message = {"role": "user", "content": text}
        if image_path:
            encoded_image = Base64Image(image_path)
            message["content"] = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ]

        self.messages_history.append(message)

        if isinstance(model, dict):
            model_name = model["model"]
            model_kwargs = {k: v for k, v in model.items() if k != "model"}

            completion = litellm.completion(
                model=model_name,
                max_tokens=self.response_max_tokens,
                messages=[message],
                logprobs=include_logprobs,
                top_logprobs=0,
                # top_p=0.1,
                **model_kwargs,
            )
        else:
            # Use OpenAI API by default
            try:
                for attempt in Retrying(
                    wait=wait_random_exponential(min=1, max=60),
                    stop=stop_after_attempt(6),
                ):
                    with attempt:
                        params = {
                            "model": model,
                            "max_tokens": self.response_max_tokens,
                            "messages": [message],
                        }
                        if include_logprobs:
                            params.update({"logprobs": True, "top_logprobs": 0})

                        completion = openai.chat.completions.create(**params)
            except RetryError:
                raise

        response = completion.choices[0].message.content
        logprobs = None

        if (
            hasattr(completion.choices[0], "logprobs")
            and completion.choices[0].logprobs
        ):
            logprobs = []

            logprobs_content = completion.choices[0].logprobs.content

            for token_data in logprobs_content:
                token = token_data.token
                logprob = token_data.logprob
                prob = math.exp(logprob)
                logprobs.append({"token": token, "logprob": logprob, "prob": prob})

        self.messages_history.append(completion.choices[0].message)

        # Delay
        if model == "gpt-4":
            sleep(self.interval or 0.75)
        else:
            sleep(self.interval or 0.25)

        return response, logprobs

    def mock_ask_llm(self, text, image_path: str | PathLike | None = None):
        message = {"role": "user", "content": text}

        if image_path:
            encoded_image = Base64Image(image_path)
            message["content"] = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ]

        self.messages_history.append(message)

        # Dummy response: "hello " * response_max_tokens
        self.messages_history.append(
            {"role": "dummy", "content": "hello " * self.response_max_tokens}
        )

        response = self.mock_response

        return response, None

    # Define the function to classify answers
    def put_answer(self, exp):
        assert isinstance(exp, str)

        exp = exp.replace("答えは: ", "")
        exp = exp.replace("The answer is: ", "")
        line = exp.strip().split("\n")[-1]
        ans = line.strip().split()[0].lower()
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

    # Define the main prediction function
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
            conclusions = df["hypothesis_en"]
        elif lang == "ja":
            premises = df["premises_ja"]
            conclusions = df["hypothesis_ja"]
        else:
            raise ValueError("Invalid language specified.")

        ids = df["ID"] if "ID" in df.columns else None

        res_ans = []
        res_orig = []
        res_logprobs = []

        total = len(premises)

        for count, (ps, h, id) in track(
            enumerate(zip(premises, conclusions, ids)),
            total=total,
            description=f"total={total}",
        ):
            p1, p2 = ps.split(self.sentence_finals.get(lang, "."))[:2]
            input_str = self.make_prompt(
                prompt,
                p1.strip(),
                p2.strip(),
                h.strip(),
                bind_vars=bind_vars,
                lang=lang,
            )

            self.log(f"=== Request #{count + 1}{f' (ID: {str(id)})' if id else ''} ===")
            self.log(input_str)

            if model == "mock" or (
                isinstance(model, str) and model.startswith("mock.")
            ):
                pred, logprobs = self.mock_ask_llm(input_str)
            else:
                pred, logprobs = self.ask_llm(
                    input_str, model=model, include_logprobs=include_logprobs
                )

            self.log(pred)
            self.log("\n")

            res_orig.append(pred)
            ans = self.put_answer(pred)
            res_ans.append(ans)

            res_logprobs.append(logprobs)

        return {
            "predictions": res_ans,
            "predictions_orig": res_orig,
            "predictions_logprobs": res_logprobs,
        }

    # Define the evaluation function
    def eval_report(
        self,
        prompt: str,
        bind_vars: Dict[str, str] = {},
        lang: str = "en",
        model: str | dict = "gpt-3.5-turbo",
        include_logprobs=True,
        output_dir: str | Path | None = None,
    ):
        timestamp = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

        if not output_dir:
            output_dir = f"experiment.{timestamp}"

        self.output_dirs.append(output_dir)

        output_dir = Path(self.rootdir, self.basedir, output_dir)

        Path(output_dir).mkdir(exist_ok=False)

        # TODO: Make customizable?
        result_file = Path(output_dir, "result.tsv")
        report_file = Path(output_dir, "report.html")

        json_file = Path(output_dir, "experiment.json")

        log_file = Path(output_dir, "experiment.log")
        self.logger.addHandler(logging.FileHandler(log_file))

        self.log(f"=== Prompt ===\n{prompt}\n")

        for i, (k, v) in enumerate(bind_vars.items()):
            self.log(f"=== {k} ===\n{v}\n")

        pred = self.predict(
            self.df_test,
            prompt,
            bind_vars=bind_vars,
            lang=lang,
            model=model,
            include_logprobs=include_logprobs,
        )

        df_result = self.df_test.copy()

        df_result["prediction"] = pred["predictions"]
        df_result["prediction_orig"] = pred["predictions_orig"]
        if pred.get("predictions_logprobs", None) is not None:
            df_result["prediction_logprobs"] = pred["predictions_logprobs"]

        # Check for 'check' label in predictions
        if "check" in df_result["prediction"]:
            self.log("Warning: Some answers are not classified correctly.")

        if result_file:
            df_result.to_csv(result_file, sep="\t", index=False)
            self.log(f"[*] Result data: {result_file}")
            create_report(
                log_file=log_file,
                tsv_file=result_file,
                prompt=prompt.strip(),
                bind_vars=bind_vars,
                output_file=report_file,
                timestamp=timestamp,
                model=model if isinstance(model, str) else model["model"],
            )
            self.log(f"[*] Output html: {report_file}")

        if json_file:
            with open(json_file, "w") as f:
                json.dump(
                    {
                        "lang": lang,
                        "prompt": prompt,
                        "bind_vars": bind_vars,
                        "timestamp": timestamp,
                        "model": model,
                        "result_tsv": "result.tsv",
                        "log_txt": "experiment.log",
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        self.result_files.append(result_file)
        self.report_files.append(report_file)

        return df_result

    def create_report_overall(
        self,
    ):
        if self.basedir == ".":
            raise Exception("No basedir specified.")

        timestamp = self.timestamp

        reports_relative = [
            f"./{Path(report_file.parent.name, report_file.name)}"
            for report_file in self.report_files
        ]

        output_file = Path(self.rootdir, self.basedir, "overall.html")
        json_file = Path(self.rootdir, self.basedir, "overall.json")

        dataset_stats = self.get_dataset_stats()

        if json_file:
            result_files_relative = [
                f"{Path(result_file.parent.name, result_file.name)}"
                for result_file in self.result_files
            ]

            with open(json_file, "w") as f:
                json.dump(
                    {
                        "timestamp": timestamp,
                        "experiment_dirs": self.output_dirs,
                        "result_tsv_files": result_files_relative,
                        "reports": reports_relative,
                        "dataset_stats": dataset_stats,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        create_report_overall(
            timestamp,
            self.result_files,
            output_file,
            names=self.output_dirs,
            reports=reports_relative,
            dataset_stats=dataset_stats,
        )

    def get_dataset_stats(self):
        """Get dataset statistics as dict"""

        stats = {}

        stats["total_count"] = len(self.df_test)

        columns = [
            # "source",
            # "category",
            "gold",
            "mood",
            "inference-type",
            "content-type",
            "conversion",
            "atmosphere",
        ]

        for col in columns:
            stats[col] = self.df_test[col].value_counts().to_dict()

        return stats

    def estimate_cost(self, model="gpt-3.5-turbo"):
        cost_from_messages(self.messages_history, model=model)
