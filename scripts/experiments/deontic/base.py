from textwrap import dedent

from scripts.lib.v2 import Experiment

# # openai-python v1: Disable httpx logging in openai-python:
# import logging
# logging.getLogger("httpx").setLevel(logging.WARNING)


def run_experiment_base(
    exp: Experiment,
    model: str,
    prompt: str | None = None,
    bind_vars: dict = {},
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
    }

    prompt = prompt or default_prompt[lang]

    # NOTE: Evaluation execution and report output
    exp.eval_report(
        prompt,
        bind_vars=bind_vars,
        output_dir=output_dir,
        model=model,
        lang=lang,
    )
