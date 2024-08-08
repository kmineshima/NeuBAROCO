import base64
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict

import jinja2
import matplotlib.pyplot as plt
import pandas as pd

from scripts.lib.calculate_scores import calculate_scores, count_labels

# from premailer import transform


def create_report(
    timestamp,
    prompt: str,
    log_file,
    tsv_file,
    output_file,
    model: str,
    bind_vars: Dict[str, str] = {},
    no_choice3: bool = False,
):
    """Create HTML report using jinja2 from log and output tsv"""

    with open(log_file, "r") as f:
        log_text = f.read()

    # NOTE: set dtype of "gold" and "prediction" columns to "object", as the "prediction" column may contain "none"
    df = pd.read_csv(
        tsv_file, delimiter="\t", dtype={"gold": object, "prediction": object}
    )

    def coloring(row):
        styles = [None for _ in row]
        if str(row["prediction"]) == str(row["gold"]):
            styles[0] = "background-color: green"
        else:
            styles[0] = "background-color: red"
        return styles

    results_table = (
        df.style.apply(coloring, axis=1)
        .set_table_attributes('border="1" class="dataframe"')
        .to_html()
    )

    # results_table = df.to_html()
    # results_table = transform(results_table)

    # Stats
    basename = Path(tsv_file).stem
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"== Data: {basename} ==")
        print(f"Total: {len(df)}")
        count_labels(df, "gold")
        count_labels(df, "inference-type")
        count_labels(df, "content-type")
        count_labels(df, "conversion")
        count_labels(df, "atmosphere")
        print("\n")

        print(f"== Score: {basename} ==")
        calculate_scores(df, no_choice3)

    stats_text = buf.getvalue()

    templateLoader = jinja2.FileSystemLoader(
        searchpath=str(Path(Path(__file__).parent))
    )
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "./report_template.html"
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render(
        timestamp=timestamp,
        model=model,
        prompt=prompt,
        bind_vars=bind_vars,
        stats_text=stats_text,
        log_text=log_text,
        results_table=results_table,
    )
    with open(output_file, "w") as f:
        f.write(outputText)


def create_report_overall(
    timestamp: str,
    tsv_files: list[str],
    output_file: str,
    names: list[str] = [],
    reports: list[str] = [],
    dataset_stats: dict | None = None,
    no_choice3: bool = False,
):
    """Create HTML overall report using jinja2 from log and output tsv

    Args:
        timestamp: Timestamp of the experiment
        tsv_files: List of tsv result files
        output_file: Output file name
        names: List of names of each run
        reports: List of report file paths for each run
    """

    if dataset_stats:
        _stats = {}
        for key, value in dataset_stats.items():
            if key == "total_count":
                _stats[key] = value
            else:
                # Dataframe from dict
                df = pd.DataFrame.from_dict(value, orient="index", columns=["#"])
                _stats[key] = df.to_html(index_names=False)

        dataset_stats = _stats

    all_scores = []

    for i, tsv_file in enumerate(tsv_files):
        # NOTE: set dtype of "gold" and "prediction" columns to "object", as the "prediction" column may contain "none"
        df = pd.read_csv(
            tsv_file, delimiter="\t", dtype={"gold": object, "prediction": object}
        )

        basename = Path(tsv_file).stem

        scores = {}
        scores["name"] = names[i] if names else basename
        scores.update(calculate_scores(df, no_choice3))
        all_scores.append(scores)

    df = pd.DataFrame(all_scores)
    df = df.set_index("name")

    table_html = df.to_html(index_names=False)

    # plt.figure()
    # Determine the size of the figure based on the number of rows
    # height = len(df) * 22
    ax = df.T.plot.barh(figsize=(10, 20))
    ax.invert_yaxis()
    plt.rcParams['font.size'] = 8
    for bars in ax.containers:
        ax.bar_label(bars)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.4, top=0.9, bottom=0.05)
    # plt.subplots_adjust(right=0.6)
    plt.legend(
        # bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0
        bbox_to_anchor=(0.5, 1.005),
        loc="lower center",
        borderaxespad=0,
    )  # , fontsize=18)

    img = io.BytesIO()
    plt.savefig(img, dpi=100, format="png")
    img.seek(0)

    plot_html = """<img src="data:image/png;base64,{}"/>""".format(
        base64.encodebytes(img.getvalue()).decode()
    )

    # plt.show()

    # with open(output_file, "w") as f:
    #     f.write(table_html + "\n" + plot_html)

    runs = [{"name": name, "href": report} for name, report in zip(names, reports)]

    templateLoader = jinja2.FileSystemLoader(
        searchpath=str(Path(Path(__file__).parent))
    )
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "./report_overall_template.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    outputText = template.render(
        dataset_stats=dataset_stats,
        timestamp=timestamp,
        runs=runs,
        table_html=table_html,
        plot_html=plot_html,
    )

    with open(output_file, "w") as f:
        f.write(outputText)


def create_report_from_json(json_file: Path, no_choice3: bool = False):
    with open(json_file, "r") as f:
        data = json.load(f)

    basedir = Path(json_file).parent

    model = data["model"]

    create_report(
        log_file=str(Path(basedir, data["log_txt"])),
        tsv_file=str(Path(basedir, data["result_tsv"])),
        prompt=data["prompt"].strip(),
        bind_vars=data["bind_vars"],
        output_file=str(Path(basedir, "report.html")),
        timestamp=data["timestamp"],
        model=model if isinstance(model, str) else model["model"],
        no_choice3=no_choice3,
    )


def create_report_overall_from_json(json_file: Path, no_choice3: bool = False):
    with open(json_file, "r") as f:
        data = json.load(f)

    basedir = Path(json_file).parent

    for exp_dir in data["experiment_dirs"]:
        exp_json = str(Path(basedir, exp_dir, "experiment.json"))
        create_report_from_json(exp_json, no_choice3=no_choice3)

    create_report_overall(
        timestamp=data["timestamp"],
        tsv_files=[str(Path(basedir, tsv)) for tsv in data["result_tsv_files"]],
        names=data["experiment_dirs"],
        reports=data["reports"],
        output_file=str(Path(basedir, "overall.html")),
        dataset_stats=data["dataset_stats"],
        no_choice3=no_choice3,
    )


def cli(
    json_file: str,
    no_choice3: bool = False,
):
    json_file = Path(json_file)

    if json_file.name == "experiment.json":
        create_report_from_json(json_file, no_choice3=no_choice3)

    elif json_file.name == "overall.json":
        create_report_overall_from_json(json_file, no_choice3=no_choice3)

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create HTML report using jinja2 from log and output tsv"
    )
    parser.add_argument("json_file", help="Input json file")
    parser.add_argument(
        "--no-choice3", action="store_true", help="Do not calculate choice3 scores"
    )
    args = parser.parse_args()

    cli(args.json_file, no_choice3=args.no_choice3)


# create_report_overall(
#     None,
#     [
#         "evalgpt_kshot.2023-09-08_00-57-50/output_random.2023-09-08_00-57-50.tsv",
#         "evalgpt_kshot.2023-09-08_00-35-40/output_random.2023-09-08_00-35-40.tsv",
#     ],
#     "overall.html",
# )
