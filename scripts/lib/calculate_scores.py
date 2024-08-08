import argparse
import collections

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


def show_counter(counter):
    for key, value in counter.items():
        print(key, value)


def count_labels(df, tag):
    """Count the number of labels in the DataFrame."""
    if tag not in df.columns:
        return
    print(f"=== {tag} ===")
    show_counter(collections.Counter(df[tag]))


def calculate_accuracy(df, section_column, section_label, preduction_column):
    """Calculate accuracy for a specific section and label in the DataFrame."""
    # if section_label is a list:
    if isinstance(section_label, list):
        section_df = df[df[section_column].isin(section_label)]
    else:
        section_df = df[df[section_column] == section_label]
    num = len(section_df["gold"])
    if num == 0:
        print(f"- {section_label}: {0} ({num})")
        return 0
    else:
        score = accuracy_score(section_df["gold"], section_df[preduction_column])
        score = round(score, 4)
        print(f"- {section_label}: {score} ({num})")
        return score


def calculate_accuracy_with_label(df, section, tag, label, model):
    """Calculate accuracy for a specific section, label, and model in the DataFrame."""
    section_df = df[df[section] == tag]
    if isinstance(label, list):
        label_df = section_df[section_df["gold"].isin(label)]
    else:
        label_df = section_df[section_df["gold"] == label]
    num = len(label_df)
    if num == 0:
        print(f"  - {label}: {0} ({num})")
        return 0
    else:
        score = accuracy_score(label_df["gold"], label_df[model])
        score = round(score, 4)
        print(f"  - {label}: {score} ({num})")
        return score


def calculate_precision(
    df, section_column, section_label, prediction_column, target_label, labels=[]
):
    """Calculate precision for a specific section and label in the DataFrame."""
    if section_column and section_label:
        if isinstance(section_label, list):
            section_df = df[df[section_column].isin(section_label)]
        else:
            section_df = df[df[section_column] == section_label]
    else:
        section_df = df
    num = len(section_df["gold"])
    if num == 0:
        print(f"- {section_label} ({target_label or 'macro'}): {0} ({num})")
        return 0
    else:
        if target_label:
            score = precision_score(
                section_df["gold"],
                section_df[prediction_column],
                labels=[target_label],
                average="macro",
            )
            # print('section_df["gold"]',section_df["gold"], file=sys.stderr)
        else:
            score = precision_score(
                section_df["gold"],
                section_df[prediction_column],
                labels=labels,
                average="macro",
            )
        score = round(score, 4)
        print(f"- {section_label} ({target_label}): {score} ({num})")
        return score


def calculate_recall(
    df, section_column, section_label, prediction_column, target_label, labels=[]
):
    """Calculate recall for a specific section and label in the DataFrame."""
    if section_label == "all":
        section_df = df
    else:
        if isinstance(section_label, list):
            section_df = df[df[section_column].isin(section_label)]
        else:
            section_df = df[df[section_column] == section_label]
    num = len(section_df["gold"])
    if num == 0:
        print(f"- {section_label} ({target_label or 'macro'}): {0} ({num})")
        return 0
    else:
        if target_label:
            score = recall_score(
                section_df["gold"],
                section_df[prediction_column],
                labels=[target_label],
                average="macro",
            )
        else:
            score = recall_score(
                section_df["gold"],
                section_df[prediction_column],
                labels=labels,
                average="macro",
            )
            import sys

            print(section_df["gold"], file=sys.stderr)
            for label in labels:
                x = recall_score(
                    section_df["gold"],
                    section_df[prediction_column],
                    labels=[label],
                    average="macro",
                )
                cnt = len(section_df[section_df["gold"] == label])
                # number of correct predictions for the label
                pred_cnt = len(
                    section_df[
                        (section_df["gold"] == label)
                        & (section_df[prediction_column] == label)
                    ]
                )
                print(f"  - {label} ({cnt}): {x} ({pred_cnt} / {cnt})", file=sys.stderr)
            print(
                f"- {section_label} ({target_label}): {score} ({num})\n",
                file=sys.stderr,
            )
            acc = accuracy_score(section_df["gold"], section_df[prediction_column])
            print(f"  - accuracy: {acc}", file=sys.stderr)
        score = round(score, 4)
        print(f"- {section_label} ({target_label}): {score} ({num})")
        return score


def calculate_scores(df, no_choice3=False):
    row = {}

    is_choice3 = False if no_choice3 else True

    """Calculate and display accuracy scores for different sections and labels."""
    all_accuracy = accuracy_score(df["gold"], df["prediction"])
    all_accuracy = round(all_accuracy, 4)
    count = len(df)
    row[f"all_accuracy ({count})"] = all_accuracy
    if is_choice3:
        labels = ["entailment", "contradiction", "neutral"]
    else:
        labels = ["1", "2", "3", "4", "5"]
        # print(df["gold"].unique())
    all_precision = calculate_precision(
        df,
        None,
        "all",
        "prediction",
        None,
        labels=labels,
    )
    row[f"all_precision ({count})"] = all_precision
    all_recall = calculate_recall(
        df,
        None,
        "all",
        "prediction",
        None,
        labels=labels,
    )
    row[f"all_recall ({count})"] = all_recall

    if is_choice3:
        sections = ["entailment", "contradiction", "neutral"]
        for section in sections:
            score = calculate_accuracy(df, "gold", section, "prediction")
            if isinstance(section, list):
                count = len(df[df["gold"].isin(section)])
            else:
                count = len(df[df["gold"] == section])
            row[f"{section} ({count})"] = score

            # for section in sections:
            score_precision = calculate_precision(
                df, None, "all", "prediction", section
            )
            if isinstance(section, list):
                count = len(df[df["gold"].isin(section)])
            else:
                count = len(df[df["gold"] == section])
            row[f"{section}_precision ({count})"] = score_precision

            # for section in sections:
            score_recall = calculate_recall(df, None, "all", "prediction", section)
            if isinstance(section, list):
                count = len(df[df["gold"].isin(section)])
            else:
                count = len(df[df["gold"] == section])
            row[f"{section}_recall ({count})"] = score_recall

    if "has_conclusion" in df.columns:
        print("*has_conclusion*")
        score = calculate_accuracy(df, "has_conclusion", True, "prediction")
        count = len(df[df["has_conclusion"] is True])
        row[f"has_conclusion_true ({count})"] = score
        score = calculate_accuracy(df, "has_conclusion", False, "prediction")
        count = len(df[df["has_conclusion"] is False])
        row[f"has_conclusion_false ({count})"] = score
        # for label in ["entailment", "contradiction", "neutral"]:
        #     score = calculate_accuracy_with_label(
        #         df, "has_conclusion", True, label, "prediction"
        #     )
        #     row[f"has_conclusion_{label}"] = score

    if "inference-type" in df.columns:
        print("*inference-type*")
        sections = ["syllogism", "propositional", "extended"]
        for section in sections:
            score = calculate_accuracy(df, "inference-type", section, "prediction")
            row[section] = score

    if "content-type" in df.columns:
        print("*content-type*")
        tags = [
            "congruent",
            "incongruent",
            "others",
            "symbolic",
        ]
        for tag in tags:
            score = calculate_accuracy(df, "content-type", tag, "prediction")
            count = len(df[df["content-type"] == tag])
            row[f"{tag} ({count})"] = score
            if is_choice3:
                for label in ["entailment", "contradiction", "neutral"]:
                    score = calculate_accuracy_with_label(
                        df, "content-type", tag, label, "prediction"
                    )
                    count = len(df[(df["content-type"] == tag) & (df["gold"] == label)])
                    row[f"{tag}_{label} ({count})"] = score

                score = calculate_accuracy_with_label(
                    df,
                    "content-type",
                    tag,
                    ["entailment", "contradiction"],
                    "prediction",
                )
                count = len(
                    df[
                        (df["content-type"] == tag)
                        & (df["gold"].isin(["entailment", "contradiction"]))
                    ]
                )
                row[f"{tag}_non-neutral ({count})"] = score

    if "conversion" in df.columns:
        print("*conversion*")
        score = calculate_accuracy(df, "conversion", True, "prediction")
        count = len(df[df["conversion"] == "yes"])
        row[f"conversion ({count})"] = score
        if is_choice3:
            score = calculate_accuracy(df, "conversion", False, "prediction")
            count = len(df[df["conversion"] == "no"])
            row[f"non-conversion ({count})"] = score

            for label in ["entailment", "contradiction", "neutral"]:
                score = calculate_accuracy_with_label(
                    df, "conversion", True, label, "prediction"
                )
                count = len(df[(df["conversion"] == "yes") & (df["gold"] == label)])
                row[f"conversion_{label} ({count})"] = score

                score = calculate_accuracy(df, "conversion", False, "prediction")
                count = len(df[(df["conversion"] == "no") & (df["gold"] == label)])
                row[f"non-conversion_{label} ({count})"] = score

        if "content-type" in df.columns:
            # symbolicとcongruentでconversionの場合
            target_df = df[
                (df["content-type"] == "symbolic") | (df["content-type"] == "congruent")
            ]
            score = calculate_accuracy(target_df, "conversion", True, "prediction")
            count = len(target_df[target_df["conversion"] == "yes"])
            row[f"conversion_(symbolic+congruent) ({count})"] = score
            if is_choice3:
                for label in ["entailment", "contradiction", "neutral"]:
                    score = calculate_accuracy_with_label(
                        target_df, "conversion", True, label, "prediction"
                    )
                    count = len(
                        target_df[
                            (target_df["conversion"] == "yes")
                            & (target_df["gold"] == label)
                        ]
                    )
                    row[f"conversion_(symbolic+congruent)_{label} ({count})"] = score

            # incongruentでconversionの場合
            target_df = df[df["content-type"] == "incongruent"]
            score = calculate_accuracy(target_df, "conversion", True, "prediction")
            count = len(target_df[target_df["conversion"] == "yes"])
            row[f"conversion_incongruent ({count})"] = score
            if is_choice3:
                for label in ["entailment", "contradiction", "neutral"]:
                    score = calculate_accuracy_with_label(
                        target_df, "conversion", True, label, "prediction"
                    )
                    count = len(
                        target_df[
                            (target_df["conversion"] == "yes")
                            & (target_df["gold"] == label)
                        ]
                    )
                    row[f"conversion_incongruent_{label} ({count})"] = score

    if "conversion_type" in df.columns:
        print("*conversion_type*")
        tags = [
            # "n/a",
            "A",
            "O",
            "AO",
        ]
        for tag in tags:
            score = calculate_accuracy(df, "conversion_type", tag, "prediction")
            count = len(df[df["conversion_type"] == tag])
            row[f"conversion_type_{tag} ({count})"] = score
            if is_choice3:
                for label in ["entailment", "contradiction", "neutral"]:
                    score = calculate_accuracy_with_label(
                        df, "conversion_type", tag, label, "prediction"
                    )
                    count = len(
                        df[(df["conversion_type"] == tag) & (df["gold"] == label)]
                    )
                    row[f"conversion_type_{tag}_{label} ({count})"] = score

            # symbolicとcongruentでconversionの場合
            target_df = df[
                (df["content-type"] == "symbolic") | (df["content-type"] == "congruent")
            ]
            score = calculate_accuracy(target_df, "conversion_type", tag, "prediction")
            count = len(target_df[target_df["conversion_type"] == tag])
            row[f"conversion_(symbolic+congruent)_type_{tag} ({count})"] = score
            if is_choice3:
                for label in ["entailment", "contradiction", "neutral"]:
                    score = calculate_accuracy_with_label(
                        target_df, "conversion_type", tag, label, "prediction"
                    )
                    count = len(
                        target_df[
                            (target_df["conversion_type"] == tag)
                            & (target_df["gold"] == label)
                        ]
                    )
                    row[
                        f"conversion_(symbolic+congruent)_type_{tag}_{label} ({count})"
                    ] = score

            # incongruentでconversionの場合
            target_df = df[df["content-type"] == "incongruent"]
            score = calculate_accuracy(target_df, "conversion_type", tag, "prediction")
            count = len(target_df[target_df["conversion_type"] == tag])
            row[f"conversion_incongruent_type_{tag} ({count})"] = score
            if is_choice3:
                for label in ["entailment", "contradiction", "neutral"]:
                    score = calculate_accuracy_with_label(
                        target_df, "conversion_type", tag, label, "prediction"
                    )
                    count = len(
                        target_df[
                            (target_df["conversion_type"] == tag)
                            & (target_df["gold"] == label)
                        ]
                    )
                    row[f"conversion_incongruent_type_{tag}_{label} ({count})"] = score

    if "atmosphere" in df.columns:
        print("*atmosphere*")
        score = calculate_accuracy(df, "atmosphere", "yes", "prediction")
        row["atmosphere"] = score
        if is_choice3:
            for label in ["entailment", "contradiction", "neutral"]:
                calculate_accuracy_with_label(
                    df, "atmosphere", "yes", label, "prediction"
                )
                row[f"atmosphere_{label}"] = score

    scores = row
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and display accuracy scores for different sections and labels."
    )
    parser.add_argument("FILE", help="Input file in tsv format")
    parser.add_argument(
        "-c", "--count", action="store_true", help="Show the statistics of labels"
    )
    args = parser.parse_args()

    filename = args.FILE
    basename = filename.split(".")[0]
    df = pd.read_csv(filename, delimiter="\t")

    if args.count:
        print(f"== Data: {basename} ==")
        print(f"Total: {len(df)}")
        count_labels(df, "gold")
        count_labels(df, "inference-type")
        count_labels(df, "content-type")
        count_labels(df, "mood")
        count_labels(df, "conversion")
        count_labels(df, "atmosphere")
        print("\n")

    print(f"== Score: {basename} ==")
    calculate_scores(df, args.no_choice3)


if __name__ == "__main__":
    main()
