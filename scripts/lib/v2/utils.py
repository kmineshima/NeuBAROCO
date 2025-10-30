def sample_balance(df, column, n_labels, sample_n):
    if sample_n % n_labels != 0:
        raise ValueError(
            f"sample_n ({sample_n}) must be a multiple of n_labels ({n_labels})"
        )

    g = df.groupby(column)

    if g.size().min() < sample_n // n_labels:
        raise ValueError("Duplication will occur. Adjust sample_n or n_labels.")

    return g.apply(lambda x: x.sample(sample_n // n_labels).reset_index(drop=True))
