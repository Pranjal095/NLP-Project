"""
Data loader for MentalManip dataset.
Uses csv.reader (not pandas.read_csv) as recommended in the repo README,
since pandas does not read the columns correctly for this CSV format.
"""

import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def load_mentalmanip(
    file_path,
    train_ratio=0.6,
    valid_ratio=0.2,
    test_ratio=None,
    random_state=42,
    stratified=True,
):
    """
    Load a MentalManip CSV file and split into train/valid/test sets.

    Args:
        file_path:     Path to mentalmanip_con.csv or mentalmanip_maj.csv
        train_ratio:   Fraction of data for training
        valid_ratio:   Fraction of data for validation
        test_ratio:    Fraction of data for testing. Defaults to remaining data.
        random_state:  Random seed for reproducibility
        stratified:    Preserve the manipulation label distribution across splits

    Returns:
        (train_df, valid_df, test_df) — each a pandas DataFrame with columns:
            Dialogue, Manipulative (int 0/1), Technique, Vulnerability
    """
    # --- read with csv.reader for reliability ---
    rows = []
    columns = None
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for idx, row in enumerate(reader):
            if idx == 0:
                columns = row
            else:
                rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Convert label to int
    df["Manipulative"] = df["Manipulative"].astype(int)

    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - valid_ratio
    split_total = train_ratio + valid_ratio + test_ratio
    if abs(split_total - 1.0) > 1e-8:
        raise ValueError(
            "train_ratio + valid_ratio + test_ratio must sum to 1.0 "
            f"(got {split_total:.4f})"
        )

    stratify_labels = df["Manipulative"] if stratified else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_labels,
    )

    valid_fraction_of_temp = valid_ratio / (valid_ratio + test_ratio)
    temp_stratify = temp_df["Manipulative"] if stratified else None
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_fraction_of_temp,
        random_state=random_state,
        shuffle=True,
        stratify=temp_stratify,
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Print summary
    split_kind = "stratified" if stratified else "shuffled"
    print(f"  Split: {split_kind} {train_ratio:.0%}/{valid_ratio:.0%}/{test_ratio:.0%}")
    for name, split in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        n_manip = (split["Manipulative"] == 1).sum()
        n_non   = (split["Manipulative"] == 0).sum()
        print(f"  {name}: {len(split)} samples  (manip={n_manip}, non-manip={n_non})")

    return train_df, valid_df, test_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick data-loader test")
    parser.add_argument("--data_path", default="../mentalmanip_dataset/mentalmanip_con.csv")
    args = parser.parse_args()

    print(f"Loading {args.data_path} ...")
    train, valid, test = load_mentalmanip(args.data_path)
    print(f"\nSample dialogue (train[0]):\n{train['Dialogue'].iloc[0][:200]}...")
