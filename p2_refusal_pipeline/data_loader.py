"""
Data loader for MentalManip dataset.
Uses csv.reader (not pandas.read_csv) as recommended in the repo README,
since pandas does not read the columns correctly for this CSV format.
"""

import csv
import pandas as pd


def load_mentalmanip(file_path, train_ratio=0.6, valid_ratio=0.2, random_state=42):
    """
    Load a MentalManip CSV file and split into train/valid/test sets.

    Args:
        file_path:     Path to mentalmanip_con.csv or mentalmanip_maj.csv
        train_ratio:   Fraction of data for training
        valid_ratio:   Fraction of data for validation
        random_state:  Random seed for reproducibility

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

    # Shuffle and split
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_end = int(train_ratio * len(df))
    valid_end = train_end + int(valid_ratio * len(df))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    valid_df = df.iloc[train_end:valid_end].reset_index(drop=True)
    test_df  = df.iloc[valid_end:].reset_index(drop=True)

    # Print summary
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
