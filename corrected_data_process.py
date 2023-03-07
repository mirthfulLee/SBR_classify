import pandas as pd
import os
from MemVul import util
import json


def data_merge():
    # Merge data & unify the column name
    df_list = []
    for src in srcs:
        data_file = os.path.join(DATA_ROOT, "{}.csv".format(src))
        df = pd.read_csv(data_file, header=0)
        # print(src, df["security"].value_counts())
        if src == "Chromium":
            df["summary"] = ""
        df = df.loc[:, ["summary", "description", "security"]]
        df = df.rename(
            columns={
                "summary": "Issue_Title",
                "description": "Issue_Body",
                "security": "Security_Issue_Full",
            }
        )
        df_list.append(df)
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    merged_df.to_csv(os.path.join(DATA_ROOT, merged_file), header=True, index=False)


def text_preprocess():
    df = pd.read_csv(os.path.join(DATA_ROOT, merged_file), header=0)
    df["Issue_Title"] = df["Issue_Title"].map(util.replace_tokens_simple)
    df["Issue_Body"] = df["Issue_Body"].map(util.replace_tokens_simple)
    df.to_csv(os.path.join(DATA_ROOT, processed_file), header=True, index=False)
    records = df.to_dict(orient="records")

    with open(os.path.join(DATA_ROOT, final_json_file), "w") as ff:
        json.dump(records, ff, indent=4)


if __name__ == "__main__":
    srcs = ["Ambari", "Camel", "Chromium", "Derby", "Wicket"]
    DATA_ROOT = "corrected_data"
    merged_file = "merged.csv"
    processed_file = "merged_processed.csv"
    final_json_file = "merged_processed.json"
    data_merge()
    text_preprocess()
