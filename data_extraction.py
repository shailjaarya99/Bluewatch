import pandas as pd
import matplotlib.pyplot as plt

for name in ["train", "vali", "test"]:
    df = pd.read_csv(name + ".csv")
    df["label"] = df["label"].astype(int)

    print("=====", name, "=====")
    print(df.shape)
    print(df.head())
    print(df.isna().sum())
    print(df["label"].value_counts().sort_index())

    df["label"].value_counts().sort_index().plot(kind="bar", logy=True)
    plt.title("Class balance - " + name)
    plt.savefig("class_balance_" + name + ".png")
    plt.close()