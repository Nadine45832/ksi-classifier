import pandas as pd


def main(file_path):
    df = pd.read_csv(file_path)

    print(df.shape)


main("TOTAL_KSI_6386614326836635957.csv")
