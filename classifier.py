from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import LabelEncoder


columns_to_drop = [
    # this column is not useful for the analysis becuase their values are uqniue
    "OBJECTID",
    "INDEX",
    "FATAL_NO",
    "ACCNUM",
    # date and time doesn't give us useful information, we have aggregated columns for that like LIGHT and VISIBILITY
    "DATE",
    "TIME",
    # we have LATITUDE and LONGITUDE columns
    "STREET1",
    "STREET2",
    "OFFSET",
    # this is outdated fields
    "HOOD_158",
    "NEIGHBOURHOOD_158",
    # we have latitude and longitude columns
    "HOOD_140",
    "NEIGHBOURHOOD_140",
    # Name of the toronto district is not useful for the analysis, we also have latitude and longitude columns
    "DISTRICT",
    # police devision is not useful for the analysis
    "DIVISION",
    # this is exactly the same as LATITUDE and LONGITUDE
    "x",
    "y",
]

boolean_columns = [
    "PEDESTRIAN",
    "CYCLIST",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "TRUCK",
    "TRSN_CITY_VEH",
    "EMERG_VEH",
    "PASSENGER",
    "SPEEDING",
    "AG_DRIV",
    "REDLIGHT",
    "ALCOHOL",
    "DISABILITY",
]

cyclist_columns = ["CYCLISTYPE", "CYCACT", "CYCCOND"]

pedestrian_columns = ["PEDTYPE", "PEDACT", "PEDCOND"]

driver_columns = ["MANOEUVER", "DRIVACT", "DRIVCOND"]

driver_columns = ["MANOEUVER", "DRIVACT", "DRIVCOND"]

env_columns = ["ROAD_CLASS", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND"]

location_columns = ["LATITUDE", "LONGITUDE"]

direction_columns = ["ACCLOC", "INITDIR"]

injury_columns = ["IMPACTYPE", "INVTYPE", "INVAGE", "INJURY"]

vehicle_columns = ["VEHTYPE"]


def describe_data(df: pd.DataFrame) -> None:
    """
    This function performs data analysis on the given dataframe.
    :param df: pandas dataframe
    :return: None
    """
    print("Dataframe shape: ", df.shape)
    print("Column names: \n", df.columns.tolist())
    print("\nColumn types:\n", df.dtypes)

    print("\nMissing values:\n", df.isnull().sum(axis=0))
    print("\nColumn descriptions:\n", df.describe().round(2), "\n")

    for c in df.columns:
        text = f'Unique values for "{c}" column: '
        print(f"{text:<50} {len(df[c].unique()):>5}   {df[c].count():>5}")

    print("\nDuplicated rows: ", df.duplicated().sum())
    print("\nCorrelation matrix:\n", df.select_dtypes(include=["number"]).corr())

    # printing unique values for each column
    print(f"\nUnique values for DISTRICT: {df["DISTRICT"].unique()}")
    print(f"\nUnique values for ACCLOC: {df["ACCLOC"].unique()}")

    for column in boolean_columns:
        print(f"Unique values for {column}: {df[column].unique()}")


def visualize_data(df: pd.DataFrame) -> None:
    """
    This function performs data visualization on the given dataframe.
    :param df: pandas dataframe
    :return: None
    """
    # histogram for the cancer class
    plt.figure(figsize=(6, 4))
    plt.hist(
        df["ACCLASS"]
        .fillna("Unknown")
        .apply(lambda x: "Fatal" if x == "Fatal" else "Non-Fatal")
    )
    plt.title("Injuries class distribution")
    plt.xlabel("Injury class")
    plt.ylabel("Frequency")
    plt.show()

    # # heat map of correlations coefs for each column
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        df[["LATITUDE", "LONGITUDE", "x", "y"]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Feature correlation")
    plt.show()

    sns.scatterplot(
        data=df,
        x="LONGITUDE",
        y="LATITUDE",
        alpha=0.1,
        hue="ACCLASS",
    )
    plt.title("Accidents distribution")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.show()

    df_copy = df.copy()
    df_copy.drop(columns_to_drop, axis=1, inplace=True)

    df_copy["ACCLASS"] = df_copy["ACCLASS"].apply(lambda x: 1 if x == "Fatal" else 0)

    for column in boolean_columns:
        # encode with apply then We will know what is yes and what is no
        df_copy[column] = df_copy[column].apply(lambda x: 1 if x == "Yes" else 0)

    plt.figure(figsize=(10, 9))
    sns.heatmap(
        df_copy[[*boolean_columns, "ACCLASS"]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Feature correlation")
    plt.show()

    # # combination of histograms of all columns
    fig = plt.figure(figsize=(15, 10))
    for index, param in enumerate(boolean_columns):
        plt.subplot2grid((7, 2), (index // 2, index % 2))
        plt.hist(df[param].fillna("No"))
        plt.xlabel(param)
        plt.ylabel("Frequency")

    fig.tight_layout(pad=1)
    plt.show()

    mutual_info = mutual_info_classif(df_copy[boolean_columns], df_copy["ACCLASS"])
    mutual_info = pd.Series(mutual_info, index=boolean_columns)
    mutual_info.sort_values(ascending=False).plot(kind="bar", figsize=(15, 10))
    plt.title("Mutual information between features and target")
    plt.show()

    encoder = LabelEncoder()
    for column in [
        *cyclist_columns,
        *pedestrian_columns,
        *driver_columns,
        *env_columns,
        *direction_columns,
        *injury_columns,
        *vehicle_columns,
    ]:
        df_copy[column] = encoder.fit_transform(
            df_copy[column].fillna("Unknown").astype(str)
        )

    features = df_copy.drop(["ACCLASS", *location_columns], axis=1)
    mutual_info = mutual_info_classif(features, df_copy["ACCLASS"])
    mutual_info = pd.Series(mutual_info, index=features.columns)
    print(mutual_info)
    mutual_info.sort_values(ascending=False).plot(kind="bar", figsize=(15, 10))
    plt.title("Mutual information between features and target")
    plt.show()

    stat, p_val = chi2(features, df_copy["ACCLASS"])
    p_val = pd.Series(p_val, index=features.columns)
    print(p_val)
    p_val.sort_values(ascending=True).plot(kind="bar", figsize=(15, 10))
    plt.title("Chi2 test p-values")
    plt.show()

    # for chunk in columns_chunks:
    #     df_copy_copy = df_copy[chunk]

    #     df_copy_copy = pd.get_dummies(
    #         df_copy_copy, drop_first=True, dummy_na=True, dtype=int
    #     )
    #     df_copy_copy.fillna(0, inplace=True)
    #     df_copy_copy["ACCLASS"] = df["ACCLASS"].apply(
    #         lambda x: 1 if x == "Fatal" else 0
    #     )

    #     mutual_info = mutual_info_classif(
    #         df_copy_copy.drop(["ACCLASS"], axis=1), df_copy_copy["ACCLASS"]
    #     )
    #     mutual_info = pd.Series(
    #         mutual_info, index=df_copy_copy.drop(["ACCLASS"], axis=1).columns
    #     )
    #     mutual_info.sort_values(ascending=False).plot(kind="bar", figsize=(15, 10))
    #     plt.title("Mutual information between features and target")
    #     plt.show()

    # # combination of histograms of all columns
    # fig = plt.figure(figsize=(15, 10))
    # for index, param in enumerate(chunk):
    #     plt.subplot2grid((5, 2), (index // 2, index % 2))
    #     plt.hist(df[param])
    #     plt.xlabel(param)
    #     plt.ylabel("Frequency")

    # fig.tight_layout(pad=1)
    # plt.show()

    # # dependency between shape and size
    # plt.scatter(data_nadzeya['shape'], data_nadzeya['size'], alpha=0.1)
    # plt.title('Correlation between size and shape')
    # plt.xlabel('Shape')
    # plt.ylabel('Size')
    # plt.show()

    # # bar chart of how many shape types per cancer class
    # crosstab = pd.crosstab(
    #     data_nadzeya['class'], data_nadzeya['shape']
    # )
    # crosstab.plot(kind='bar')
    # plt.title('Shape distribution')
    # plt.xlabel('Breast cancer class')
    # plt.ylabel('Number of patients with particular shape')
    # plt.grid()
    # plt.show()


def main(file_path):
    df = pd.read_csv(file_path)

    describe_data(df)
    visualize_data(df)


main("TOTAL_KSI_6386614326836635957.csv")
