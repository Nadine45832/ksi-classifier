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
    print("First 5 rows:\n", df.head(5))

    print("\nDataframe shape: ", df.shape)
    print("Column names: \n", df.columns)
    print("\nColumn types:\n", df.dtypes)

    print("\nMissing values:\n", df.isnull().sum(axis=0))
    print("\nColumn descriptions:\n", df.describe().round(2), "\n")

    for c in df.columns:
        text = f'Unique values for "{c}" column: '
        print(f"{text:<50} {len(df[c].unique()):>5}   {df[c].count():>5}")

    print("\nDuplicated rows: ", df.duplicated().sum())
    print("\nCorrelation matrix:\n", df.select_dtypes(include=["number"]).corr())

    # printing unique values for each column
    print(f"\nUnique values for DISTRICT: {df['DISTRICT'].unique()}")
    print(f"\nUnique values for ACCLOC: {df['ACCLOC'].unique()}")

    for column in boolean_columns:
        print(f"Unique values for {column}: {df[column].unique()}")


def visualize_data(df: pd.DataFrame) -> None:
    """
    This function performs data visualization on the given dataframe.
    :param df: pandas dataframe
    :return: None
    """
    # histogram for the accident class to see whether date is imbalanced or not
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

    # # heat map of correlations coefs for coordinates columns to check whether they are correlated or not
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        df[["LATITUDE", "LONGITUDE", "x", "y"]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Feature correlation")
    plt.show()

    # scatter plot of accidents distribution
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
    plt.legend(title="Injury class")
    plt.show()

    # copy dataframe to avoid modifying the original one
    df_copy = df.copy()
    df_copy.drop(columns_to_drop, axis=1, inplace=True)

    # create bar chart to show the number of non-null values per column and to see columns with missing values
    non_null_values = df_copy.notnull().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 9))
    non_null_values.plot(kind="bar")
    plt.title("Non-null values per column")
    plt.xlabel("Features")
    plt.ylabel("Number of non-null values")
    plt.show()

    # encode boolean columns
    df_copy["ACCLASS"] = df_copy["ACCLASS"].apply(lambda x: 1 if x == "Fatal" else 0)
    for column in boolean_columns:
        # encode with apply then We will know what is yes and what is no
        df_copy[column] = df_copy[column].apply(lambda x: 1 if x == "Yes" else 0)

    # heat map of the mutual information matrix between every boolean feature and target to check whether they are correlated or not
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        compute_mi_matrix(df_copy[[*boolean_columns, "ACCLASS"]]),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Feature correlation")
    plt.show()

    # combination of histograms of all boolean columns
    fig = plt.figure(figsize=(15, 10))
    for index, param in enumerate(boolean_columns):
        plt.subplot2grid((7, 2), (index // 2, index % 2))
        plt.hist(df[param].fillna("No"))
        plt.xlabel(param)
        plt.ylabel("Frequency")

    fig.tight_layout(pad=1)
    plt.show()

    categorical_columns = [
        *cyclist_columns,
        *pedestrian_columns,
        *driver_columns,
        *env_columns,
        *direction_columns,
        *injury_columns,
        *vehicle_columns,
    ]

    fig = plt.figure(figsize=(15, 10))
    for index, param in enumerate(
        [*injury_columns, *direction_columns, *vehicle_columns]
    ):
        plt.subplot2grid((4, 2), (index // 2, index % 2))
        plt.hist(df[param].fillna("Unknown"))
        plt.xlabel(param)
        plt.xticks(rotation=45)
        plt.ylabel("Frequency")

    fig.tight_layout(pad=1)
    plt.show()

    # encode categorical columns
    encoder = LabelEncoder()
    for column in categorical_columns:
        df_copy[column] = encoder.fit_transform(
            df_copy[column].fillna("Unknown").astype(str)
        )

    # heat map of the mutual information matrix between every feature and target to check whether they are correlated or not
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        compute_mi_matrix(df_copy[categorical_columns]),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Feature correlation")
    plt.show()

    # calculate mutual information for pedestrian columns
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        compute_mi_matrix(df_copy[[*pedestrian_columns, "PEDESTRIAN"]]),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Pedestrian feature correlation")
    plt.show()

    # calculate mutual information and chi2 test p-values for all categorical columns and target column
    features = df_copy.drop(["ACCLASS", *location_columns], axis=1)
    mutual_info = mutual_info_classif(features, df_copy["ACCLASS"])
    mutual_info = pd.Series(mutual_info, index=features.columns)
    mutual_info.sort_values(ascending=False).plot(kind="bar", figsize=(15, 10))
    plt.title("Mutual information between features and target")
    plt.show()

    stat, p_val = chi2(features, df_copy["ACCLASS"])
    p_val = pd.Series(p_val, index=features.columns)
    p_val.sort_values(ascending=True).plot(kind="bar", figsize=(15, 10))
    plt.title("Chi2 test p-values")
    plt.show()


def compute_mi_matrix(df):
    """
    This function computes the mutual information matrix between every feature in the dataframe.
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns).astype(float)
    for i in df.columns:
        for j in df.columns:
            if i == j:
                mi_matrix.loc[i, j] = 1.0
            else:
                mi_matrix.loc[i, j] = mutual_info_classif(
                    df[[i]], df[j], discrete_features=True
                )[0]
    return mi_matrix


def main(file_path):
    df = pd.read_csv(file_path)

    describe_data(df)
    visualize_data(df)


main("TOTAL_KSI_6386614326836635957.csv")
