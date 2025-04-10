import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    accuracy_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from imblearn.over_sampling import SMOTENC
from collections import Counter

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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

env_columns = [
    "ROAD_CLASS",
    "TRAFFCTL",
    "VISIBILITY",
    "LIGHT",
    "RDSFCOND",
    "INVTYPE",
]

location_columns = ["LATITUDE", "LONGITUDE"]

direction_columns = ["ACCLOC", "INITDIR"]

injury_columns = ["IMPACTYPE", "INVAGE", "INJURY"]

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

    # heat map of correlations coefs for coordinates columns to check whether they are correlated or not
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

    # columns to create cross tabulation
    columns_to_crossrab = [
        "ALCOHOL",
        "CYCLISTYPE",
        "CYCACT",
        "CYCCOND",
        "TRUCK",
        "MOTORCYCLE",
        "DISABILITY",
    ]

    df_copy["ACCLASS"] = df_copy["ACCLASS"].apply(
        lambda x: "Fatal" if x == "Fatal" else "Non-Fatal"
    )
    # create cross tabulation between feature and target class
    for column in columns_to_crossrab:
        crosstab = pd.crosstab(df_copy[column], df_copy["ACCLASS"], dropna=False)
        crosstab.plot(kind="bar", figsize=(10, 6))
        plt.title(f"Cross tabulation between {column} and ACCLASS")
        plt.xlabel(column)
        plt.ylabel("Frequency")
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

    # combination of histograms of some categorical columns the may correlate
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


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocesses the given dataframe.
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # copy target column
    # this colum has other values than "Fatal" and "Non-Fatal".
    # we need to transform it to binary values
    df_target = df_processed["ACCLASS"].apply(lambda x: 1 if x == "Fatal" else 0)

    columns_to_drop_based_on_analysys = [
        # correlate with INVTYPE and too many missing values
        "VEHTYPE",
        # too many missing values and too rare conditions for boolean categories
        "ALCOHOL",
        "CYCLISTYPE",
        "CYCACT",
        "CYCCOND",
        "TRUCK",
        "MOTORCYCLE",
        "DISABILITY",
        # chi2 test p-value is too high (more then 0.05)
        "VISIBILITY",
        "ROAD_CLASS",
        "PEDCOND",
        "PASSENGER",
        "MOTORCYCLE",
        "REDLIGHT",
        "INITDIR",
        "DISABILITY",
        "CYCCOND",
        "RDSFCOND",
        "MANOEUVER",
        "EMERG_VEH",
        # has very small mutial information with the target
        "TRSN_CITY_VEH",
        # Correlate with PEDTYPE
        "PEDACT",
        "PEDCOND",
    ]

    # drop columns that are not useful for the analysis and target column
    df_processed.drop(
        [*columns_to_drop, *columns_to_drop_based_on_analysys, "ACCLASS"],
        axis=1,
        errors="ignore",
        inplace=True,
    )

    print("Left columns:", df_processed.columns)

    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        df_processed,
        df_target,
        test_size=0.2,
        train_size=0.8,
        random_state=47,
    )

    # define pipelines for location columns
    # to fill missing values with the mean and scale the data
    # we will use MinMaxScaler because the data is not normally distributed
    location_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # define pipelines for boolean columns
    # to fill missing values with "No" and encode the data
    # we will use OrdinalEncoder because the data is yes or no (cgange it to 1 and 0)
    boolean_columns_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="No")),
            ("encoder", OrdinalEncoder()),
        ]
    )

    # define pipelines for condition columns
    # to fill missing values with the most frequent value and encode the data
    # we will use OneHotEncoder because the data is categorical to create a binary matrix
    condition_columns_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )

    # define pipelines for participant columns
    # to fill missing values with "None" and encode the data
    # we will use OneHotEncoder because the data is categorical to create a binary matrix
    participant_columns_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )

    # initialize preprocessor
    # we will use ColumnTransformer to apply different preprocessing pipelines to different columns
    preprocesser = ColumnTransformer(
        transformers=[
            (
                "boolean_columns",
                boolean_columns_pipeline,
                [
                    c
                    for c in boolean_columns
                    if c not in columns_to_drop_based_on_analysys
                ],
            ),
            (
                "coordinates_pipeline",
                location_pipeline,
                location_columns,
            ),
            (
                "conditions",
                condition_columns_pipeline,
                [
                    c
                    for c in [
                        *env_columns,
                        *direction_columns,
                        "IMPACTYPE",
                        "INVAGE",
                    ]
                    if c not in columns_to_drop_based_on_analysys
                ],
            ),
            (
                "participant_columns",
                participant_columns_pipeline,
                [
                    c
                    for c in [
                        *cyclist_columns,
                        *pedestrian_columns,
                        *driver_columns,
                        *vehicle_columns,
                        "INJURY",
                    ]
                    if c not in columns_to_drop_based_on_analysys
                ],
            ),
        ],
        remainder="drop",
    )

    # create a pipeline
    preproccesing_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocesser),
        ]
    )

    # fit and transform the training data
    x_train = preproccesing_pipeline.fit_transform(x_train, y_train)

    # transform the testing data
    x_test = preproccesing_pipeline.transform(x_test)

    # convert x_train to dataframe to use feature names
    feature_names = preproccesing_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    x_train = pd.DataFrame(x_train, columns=feature_names)

    # get the categorical features columns names
    categorical_features = [
        col for col in feature_names if col.split("__")[-1] not in location_columns
    ]

    # apply SMOTENC to balance the data
    smote = SMOTENC(categorical_features=categorical_features, random_state=47)

    print(f"Original dataset samples per class {Counter(y_train)}")

    # fit resampling to balance fatal and non-fatal accidents
    x_train, y_train = smote.fit_resample(x_train, y_train)

    print(f"Resampled dataset samples per class {Counter(y_train)}")

    return x_train, x_test, y_train, y_test, preproccesing_pipeline


def fit_models(x_train, x_test, y_train, y_test):
    # Define models to try
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=47
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=47
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=47
        ),
        "SVM": SVC(
            class_weight="balanced", probability=True, random_state=47, cache_size=2000
        ),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=47),
        "Ada Boost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(class_weight="balanced", max_depth=10),
            random_state=47,
        ),
        "Gradient Boost": GradientBoostingClassifier(
            random_state=47,
        ),
    }

    # Define hyperparameter grids for each model
    param_grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 1.5, 2.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear", "saga"],
        },
        "Decision Tree": {
            "max_depth": range(10, 100, 2),
            "min_samples_split": range(2, 10, 2),
            "criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "n_estimators": range(100, 1000, 5),
            "max_depth": range(10, 100, 2),
            "min_samples_split": range(2, 10, 2),
            "max_features": ["sqrt", "log2"],
            "criterion": ["gini", "entropy"],
        },
        "SVM": {
            "C": [0.01, 0.1, 1.0, 2.0],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3],
        },
        "Neural Network": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
        },
        "Ada Boost": {
            "n_estimators": range(50, 500, 10),
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
            "estimator__max_depth": range(1, 10, 2),
        },
        "Gradient Boost": {
            "n_estimators": range(100, 500, 50),
            "learning_rate": [0.01, 0.1, 0.5, 1],
            "max_depth": range(3, 10, 2),
            "subsample": [0.8, 0.9, 1.0],
        },
    }

    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Grid search for hyperparameter tuning
        print(f"Performing hyperparameter tuning with Grid Search...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
        grid_search = RandomizedSearchCV(
            model,
            param_grids[model_name],
            cv=skf,
            n_iter=20,
            refit=True,
            random_state=47,
            scoring="recall",
            n_jobs=-1,
            verbose=3,
        )
        # recall is more important in our case becuase we need to find as many true positive as possible
        # false positives are acceptable

        grid_search.fit(x_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best parameters: {best_params}")

        # Make predictions
        y_pred = best_model.predict(x_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Calculate ROC curve and AUC
        y_pred_proba = best_model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)

        # Store results
        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_score,
            "best_params": best_params,
            "model": best_model,
        }

        # Print results
        print(f"Results for {model_name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc_score:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    {conf_matrix[0]}")
        print(f"    {conf_matrix[1]}")

    best_model_name = max(results, key=lambda x: results[x]["accuracy"])
    print(f"Best model: ", best_model_name)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        results[best_model_name]["fpr"],
        results[best_model_name]["tpr"],
        label=f"{best_model_name} (AUC = {results[best_model_name]['auc']:.4f})",
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    joblib.dump(results[best_model_name]["model"], "ksi_classifier.pkl")


def main(file_path):
    df = pd.read_csv(file_path)

    describe_data(df)
    # visualize_data(df)
    x_train, x_test, y_train, y_test, preproccesing_pipeline = data_preprocessing(df)
    fit_models(x_train, x_test, y_train, y_test)


main("TOTAL_KSI_6386614326836635957.csv")
