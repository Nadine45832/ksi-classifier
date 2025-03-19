from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pickle


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

    # encode categorical columns
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

    # heat map of the mutual information matrix between every feature and target to check whether they are correlated or not
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        compute_mi_matrix(
            df_copy[
                [
                    *cyclist_columns,
                    *pedestrian_columns,
                    *driver_columns,
                    *env_columns,
                    *direction_columns,
                    *injury_columns,
                    *vehicle_columns,
                ]
            ]
        ),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Feature correlation")
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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocesses the data for modeling:
    - Handles missing values
    - Encodes categorical variables
    - Performs feature selection
    - Prepares data for training

    :param df: pandas dataframe with raw data
    :return: preprocessed dataframe ready for modeling
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    
    # Step 1: Drop unnecessary columns
    print(f"\nDropping columns that are not useful for prediction...")
    columns_to_drop_available = [col for col in columns_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=columns_to_drop_available)
    print(f"Remaining columns: {len(df_processed.columns)}")
    
    # Step 2: Handle missing values
    print("\nHandling missing values...")
    
    # For boolean columns, fill NaN with "No"
    for col in boolean_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna("No")
            print(f"Filled missing values in {col} with 'No'")
    
    # For categorical columns, fill NaN with "Unknown"
    categorical_cols = [
        *cyclist_columns,
        *pedestrian_columns,
        *driver_columns,
        *env_columns,
        *direction_columns,
        *injury_columns,
        *vehicle_columns,
    ]
    
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna("Unknown")
            print(f"Filled missing values in {col} with 'Unknown'")
    
    # For location columns, use median to fill NaN
    for col in location_columns:
        if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
            print(f"Filled missing values in {col} with median: {median_value:.4f}")
    
    # Step 3: Encode categorical variables
    print("\nEncoding categorical variables...")
    
    # Encode target variable
    if 'ACCLASS' in df_processed.columns:
        df_processed['ACCLASS_ENCODED'] = df_processed['ACCLASS'].apply(
            lambda x: 1 if x == 'Fatal' else 0
        )
        print("Encoded ACCLASS as binary (1 for Fatal, 0 for Non-Fatal)")
    
    # Encode boolean columns
    for col in boolean_columns:
        if col in df_processed.columns:
            df_processed[f"{col}_ENCODED"] = df_processed[col].apply(
                lambda x: 1 if x == "Yes" else 0
            )
            print(f"Encoded {col} as binary (1 for Yes, 0 for No)")
    
    # Encode other categorical columns using LabelEncoder
    encoder = LabelEncoder()
    for col in categorical_cols:
        if col in df_processed.columns:
            # Convert to string and ensure all values are strings
            df_processed[col] = df_processed[col].astype(str)
            # Apply label encoding
            df_processed[f"{col}_ENCODED"] = encoder.fit_transform(df_processed[col])
            print(f"Applied label encoding to {col}")
    
    # Step 4: Feature selection based on mutual information
    print("\nPerforming feature selection based on mutual information...")
    
    # Get all encoded columns
    encoded_cols = [col for col in df_processed.columns if col.endswith('_ENCODED') and col != 'ACCLASS_ENCODED']
    
    # Calculate mutual information with target
    if 'ACCLASS_ENCODED' in df_processed.columns and len(encoded_cols) > 0:
        target = df_processed['ACCLASS_ENCODED']
        features = df_processed[encoded_cols]
        
        mi_scores = mutual_info_classif(features, target)
        mi_scores = pd.Series(mi_scores, index=encoded_cols)
        
        # Sort features by mutual information score
        sorted_features = mi_scores.sort_values(ascending=False)
        
        print("\nTop 10 features by mutual information score:")
        for feature, score in sorted_features.head(10).items():
            print(f"{feature}: {score:.4f}")
        
        # Create a DataFrame with selected features and target
        important_features = sorted_features.head(20).index.tolist()
        important_features.append('ACCLASS_ENCODED')
        
        df_selected = df_processed[important_features]
        
        print(f"\nSelected {len(important_features)-1} features for modeling")
    else:
        print("Couldn't perform feature selection: missing encoded columns or target")
        df_selected = df_processed
    
    # Step 5: Check for class imbalance
    if 'ACCLASS_ENCODED' in df_selected.columns:
        class_counts = df_selected['ACCLASS_ENCODED'].value_counts()
        print("\nClass distribution:")
        print(f"Non-Fatal (0): {class_counts.get(0, 0)}")
        print(f"Fatal (1): {class_counts.get(1, 0)}")
        
        imbalance_ratio = class_counts.get(0, 0) / class_counts.get(1, 0) if class_counts.get(1, 0) > 0 else float('inf')
        print(f"Imbalance ratio (Non-Fatal to Fatal): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            print("Warning: Severe class imbalance detected. Consider using SMOTE or other resampling techniques.")
    
    print("\nPreprocessing completed.")
    return df_selected


def build_and_evaluate_models(data: pd.DataFrame) -> dict:
    """
    Build, train, and evaluate multiple classification models.
    
    :param data: Preprocessed DataFrame with features and target
    :return: Dictionary containing trained models and their performance metrics
    """
    print("\n" + "="*80)
    print("MODEL BUILDING AND EVALUATION")
    print("="*80)
    
    # Separate features and target
    if 'ACCLASS_ENCODED' not in data.columns:
        raise ValueError("Target column 'ACCLASS_ENCODED' not found in data")
    
    X = data.drop('ACCLASS_ENCODED', axis=1)
    y = data['ACCLASS_ENCODED']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nData split complete:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Check for class imbalance and apply SMOTE if needed
    class_counts = y_train.value_counts()
    imbalance_ratio = class_counts.get(0, 0) / class_counts.get(1, 0) if class_counts.get(1, 0) > 0 else float('inf')
    
    if imbalance_ratio > 10:
        print(f"\nApplying SMOTE to handle class imbalance (ratio: {imbalance_ratio:.2f})...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Original training data shape: {X_train.shape}")
        print(f"Resampled training data shape: {X_train_resampled.shape}")
        
        # Update training data
        X_train = X_train_resampled
        y_train = y_train_resampled
    
    # Define models to try
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }
    
    # Define hyperparameter grids for each model
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        }
    }
    
    # Results storage
    results = {}
    best_models = {}
    
    # Create a pipeline with preprocessing steps
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Apply preprocessing to datasets
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Grid search for hyperparameter tuning
        print(f"Performing hyperparameter tuning with Grid Search...")
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name], 
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_processed, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        
        # Make predictions
        y_pred = best_model.predict(X_test_processed)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc_score,
            'best_params': best_params
        }
        
        best_models[model_name] = best_model
        
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
    
    # Find the best performing model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"AUC: {results[best_model_name]['auc']:.4f}")
    print("="*80)
    
    # Create a pipeline with preprocessing and the best model
    final_pipeline = Pipeline([
        ('preprocessor', preprocessing_pipeline),
        ('classifier', best_models[best_model_name])
    ])
    
    # Fit the pipeline on the full training data
    final_pipeline.fit(X_train, y_train)
    
    # Return all results and the best model
    return {
        'results': results,
        'best_model_name': best_model_name,
        'best_model': final_pipeline,
        'feature_names': X.columns.tolist(),
        'preprocessor': preprocessing_pipeline
    }


def save_model(model_info, filename="ksi_model.pkl"):
    """
    Save the trained model and related information to a file
    
    :param model_info: Dictionary containing model information
    :param filename: Output filename for the model
    """
    with open(filename, 'wb') as file:
        pickle.dump(model_info, file)
    
    print(f"\nModel saved to {filename}")


def plot_model_comparisons(results):
    """
    Plot comparison graphs for all trained models
    
    :param results: Dictionary with model results
    """
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    # Bar chart comparison of metrics
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        plt.bar(x + (i - 2) * width, values, width, label=metric.capitalize())
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # ROC curves for all models
    plt.figure(figsize=(10, 8))
    
    for model_name in model_names:
        fpr = results[model_name]['fpr']
        tpr = results[model_name]['tpr']
        auc_score = results[model_name]['auc']
        
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.show()


def feature_importance_analysis(model_info):
    """
    Analyze feature importance from the best model if available
    
    :param model_info: Dictionary with model information
    """
    best_model = model_info['best_model']
    best_model_name = model_info['best_model_name']
    
    # Extract the classifier from the pipeline
    classifier = best_model.named_steps['classifier']
    
    # Check if the model supports feature importance
    if best_model_name in ['Random Forest', 'Decision Tree']:
        # Get feature importance scores
        importances = classifier.feature_importances_
        
        # Map importances to feature names
        feature_names = model_info['feature_names']
        
        # Create a DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Visualize top 15 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title(f'Top 15 Feature Importance for {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()
        
        print("\nTop 10 most important features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    elif best_model_name == 'Logistic Regression':
        # For logistic regression we can get coefficients
        coefficients = classifier.coef_[0]
        
        # Map coefficients to feature names
        feature_names = model_info['feature_names']
        
        # Create a DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': np.abs(coefficients)  # Use absolute value for ranking
        })
        
        # Sort by absolute coefficient value
        feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
        
        # Visualize top 15 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance.head(15))
        plt.title(f'Top 15 Feature Coefficients for {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_coefficients.png')
        plt.show()
        
        print("\nTop 10 features with highest coefficient magnitude:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"{row['Feature']}: {row['Coefficient']:.4f}")


def main(file_path):
    df = pd.read_csv(file_path)

    # Data exploration
    describe_data(df)
    
    # Data visualization
    visualize_data(df)
    
    # Data preprocessing
    processed_df = preprocess_data(df)
    
    # Model building and evaluation
    model_info = build_and_evaluate_models(processed_df)
    
    # Plot model comparisons
    plot_model_comparisons(model_info['results'])
    
    # Analyze feature importance
    feature_importance_analysis(model_info)
    
    # Save the best model
    save_model(model_info)
    
    # Save the preprocessed data for further modeling
    processed_df.to_csv("processed_ksi_data.csv", index=False)
    print(f"\nPreprocessed data saved to processed_ksi_data.csv")
    
    return processed_df, model_info


if __name__ == "__main__":
    main("TOTAL_KSI_6386614326836635957.csv")
