import click
import mlflow
import pandas as pd
from tqdm.auto import tqdm
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@click.command(help="This program trains a model to predict customer churn based on cleaned data.")
@click.option("--input_dir", default="datasets", help="This is the path where the data will be loaded from.")
@click.option("--input_filename", default="cleaned_telco_data.csv", help="This is the name of the file to be load.")
@click.option("--scaler_dir", default="models", help="This is the path where the scaler will be saved.")
@click.option("--model_dir", default="models", help="This is the path where the model will be saved.")
@click.option("--max_iterations", default=1000, help="Maximum iterations for Logistic Regression.")
@click.option("--n_estimators", default=100, help="Number of estimators for Random Forest.")
@click.option("--evaluation_metric", default="logloss", help="Evaluation metric for XGBoost.")
@click.option("--test_size", default=0.2, help="Proportion of the dataset to include in the test split.")
@click.option("--random_state", default=42, help="Random state for reproducibility.")
@click.option("--pipeline_run_name", default="data-training", help="This is a mlflow run name.")
def task(input_dir, input_filename, scaler_dir, model_dir, max_iterations, n_estimators, pipeline_run_name, evaluation_metric, test_size, random_state):
    
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        logger.info("Loading clean data from  %s", input_dir)
        
        # 📦 Load Data
        try:
            df = pd.read_csv(input_dir + "/" + input_filename)
            logger.info(f"Shape after loading: {df.shape}")
        except FileNotFoundError:
            logger.error("Error: The CSV file was not found. Please check the path: datasets/cleaned_telco_data.csv")
            exit() # Exit if the file isn't found, as there's no data to process

        if df.empty:
            logger.error("Error: The DataFrame is empty immediately after reading the CSV. The CSV might be empty or corrupted.")
            exit()

        # 🎯 Encode Target and handle NaNs
        # Inspect unique values to catch unexpected entries
        logger.info("Unique values in 'Churn' before mapping:")
        logger.info(df['Churn'].unique()) # Uncomment to inspect


        # Separate features (X) and target (y)
        X = df.drop(columns=['Churn'])
        y = df['Churn']

        logger.info(f"\nFinal X shape before split: {X.shape}")
        logger.info(f"Final y shape before split: {y.shape}")
        
        
        # 🔄 Train/Test Split
        # Crucial check: Is X empty? Is y empty?
        if X.empty or y.empty:
            logger.error("Error: X or y is empty before train_test_split. This means all your data was lost during previous processing steps.")
            logger.info("Please review the 'Shape after...' logger.info statements above to identify where the data was lost.")
        else:
            # ✨ Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Verify the shapes of your new datasets
            logger.info("\n---")
            logger.info("Data Split Successful!")
            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"y_test shape: {y_test.shape}")
            logger.info("---")

            # 🧪 Scale Features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            scaler_name = "scaler.joblib"
            model_name = "best_churn_model.joblib"

            joblib.dump(scaler, f"{scaler_dir}/{scaler_name}")

            mlflow.log_artifact(f"{scaler_dir}/{scaler_name}", artifact_path="scalers")
            mlflow.log_param("scaler_name", scaler_name)

        
        best_score = 0
        best_model = None
        best_model_name = ""

        # 📌 Model Candidates
        models = {
            "LogisticRegression": LogisticRegression(max_iter=max_iterations),
            "RandomForest": RandomForestClassifier(n_estimators=n_estimators),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric=evaluation_metric)
        }


        for name, model in tqdm(models.items(), desc="Training Models"):
            logger.info(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            acc = model.score(X_test_scaled, y_test)
            roc = roc_auc_score(y_test, y_proba)

            mlflow.log_metric(f"{name}_accuracy", acc)
            mlflow.log_metric(f"{name}_roc_auc", roc)
            # mlflow.log_params(model.get_params())

            logger.info(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")

            if roc > best_score:
                best_score = roc
                best_model = model
                best_model_name = name
                best_run_id = mlflow.active_run().info.run_id

        logger.info(f"\n✅ Best Model: {best_model_name} with ROC-AUC: {best_score:.4f}")
        joblib.dump(best_model, f"{model_dir}/{model_name}")

        mlflow.sklearn.log_model(model, model_dir, input_example=X_train.head(1))
        logger.info(f"Model saved to {model_dir}/{model_name}")
        mlflow.log_artifact(f"{model_dir}/{model_name}", artifact_path="models")
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_param("best_model_score", best_score)
        mlflow.log_param("mlflow_run_id", mlrun.info.run_id)
        mlflow.set_tag('pipeline_step', __file__)
        mlflow.set_tag('pipeline_run_name', pipeline_run_name)
        logger.info("MLflow run completed with run_id: %s", mlrun.info.run_id)

        logger.info(f"Best model saved and run ID for API: {best_run_id}")
        logger.info(f"finished training the model and saving it to {model_dir}/{model_name}")


if __name__ == '__main__':
    task()
