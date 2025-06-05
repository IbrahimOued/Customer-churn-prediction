import click
import mlflow
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@click.command(help="This program ingests data from a local source and clean it up for further processing.")
@click.option("--input_dir", default="datasets", help="This is the path where the data will be loaded from.")
@click.option("--input_filename", default="Telco-Customer-Churn.csv", help="This is the name of the file to be load.")
@click.option("--output_dir", default="datasets", help="This is the path where the cleaned data will be saved.")
@click.option("--pipeline_run_name", default="data-preprocessing", help="This is a mlflow run name.")
def task(input_dir, input_filename, output_dir, pipeline_run_name):
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        logger.info("Loading data from  %s", input_dir)
        df = pd.read_csv(input_dir + "/" + input_filename)
        # 🧹 Data Cleaning
        df.replace(" ", np.nan, inplace=True)
        df.dropna(inplace=True)
        # 🔄 Convert TotalCharges to numeric
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        # 🎯 Encode target
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        # 🛠️ Drop customerID (not predictive)
        df.drop('customerID', axis=1, inplace=True)
        # 🔁 Encode categorical variables
        cat_cols = df.select_dtypes(include='object').columns
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # 💾 Save cleaned data
        output_filename = "cleaned_telco_data.csv"
        df_encoded.to_csv(output_dir + "/" + output_filename, index=False)

        logger.info(f"✅ Cleaned data saved to {output_dir}/{output_filename}")
        
        # mlflow.log_artifact(output_dir + "/" + output_filename, artifact_path="data")
        mlflow.log_param("raw_dataset", input_dir + "/" + input_filename)
        mlflow.log_param("output_path", output_dir + "/" + output_filename)
        mlflow.set_tag('pipeline_step', __file__)
        mlflow.set_tag('pipeline_run_name', pipeline_run_name)
        mlflow.set_tag('mlflow_run_id', mlrun.info.run_id)
        logger.info("MLflow run completed with run_id: %s", mlrun.info.run_id)

    logger.info("finished clearning the data and saving it to %s", output_dir + "/" + output_filename)




if __name__ == '__main__':
    task()