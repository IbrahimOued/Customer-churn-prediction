import logging
import os

import click
import mlflow
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "data_preprocessing",
    "model_training",
    # "model_evaluation",
    # "model_deployment",
]

# Load environment variables
load_dotenv()



@click.command()
@click.option("--steps", default="all", type=str)
def run_pipeline(steps):


    EXPERIMENT_NAME = "Churn Prediction Pipeline"
    mlflow.set_tracking_uri("http://localhost") # important to run the experiment inside the docker experimentation env
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info("pipeline experiment_id: %s", experiment.experiment_id)

    # Steps to execute
    active_steps = steps.split(",") if steps != "all" else _steps
    logger.info("pipeline active steps to execute in this run: %s", active_steps)

    with mlflow.start_run(run_name='pipeline', nested=True) as active_run:
        if "data_preprocessing" in active_steps:
            preprocessing_run = mlflow.run(".", "data_preprocessing", parameters={})
            preprocessing_run = mlflow.tracking.MlflowClient().get_run(preprocessing_run.run_id)
            file_path_uri = preprocessing_run.data.params['input_dir'] # this is the path where the data is st
            clean_file_path_uri = preprocessing_run.data.params['output_dir']
            logger.info(f"Loading raw data from: {file_path_uri}")
            logger.info(f"Stored clean data to {clean_file_path_uri}")
            logger.info(preprocessing_run)

        if "model_training" in active_steps:
            training_run = mlflow.run(".", "model_training", parameters={"input_dir": file_path_uri})
            training_run = mlflow.tracking.MlflowClient().get_run(training_run.run_id)
            fine_tuning_run_id = training_run.data.params.get('mlflow_run_id', None)
            logger.info('Training run completed with run_id: %s', training_run.info.run_id)
            logger.info(training_run)

        if "model_evaluation" in active_steps:
            pass

        if "model_deployment" in active_steps:
            pass

        if "register_model" in active_steps:
            if fine_tuning_run_id is not None and fine_tuning_run_id != 'None':
                register_model_run = mlflow.run(".", "register_model", parameters={"mlflow_run_id": fine_tuning_run_id})
                register_model_run = mlflow.tracking.MlflowClient().get_run(register_model_run.run_id)
                logger.info(register_model_run)
            else:
                logger.info("no model to register since no trained model run id.")
    logger.info('finished mlflow pipeline run with a run_id = %s', active_run.info.run_id)


if __name__ == "__main__":
    run_pipeline()