from pipelines.dataProcesssingPipeline import data_preprocessing_pipeline
from pipelines.trainingPipeline import training_pipeline
import mlflow
import dagshub
dagshub.init(repo_owner='Aly-EL-Badry', repo_name='DisasterPrediction', mlflow=True)

if __name__ == "__main__":
    mlflow.set_experiment("disaster-prediction v1")

    data_preprocessing_pipeline()
    training_pipeline()