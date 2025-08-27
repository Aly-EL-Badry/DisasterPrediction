from pipelines.dataProcesssingPipeline import data_preprocessing_pipeline
from pipelines.trainingPipeline import training_pipeline

if __name__ == "__main__":
    data_preprocessing_pipeline()
    training_pipeline()