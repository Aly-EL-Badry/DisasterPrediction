from zenml import pipeline
from steps.dataHandling import data_cleaning_step, outlier_handling_step, feature_engineering_step, scaling_step, save_to_csv_step
from steps.dataIngestion import data_ingestion_step
import yaml
import os

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


data_path = os.path.join(config['project']['root'], config['project']['data_path'])
output_path = os.path.join(config['project']['root'], config['project']['output_path'] )
drop_cols = config['data_cleaning']['drop_columns']
remove_cols = config['outlier_handling']['remove_columns']
cap_cols = config['outlier_handling']['cap_columns']
transformations = config['feature_engineering']['transformations']
columns_to_scale = config['scaling']['columns_to_scale']

@pipeline
def data_preprocessing_pipeline():
    
    data = data_ingestion_step(DATA_PATH=data_path)
    data = data_cleaning_step(data, drop_cols=drop_cols)
    data = outlier_handling_step(data, remove_cols=remove_cols, cap_cols=cap_cols)
    data = feature_engineering_step(data, transformations=transformations)
    data = scaling_step(data, columns_to_scale=columns_to_scale)
    save_to_csv_step(data,output_path)
    