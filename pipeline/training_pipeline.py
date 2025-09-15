from src.data_processing import DataProcessing
from src.model_training import ModelTraining

if __name__=="__main__":
    processor = DataProcessing("C:/Users/tegbe/Downloads/MLOps Project/Weather_Prediction/CODE/artifacts/raw/data.csv" , "C:/Users/tegbe/Downloads/MLOps Project/Weather_Prediction/CODE/artifacts/processed")
    processor.run()

    trainer = ModelTraining("C:/Users/tegbe/Downloads/MLOps Project/Weather_Prediction/CODE/artifacts/processed" , "C:/Users/tegbe/Downloads/MLOps Project/Weather_Prediction/CODE/artifacts/models")
    trainer.run()