import os
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.model = xgb.XGBClassifier()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_path = "C:/Users/tegbe/Downloads/MLOps Project/Weather_Prediction/CODE/artifacts/path/train"
        self.test_path = "C:/Users/tegbe/Downloads/MLOps Project/Weather_Prediction/CODE/artifacts/path/test"

        os.makedirs(self.output_path,exist_ok=True)

        logger.info("Model Training intilaized...")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.input_path,"X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.input_path,"X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.input_path,"y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.input_path,"y_test.pkl"))

            logger.info("Data loaded sucesfully...")

        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load Data ", e)
        
    def train_model(self):
        try:
            self.model.fit(self.X_train,self.y_train)

            joblib.dump(self.model , os.path.join(self.output_path,"model.pkl"))

            logger.info("Training and saving of model done...")
        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model ", e)
        
    def eval_model(self):
        try:
            training_score = self.model.score(self.X_train,self.y_train)
            logger.info(f"Training model score : {training_score} ")

            y_pred = self.model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test,y_pred)
            precision = precision_score(self.y_test,y_pred,average="weighted")
            recall = recall_score(self.y_test,y_pred,average="weighted")
            f1 = f1_score(self.y_test,y_pred,average="weighted")

            logger.info(f"Accuracy : {accuracy} ; Precision : {precision} ; Recall : {recall}  : F1-Score : {f1}")

            logger.info("Model evaluation done..")

            return {
                "accuracy"  :   accuracy,
                "precision" : precision,
                "recall"    : recall,
                "f1"    : f1
            }

        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model ", e)
        
    def run(self):

        with mlflow.start_run():
            logger.info("Starting our MLFLOW experimentation...")
            logger.info("Logging the training and testing to MLFLOW..")
            mlflow.log_artifact(self.train_path, artifact_path="datasets")
            mlflow.log_artifact(self.test_path, artifact_path="datasets")

            self.load_data()
            self.train_model()
           # self.eval_model()
            metrics = self.eval_model()

            mlflow.log_artifact(self.output_path)

            logger.info("Logging params and metrics to MLFLOW...")
            mlflow.log_params(self.model.get_params())
            mlflow.log_metrics(metrics)

            logger.info("Model training and Evaluation Done...")


if __name__ == "__main__":
    trainer = ModelTraining("C:/Users/tegbe/Downloads/MLOps Project/Weather_Prediction/CODE/artifacts/processed", "C:/Users/tegbe/Downloads/MLOps Project/Weather_Prediction/CODE/artifacts/models")
    trainer.run()

        

            

        
