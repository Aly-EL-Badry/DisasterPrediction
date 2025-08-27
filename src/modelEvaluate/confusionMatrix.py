from .base import ModelEvaluate
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class confusionMatrixEvaluation(ModelEvaluate):
    def evaluate(self, y_test, y_predict):
        try:
            logger.info("Evaluating model...")
            conf_matrix = confusion_matrix(y_test, y_predict)
            logger.info("Model evaluation completed.")
            logger.info(conf_matrix)
            return conf_matrix
        except Exception as e:
            logger.error(f"Error in evaluating model: {e}")
            raise
            
        