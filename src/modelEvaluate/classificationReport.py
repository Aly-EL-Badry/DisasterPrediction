from .base import ModelEvaluate
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class classificationReportEvaluation(ModelEvaluate):
    def evaluate(self, y_test, y_predict):
        """
        Evaluate the model with a classification report.

        Parameters
        ----------
        X_test : array-like
            The features to use for evaluation.
        y_test : array-like
            The target to use for evaluation.

        Returns
        -------
        str
            The classification report as a string.

        Raises
        ------
        Exception
            If any error occurs during evaluation.
        """
        try:
            logger.info("Evaluating model...")
            report = classification_report(y_test, y_predict)
            logger.info("Model evaluation completed.")
            logger.info(report)
            return report
        except Exception as e:
            logger.error(f"Error in evaluating model: {e}")
            raise
            
        