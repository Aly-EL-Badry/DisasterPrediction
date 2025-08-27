from abc import ABC, abstractmethod

class ModelEvaluate(ABC):
    @abstractmethod
    def evaluate(self, y_test, y_predict):
    
        """
        Evaluate the model with given test and predicted values.

        Parameters
        ----------
        y_test : array-like
            The target values to evaluate the model against.
        y_predict : array-like
            The predicted values of the model.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any error occurs during evaluation.
        """
        pass