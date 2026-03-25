from abc import ABC, abstractmethod
from typing import *


class EvaluationMetric(ABC):

    @abstractmethod
    def __call__(
            self,
            ground_truth: List[str],
            predictions: List[str],
            sources: List[str],
    ) -> Dict[str, float]:
        pass
