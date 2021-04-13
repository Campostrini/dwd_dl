from typing import Tuple

import torch
import torchmetrics as tm


class Contingency(tm.Metric):
    def __init__(self, class_number):
        super().__init__()
        self.class_number = torch.tensor(class_number, dtype=torch.int)
        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_negative", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("numel", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds, target = self._format_input(preds, target)
        assert preds.shape == target.shape
        self.true_positive += torch.sum(preds == target)
        self.false_positive += torch.sum(preds == ~target)
        self.false_negative += torch.sum(~preds == target)
        self.true_negative += torch.sum(~preds == ~target)
        self.numel += torch.numel(preds)
        assert self.true_positive + self.false_positive + self.false_negative + self.true_negative == self.numel

    def compute(self):
        raise NotImplementedError

    def _format_input(self, preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor]:
        preds = torch.argmax(preds, dim=1)
        preds = self.class_number == preds
        target = self.class_number == target
        return preds, target


class PercentCorrect(Contingency):
    def __init__(self, class_number):
        super().__init__(class_number)

    def compute(self):
        return (self.true_positive + self.true_negative) / self.numel


class HitRate(Contingency):
    def __init__(self, class_number):
        super().__init__(class_number)

    def compute(self):
        return self.true_positive / (self.true_positive + self.false_negative)


class FalseAlarmRatio(Contingency):
    def __init__(self, class_number):
        super().__init__(class_number)

    def compute(self):
        return self.false_positive / (self.true_positive + self.false_positive)


class CriticalSuccessIndex(Contingency):
    def __init__(self, class_number):
        super().__init__(class_number)

    def compute(self):
        return self.true_positive / (self.true_positive + self.false_negative + self.false_positive)


class Bias(Contingency):
    def __init__(self, class_number):
        super().__init__(class_number)

    def compute(self):
        return (self.true_positive + self.false_positive) / (self.true_positive + self.false_negative)


class HeidkeSkillScore(Contingency):
    def __init__(self, class_number):
        super().__init__(class_number)

    def compute(self):
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative
        return 2 * (tp*tn + fp*fn) / ((tp + fn)*(fn + tn) + (tp + fp)*(fp + tn))
