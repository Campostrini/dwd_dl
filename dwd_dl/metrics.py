from typing import Tuple

import torch
import torchmetrics as tm
import sklearn.metrics as sklm
from torchmetrics import Precision, Recall, F1, PrecisionRecallCurve, ConfusionMatrix

import dwd_dl.cfg as cfg
from dwd_dl import log


class Contingency(tm.Metric):
    def __init__(self, class_number, persistence_as_metric=False):
        super().__init__()
        self.persistence_as_metric = persistence_as_metric
        self.add_state("class_number", default=torch.tensor(class_number, dtype=torch.int))
        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_negative", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("numel", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # preds, target = self.format_input(preds, target, self.persistence_as_metric)
        preds = self.class_number == preds
        target = self.class_number == target
        assert preds.shape == target.shape
        self.true_positive += torch.sum(preds & target)
        self.false_positive += torch.sum(preds & ~target)
        self.false_negative += torch.sum(~preds & target)
        self.true_negative += torch.sum(~preds & ~target)
        self.numel += torch.numel(preds)
        log.debug(f"{self.true_positive=} {self.false_positive=} {self.false_negative=} {self.true_negative=} {self.__class__.__name__} {self.class_number}")
        assert self.true_positive + self.false_positive + self.false_negative + self.true_negative == self.numel

    def compute(self):
        raise NotImplementedError

    @staticmethod
    def format_input(preds: torch.Tensor, target: torch.Tensor, persistence=False):
        if not persistence:
            preds = torch.argmax(preds, dim=1)
        else:
            if len(preds.shape) == 3:
                preds = torch.unsqueeze(preds, dim=1)  # to go from N, 256, 256 to N, 1, 256, 256 for persistence
                log.debug(f"{preds=}")
                if preds.is_floating_point():
                    device = preds.device
                    preds = torch.cat(
                        [(cfg.CFG.CLASSES[class_name][0] <= preds) &
                         (preds < cfg.CFG.CLASSES[class_name][1]) for class_name in cfg.CFG.CLASSES],
                        dim=1)
                    zeros = torch.zeros(size=(*preds.shape[:1], 1, *preds.shape[2:]), device=device)
                    log.debug(f"{preds=} after zeros")
                    for n in range(preds.shape[1]):
                        zeros += n * torch.unsqueeze(preds[:, n, ...], dim=1)
                    log.debug(f"{preds=} after range {preds.shape[1]}")
                    preds = zeros.to(dtype=torch.int).to(device=device)
        return preds, target


class TruePositive(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.true_positive


class TrueNegative(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.true_negative


class FalsePositive(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.false_positive


class FalseNegative(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.false_negative


class TruePositiveRatio(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.true_positive / self.numel


class TrueNegativeRatio(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.true_negative / self.numel


class FalsePositiveRatio(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.false_positive / self.numel


class FalseNegativeRatio(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.false_negative / self.numel


class PercentCorrect(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return (self.true_positive + self.true_negative) / self.numel


class HitRate(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.true_positive / (self.true_positive + self.false_negative)


class FalseAlarmRatio(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.false_positive / (self.true_positive + self.false_positive)


class CriticalSuccessIndex(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.true_positive / (self.true_positive + self.false_negative + self.false_positive)


class Bias(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return (self.true_positive + self.false_positive) / (self.true_positive + self.false_negative)


class PrecisionCustom(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.true_positive / (self.true_positive + self.false_positive)


class RecallCustom(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return self.true_positive / (self.true_positive + self.false_negative)


class F1Custom(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        precision = self.true_positive / (self.true_positive + self.false_positive)
        recall = self.true_positive / (self.true_positive + self.false_negative)
        return 2 * (precision * recall) / (precision + recall)


class HeidkeSkillScore(Contingency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative
        return 2 * (tp*tn + fp*fn) / ((tp + fn)*(fn + tn) + (tp + fp)*(fp + tn))


def modified_metric(cls):
    def new_update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if len(preds.shape) == 3:
            preds = torch.unsqueeze(preds, dim=1)  # to go from N, 256, 256 to N, 1, 256, 256 for persistence
            if preds.is_floating_point():
                device=preds.device
                preds = torch.cat(
                    [(cfg.CFG.CLASSES[class_name][0] <= preds) &
                     (preds < cfg.CFG.CLASSES[class_name][1]) for class_name in cfg.CFG.CLASSES],
                    dim=1)
                zeros = torch.zeros(size=(*preds.shape[:1], 1, *preds.shape[2:]), device=device)
                for n in range(preds.shape[1]):
                    zeros += n * torch.unsqueeze(preds[:, n, ...], dim=1)
                preds = zeros.to(dtype=torch.int).to(device=device)
        super(cls, self).update(preds, target)
    cls.update = new_update
    return cls


# @modified_metric
class Precision(Precision):
    def __init__(self, *args, **kwargs):
        try:
            kwargs.pop('persistence_as_metric')
        except KeyError:
            pass
        super().__init__(*args, **kwargs, mdmc_average='samplewise')


# @modified_metric
class Recall(Recall):
    def __init__(self, *args, **kwargs):
        try:
            kwargs.pop('persistence_as_metric')
        except KeyError:
            pass
        super().__init__(*args, **kwargs, mdmc_average='samplewise')


# @modified_metric
class F1(F1):
    def __init__(self, *args, **kwargs):
        try:
            kwargs.pop('persistence_as_metric')
        except KeyError:
            pass
        super().__init__(*args, **kwargs, mdmc_average='samplewise')


# @modified_metric
class PrecisionRecallCurve(PrecisionRecallCurve):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,)


# @modified_metric
class ConfusionMatrix(ConfusionMatrix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ConfusionMatrixScikit(tm.Metric):
    def __init__(self, persistence_as_metric=False):
        super().__init__()
        self.persistence_as_metric = persistence_as_metric
        self.add_state("confusion_matrix", default=torch.zeros(
            (len(cfg.CFG.CLASSES), len(cfg.CFG.CLASSES)), dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("numel", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # preds, target = Contingency.format_input(preds, target, self.persistence_as_metric)
        assert preds.shape == target.shape
        # tn, fp, fn, tp
        device = preds.device
        cm = sklm.confusion_matrix(
            target.cpu().numpy().ravel(), preds.cpu().numpy().ravel(), labels=range(len(cfg.CFG.CLASSES)))
        self.numel += torch.numel(preds)
        self.confusion_matrix += torch.tensor(cm, device=device)

    def compute(self):
        return self.confusion_matrix


class NormalizedConfusionMatrix(tm.Metric):
    def __init__(self, persistence_as_metric=False):
        super().__init__()
        self.persistence_as_metric = persistence_as_metric
        self.add_state("confusion_matrix", default=torch.zeros(
            (len(cfg.CFG.CLASSES), len(cfg.CFG.CLASSES)), dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("numel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("normalized_confusion_matrix", default=torch.zeros(
            (len(cfg.CFG.CLASSES), len(cfg.CFG.CLASSES)), dtype=torch.double), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # preds, target = Contingency.format_input(preds, target, self.persistence_as_metric)
        assert preds.shape == target.shape
        # tn, fp, fn, tp
        device = preds.device
        cm = sklm.confusion_matrix(
            target.cpu().numpy().ravel(), preds.cpu().numpy().ravel(), labels=range(len(cfg.CFG.CLASSES)))
        self.numel += torch.numel(preds)
        self.confusion_matrix += torch.tensor(cm, device=device)

    def compute(self):
        self.add_state("normalized_confusion_matrix", default=torch.tensor(
            [[element/row.sum() for element in row] for row in self.confusion_matrix], dtype=torch.double
        ), dist_reduce_fx='mean')
        return self.normalized_confusion_matrix
