import torchmetrics as tm


class CPIMetric(tm.Metric):
    def __init__(self):
        super().__init__()

    def update(self) -> None:
        pass

    def compute(self):
        pass
