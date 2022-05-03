import datetime as dt


class TSTracker:
    def __init__(self):
        self._used_timestamps_validation = []
        self._used_timestamps_training = []
        self._used_indices_validation = []
        self._used_indices_training = []

    def add_timestamp_validation(self, ts: dt.datetime):
        if ts in self._used_timestamps_validation:
            raise ValueError(f"{ts=} already in {self._used_timestamps_validation=}")

        self._used_timestamps_validation.append(ts)

    def add_timestamp_training(self, ts:dt.datetime):
        if ts in self._used_timestamps_training:
            raise ValueError(f"{ts=} already in {self._used_timestamps_training=}")

        self._used_timestamps_training.append(ts)

    def add_index_validation(self, idx):
        if idx in self._used_indices_validation:
            raise ValueError(f"{idx=} already in {self._used_indices_validation}")

        self._used_indices_validation.append(idx)

    def add_index_training(self, idx):
        if idx in self._used_indices_training:
            raise ValueError(f"{idx=} already in {self._used_indices_training}")

        self._used_indices_training.append(idx)

    def reset(self):
        for ts in self._used_timestamps_training:
            if ts in self._used_timestamps_validation:
                raise ValueError(f"{ts=} both in used training and used validation")

        self._used_timestamps_training = self._used_timestamps_validation = []
        self._used_indices_training = self._used_indices_validation = []
