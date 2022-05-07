import os

import matplotlib.pyplot as plt

import dwd_dl.cfg as cfg
from dwd_dl import log

cfg.initialize2(True, True, True)

counts_training = {
    '0': 3445038421,
    '0.1': 321439313,
    '1': 69894021,
    '2.5': 16096469,
}

counts_validation = {
    '0': 410481916,
    '0.1': 44098083,
    '1': 9562197,
    '2.5': 2211980,
}

counts_test = {
    '0': 406964290,
    '0.1': 44347124,
    '1': 10215710,
    '2.5': 2074540,
}


def normalize(counts: dict):
    counts_sum = sum(counts.values())
    return {k: counts[k]/counts_sum for k in counts}


fig, axs = plt.subplots(1, 1, figsize=(8.27, 11.69/2),)
width = 0.25
for counts, title, i in zip(
        (counts_training, counts_validation, counts_test), ("Training", "Validation", "Test"), range(3)):
    counts = normalize(counts)
    log.info(f"{title=}, {counts=}")
    x_axis = [x +i*width for x in range(len(counts))]
    axs.bar(x_axis, counts.values(), label=title, width=width)
axs.set_xticks([x-width for x in x_axis], list(counts_training.keys()))
axs.set_ylabel("Relative Frequency")

axs.set_xlabel("Class Label")
plt.legend(loc="upper right")
plt.savefig(os.path.join(cfg.CFG.RADOLAN_ROOT, 'figures', 'Class Frequencies Bar.png'))
