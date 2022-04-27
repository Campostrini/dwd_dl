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


fig, axs = plt.subplots(3, 1, figsize=(8.27, 11.69),)
for ax, counts, title in zip(
        axs, (counts_training, counts_validation, counts_test), ("Training", "Validation", "Test")):
    counts = normalize(counts)
    log.info(f"{title=}, {counts=}")
    ax.bar(range(len(counts)), counts.values(),)
    ax.set_xticks(range(len(counts.keys())), list(counts.keys()))
    ax.set_title(title)
    ax.set_ylabel("Relative Frequency")

ax.set_xlabel("Class Label")

plt.savefig(os.path.join(cfg.CFG.RADOLAN_ROOT, 'figures', 'Class Frequencies Bar.png'))
