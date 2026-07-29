import csv
import itertools
import random
from pathlib import Path


SAMPLE_COUNT = 48
RANDOM_SEED = 5060
BASELINE = (128, 128, 128, 16, 64, 64, 4, 8, 4)

SEARCH_SPACE = {
    "num_threads": [128, 256],
    "bm": [64, 128],
    "bn": [64, 128, 256],
    "bk": [8, 16, 32],
    "wm": [32, 64, 128],
    "wn": [32, 64, 128],
    "wniter": [1, 2, 4],
    "tm": [4, 8],
    "tn": [4, 8],
}


def is_legal(c):
    threads, bm, bn, bk, wm, wn, wniter, tm, tn = c
    if bm % wm or bn % wn:
        return False
    if (bm // wm) * (bn // wn) != threads // 32:
        return False

    denominator = 32 * tm * tn * wniter
    if (wm * wn) % denominator:
        return False
    wmiter = (wm * wn) // denominator
    if wmiter == 0 or wm % wmiter or wn % wniter:
        return False

    if (threads * 4) % bk or (threads * 4) % bn:
        return False
    if bn % (16 * tn) or bm % (16 * tm):
        return False
    if (bm * bk) % (4 * threads) or (bn * bk) % (4 * threads):
        return False

    shared_memory_bytes = (bm * bk + bk * bn) * 4
    return shared_memory_bytes <= 48 * 1024


def main():
    names = list(SEARCH_SPACE)
    candidates = itertools.product(*(SEARCH_SPACE[name] for name in names))
    legal = [candidate for candidate in candidates if is_legal(candidate)]

    output = Path(__file__).with_name("k10_legal_configs.csv")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(names)
        writer.writerows(legal)

    if BASELINE not in legal:
        raise RuntimeError("The baseline configuration is not legal.")

    random_generator = random.Random(RANDOM_SEED)
    remaining = [candidate for candidate in legal if candidate != BASELINE]
    sampled = [BASELINE]
    sampled.extend(random_generator.sample(remaining, SAMPLE_COUNT - 1))

    sample_output = Path(__file__).with_name("k10_sampled_configs.csv")
    with sample_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(names)
        writer.writerows(sampled)

    print(f"Generated {len(legal)} legal configurations: {output}")
    print(
        f"Sampled {len(sampled)} configurations with seed {RANDOM_SEED}: "
        f"{sample_output}"
    )


if __name__ == "__main__":
    main()
