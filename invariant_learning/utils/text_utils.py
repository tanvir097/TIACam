import random


def get_mismatched_texts(batch_texts):
    batch_size = len(batch_texts)

    fake_texts = []

    for i in range(batch_size):
        candidates = list(range(batch_size))
        candidates.remove(i)

        j = random.choice(candidates)

        fake_texts.append(batch_texts[j])

    return fake_texts