PHRASES = []


def get_phrases():
    with open("./data/rejection_phrases", "r") as f:
        for line in f.readlines():
            line = line.strip()
            PHRASES.append(line)


def is_valid_answer(sentence: str) -> bool:
    if not PHRASES:
        get_phrases()

    for item in PHRASES:
        if sentence.startswith(item):
            return False

    return True
