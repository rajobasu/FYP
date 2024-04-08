PHRASES = []


def get_phrases():
    with open("./data/rejection_phrases.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            PHRASES.append(line)


def is_valid_answer(sentence: str) -> bool:
    if not PHRASES:
        get_phrases()

    # check for phrases.
    for item in PHRASES:
        if item in sentence:
            return False

    # check if the output is too short. we assume that a correct output will be atleast 20 words long.
    if len(sentence.split()) <= 20:
        return False

    return True
