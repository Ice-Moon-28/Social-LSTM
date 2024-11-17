import constants

def transformIdIntoSentenceDict():
    words = []
    with open(constants.WORD_EMBEDS) as fp:
        for i, line in enumerate(fp):
            words.append(line.split()[0])
    
    return words

dict = transformIdIntoSentenceDict()

def transformIdIntoSentence(ids):
    nums = filter(lambda x: x >= 0, map(lambda x: int(x, base=10) if x.strip() != '' else -1, ids))
    return " ".join(map(lambda x: dict[x], nums))