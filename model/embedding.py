import enum

class EmbeddingType(enum.Enum):
    RANDOM_INITIALIZE = 1
    PRETRAINED = 2
    ONE_HOT = 3
    ONE_HOT_WITH_BIGGER_SIZE = 4