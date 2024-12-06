import enum

class EmbeddingType(enum.Enum):
    RANDOM_INITIALIZE = 1
    PRETRAINED = 2
    ONE_HOT = 3