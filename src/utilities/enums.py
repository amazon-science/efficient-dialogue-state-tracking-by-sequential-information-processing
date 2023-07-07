from pytorch_lightning.utilities.enums import LightningEnum


class RunningStage(LightningEnum):
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"


class OutputKeys(LightningEnum):
    PRED: str = "y_hat"
    TARGET: str = "y"
    LOSS: str = "loss"
    LOGS: str = "logs"
    LOGITS: str = "logits"
    BATCH_SIZE: str = "batch_size"


class InputKeys(LightningEnum):
    TARGET: str = "labels"
    INPUT_IDS: str = "input_ids"
    ATT_MASK: str = "attention_mask"
    TOKEN_TYPE_IDS: str = "token_type_ids"
