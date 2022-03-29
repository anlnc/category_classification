from libs import *

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

MODEL_CHECKPOINT = "vinai/phobert-base"
SEQ_CLS_MODEL_CHECKPOINT = "./checkpoints/original/phoBERT_base"
TOKENIZER_CHECKPOINT = "./checkpoints/original/tokenizer"

DATASET_FOR_CATEGORY_CLS = (
    "./dataset/28-3-2022/dataset_for_category_classification.json"
)
TRAIN_SET_FOR_CATEGORY_CLS = "./dataset/28-3-2022/train_dataset_for_category_cls.json"
VAL_SET_FOR_CATEGORY_CLS = "./dataset/28-3-2022/val_dataset_for_category_cls.json"


VNCORENLP_JAR_PATH = "./vncorenlp/VnCoreNLP-1.1.1.jar"
MAX_LEN = 125
BATCH_SIZE = 32
USE_CLASS_WEIGHT = True
