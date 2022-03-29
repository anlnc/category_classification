from libs import *
from config import *

def preprocess_sentence(sent: str) -> str:
    sent = sent.lower().strip()
    sent = re.sub(r"\s{2,}", " ", sent)  # Remove redundant spaces
    sent = " ".join(word_segmenter.tokenize(sent)[0])
    return sent

def predict(item_name: str) -> Dict[str, str]:
    preprocessed_item_name = preprocess_sentence(item_name)
    prediction = category_classifier(preprocessed_item_name)[0]
    prediction["itemName"] = item_name
    return prediction


if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained("./checkpoints/category_classification_model")
    tokenizer = AutoTokenizer.from_pretrained("./checkpoints/tokenizer")
    word_segmenter = VnCoreNLP(
            "./vncorenlp/VnCoreNLP-1.1.1.jar",
            annotators="wseg",
            max_heap_size="-Xmx500m",
        )
    category_classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer
    )
    
    item_names = [
        "Bàn Chải Đánh Răng P/S Lông Tơ Mềm Mại", 
        "Bàn Chải Đánh Răng P/S Lông Tơ Mềm Mại",
        "Dung dịch chuẩn độ karl ficher (Titrant 5) 188010, 1lít/chai (Merck) PO: 20/01373 "
    ]
    predictions = list(map(lambda item_name: predict(item_name), item_names))