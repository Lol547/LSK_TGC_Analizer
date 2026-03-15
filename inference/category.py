import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from preprocess.clean import clean_text
from collections import Counter

CLASS_LABELS = ['IT', 'Блоги', 'Игры', 'Новости', 'Образование', 'Политика', 'Путешествия', 'Финансы']


class CategoryClassifier:
    def __init__(self, model_path: str, device=None, max_len: int = 256):
        """
        model_path: путь к папке с моделью
        device: "cuda" или "cpu"
        max_len: максимальная длина токенов для модели
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len: int = max_len

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, texts: list[str], top_n: int = 3) -> list[str]:
        """
        texts: список текстов сообщений
        top_n: сколько топ категорий вернуть
        """
        texts_clean: list[str] = [clean_text(t) for t in texts if t.strip()]
        if not texts_clean:
            return []

        encodings = self.tokenizer(
            texts_clean,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids: torch.Tensor = encodings['input_ids'].to(self.device)
        attention_mask: torch.Tensor = encodings['attention_mask'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            preds: torch.Tensor = torch.argmax(outputs.logits, dim=1).cpu()

        categories: list[str] = [CLASS_LABELS[p] for p in preds]
        top_categories: list[str] = [cat for cat, _ in Counter(categories).most_common(top_n)]
        return top_categories
