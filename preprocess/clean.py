import re
import emoji

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lem_en = WordNetLemmatizer()
stop_en = set(stopwords.words('english'))

stop_ru_custom = {
    "в", "во", "на", "по", "из", "у", "около", "с", "со", "от", "для", "через", "перед",
    "при", "к", "до", "над", "под", "об", "о", "про", "без", "при", "между",
    "и", "а", "но", "да", "же", "ли", "бы", "то", "ни", "не", "ну", "что", "как"
}

whitelist_short_words = {
    "ai", "vr", "os", "cpu", "gpu", "ram", "ssd", "hdd", "pc", "ios", "php", "js",
    "ip", "dns", "usa", "ios", "vr", "xr", "oc", "ms", "cs", "dl", "nlp", "ml"
}


def clean_text(text: str) -> str:
    """
    Очищает текст:\n
    - Убирает ссылки, @имена\n
    - Убирает эмоджи\n
    - Заменяет спецсимволы на пробел\n
    - Приводит к нижнему регистру\n
    - Лемматизация русских и английских слов\n
    - Убирает стоп-слова и короткие слова
    """
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.lower()
    words = text.split()
    clean_words = []

    for word in words:
        if word in whitelist_short_words:
            clean_words.append(word)
            continue
        if len(word) < 3:
            continue
        if word in stop_ru_custom or word in stop_en:
            continue
        if re.match(r'^[a-z]+$', word):
            word = lem_en.lemmatize(word)
        clean_words.append(word)

    return " ".join(clean_words)
