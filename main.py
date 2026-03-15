import asyncio
from config import API_ID, API_HASH, SESSION_NAME, NUM_POSTS_FETCH
from config import CATEGORY_MODEL_PATH, MAX_TOKENS, TOP_N_CATEGORIES
from config import SUMMARY_MODEL, OPENROUTER_URL, MAX_POSTS_SUMMARY, OPENROUTER_API_KEY

from parser.telegram import TelegramParser
from inference.category import CategoryClassifier
from inference.summary import SummaryGenerator


class TelegramAnalyzer:
    def __init__(self):
        self.parser = TelegramParser(API_ID, API_HASH, SESSION_NAME)
        self.classifier = CategoryClassifier(CATEGORY_MODEL_PATH, max_len=MAX_TOKENS)
        self.summarizer = SummaryGenerator(
            api_key=OPENROUTER_API_KEY,
            api_url=OPENROUTER_URL,
            model=SUMMARY_MODEL
        )
        self._started = False

    async def start(self):
        if not self._started:
            await self.parser.start()
            self._started = True

    async def stop(self):
        if self._started:
            await self.parser.stop()
            self._started = False

    async def analyze_channel(self, channel_name: str) -> dict:
        """
        channel_name: str с @, например "@MAIuniversity"
        Возвращает словарь:
        {
            "channel": str,
            "categories": list[str],
            "summary": str,
            "num_posts": int
        }
        """
        await self.start()

        posts = await self.parser.fetch_channel_messages(channel_name, n=NUM_POSTS_FETCH)
        if not posts:
            return {
                "channel": channel_name,
                "categories": [],
                "summary": "Нет доступных сообщений для анализа.",
                "num_posts": 0
            }

        categories = self.classifier.predict(posts, top_n=TOP_N_CATEGORIES)
        summary = self.summarizer.summarize(
            posts[:MAX_POSTS_SUMMARY],
            category=categories[0] if categories else "Общее"
        )

        return {
            "channel": channel_name,
            "categories": categories,
            "summary": summary,
        }


def analyze_channel_sync(channel_name: str) -> dict:
    analyzer = TelegramAnalyzer()
    return asyncio.run(analyzer.analyze_channel(channel_name))


if __name__ == "__main__":
    ch = input("Введите название канала Telegram (с @): ")
    result = analyze_channel_sync(ch)
    print("\nРезультат анализа")
    print(f"Канал: {result['channel']}")
    print(f"Топ категории: {result['categories']}")
    print(f"Сводка канала:{result['summary']}")
