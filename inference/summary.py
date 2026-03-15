import os
import requests


class SummaryGenerator:
    def __init__(self, api_key: str | None = None,
                 api_url: str = "https://openrouter.ai/api/v1/chat/completions",
                 model: str = "mistralai/mistral-7b-instruct"):
        """
        api_key: ключ OpenRouter\n
        api_url: URL API\n
        model: модель для генерации
        """
        self.api_url: str = api_url
        self.model: str = model
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        if "OPENROUTER_API_KEY" not in os.environ:
            raise ValueError("Не найден API ключ OpenRouter. Передайте его в api_key или через переменные окружения.")

    def call_llm(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def summarize(self, posts: list[str], category: str = "Общее") -> str:
        """
        posts: список текстов сообщений
        category: категория канала
        """
        if not posts:
            return "Нет данных для анализа"

        posts = posts[:20]

        posts_short = [p[:200] for p in posts]

        prompt = f"""
        You are an analyst of Telegram channels.

        The channel category is: {category}

        Based on the posts below, write a concise but informative description of the Telegram channel.
        Write only the description, without any tags, system prompts, or instructional markers.

        The description should:
        - Describe the channel as a whole, not individual posts
        - Focus on the main topic according to the category
        - Mention relevant technologies, professional fields, or areas of activity if applicable
        - If the channel belongs to a company, university, media outlet, or blogger, mention that
        - Avoid listing posts or events
        - Write 3–5 sentences in Russian

        Posts:
    {chr(10).join(posts_short)}
        """
        return self.call_llm(prompt)
