from telethon import TelegramClient
from telethon.tl.custom.message import Message


class TelegramParser:
    def __init__(self, api_id: int, api_hash: str, session_name: str = "sessions/my_telegram"):
        self.client: TelegramClient = TelegramClient(session_name, api_id, api_hash)
        self.started: bool = False

    async def start(self) -> None:
        if not self.started:
            await self.client.start()
            self.started = True

    async def stop(self) -> None:
        if self.started:
            await self.client.disconnect()
            self.started = False

    async def fetch_channel_messages(self, channel_name: str, n: int = 50) -> list[str]:
        """
        channel_name: название канала через @ \n
        Забирает последние n сообщений с канала.\n
        Возвращает список текстов сообщений.
        """
        if not self.started:
            await self.start()
        posts: list[Message] = await self.client.get_messages(channel_name, limit=n)
        return [p.message for p in posts if p.message is not None]
