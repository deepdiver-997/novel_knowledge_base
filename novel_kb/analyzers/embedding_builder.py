from novel_kb.llm.provider import LLMProvider


class EmbeddingBuilder:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def build(self, text: str):
        """同步方法 - 不推荐使用"""
        import asyncio
        return asyncio.run(self.build_async(text))

    async def build_async(self, text: str):
        """异步方法 - 推荐使用"""
        return await self.provider.generate_embedding(text)
