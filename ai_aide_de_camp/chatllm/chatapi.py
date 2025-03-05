import requests
import json
from pymongo import MongoClient
import ai_aide_de_camp.config as config

class ChatBot:
    def __init__(self, base_url=config.CHAT_BOT_URL, model=config.CHAT_BOT_MODEL):
        self.base_url = base_url
        self.model = model

    def raw_reply(self, messages, temperature=0.7, max_tokens=-1, stream=False):
        """
        发送消息到 LLM Studio 并获取响应
        :param messages: 用户输入
        :param temperature: 控制生成的随机性
        :param max_tokens: 生成的最大 token 数，-1 表示不限制
        :param stream: 是否使用流式输出
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.base_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def reply(self, messages, temperature=0.7, max_tokens=-1, stream=False):
        """
        发送消息到 LLM Studio 并获取响应
        :param messages: 用户输入
        :param temperature: 控制生成的随机性
        :param max_tokens: 生成的最大 token 数，-1 表示不限制
        :param stream: 是否使用流式输出
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.base_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            reply = response.json()
            return reply["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

# 使用示例
if __name__ == "__main__":
    bot = ChatBot()
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [{"type": "text", "text": "knock knock."}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Who's there?"}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "我的上一句话和你的回答是什么"}]
        }
    ]
    response = bot.reply(messages)
    print(response)
