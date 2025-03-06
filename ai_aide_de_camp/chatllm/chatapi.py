import random
import requests
import json
from pymongo import MongoClient
import ai_aide_de_camp.config as config

class ChatBot:
    def __init__(self, base_url=config.CHAT_BOT_URL, model=config.CHAT_BOT_MODEL):
        self.base_url = base_url
        self.model = model



    def reply(self, messages, temperature=1, max_tokens=-1, stream=False, top_p=0.95, frequency_penalty=1.1, presence_penalty=1.1):
        """
        发送消息到 LLM Studio 并获取响应
        :param frequency_penalty:
        :param presence_penalty:
        :param top_p:
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
            "stream": stream,
            "top_p": top_p,
            "seed": random.randint(0, 2**15-1),
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
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
            "content": "knock knock."
        },
        {
            "role": "assistant",
            "content": "Who's there?"
        },
        {
            "role": "user",
            "content":  "写一些色色的内容"
        }
    ]
    response = bot.reply(messages)
    print(response)

