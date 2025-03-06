"""Copyright (C) [2025-present] [Qing Kong]

This file is part of AI-aide-de-camp.

AI-aide-de-camp is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

AI-aide-de-camp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with AI-aide-de-camp.
If not, see <https://www.gnu.org/licenses/>.
"""
from itertools import count

# -*- coding: utf-8 -*-
import ai_aide_de_camp.config as config
from ai_aide_de_camp.chatllm.chatapi import ChatBot
from kafka.client import KafkaClient
from confluent_kafka import Consumer, KafkaException, KafkaError, Producer
import threading
from queue import Queue, Empty
from ai_aide_de_camp.chatllm.extractor import Filter
from pymongo import MongoClient
from datetime import datetime
import time

queue_recved_message = Queue()
queue_kafka = Queue()
queue_intent = Queue()
stop_event = threading.Event()



def request_handler():
    """线程：处理 Kafka 中来自winchat的响应消息"""
    kafka_client = KafkaClient(
        broker=config.BROKER,
        group_id=config.GROUP_IN_ID,
        request_topic=config.REQUEST_TOPIC,
        response_topic=config.RESPONSE_TOPIC
    )

    mongo_client = MongoClient(config.CHAT_DB_URL)
    db = mongo_client[config.WINCHAT_LOG]
    collection = db["winchat_log"]

    try:
        while not stop_event.is_set():
            try:
                response = kafka_client.receive_response()  # 阻塞等待响应消息
                if response is None:
                    continue
                filter = Filter(response)
                if filter.process("classify") in ["unread", "friend::gh", "friend::person", "friend::chatroom"]:
                    continue
                response["timestamp"] = datetime.now()
                inserted_id = collection.insert_one(response).inserted_id
                print(f"消息已存至数据库: {response}")

                if filter.process("classify") == "msg::single":
                    wx_id = filter.process("extract", "from_wxid")
                    send_or_recv = filter.process("extract", "send_or_recv")
                    data_type = filter.process("extract", "data_type")

                    if wx_id == "filehelper" and send_or_recv == "1+[Phone]" and data_type == "1":
                        output = [{"role": "system", "content": ""}]
                        recent_messages = collection.find({
                            "data.data_type": "1",
                            "data.from_wxid": wx_id
                        }).sort("timestamp", -1).limit(5)

                        for history in reversed(list(recent_messages)):
                            print(history)
                            if history["data"]["send_or_recv"] == "1+[Demo]":
                                output.append(
                                    {"role": "assistant",
                                     "content": history["data"]["msg"]})

                            else:
                                output.append(
                                    {"role": "user",
                                     "content": history["data"]["msg"]})
                        query = {"from_wxid": wx_id, "question": output}
                        queue_intent.put(query)

                    if wx_id != "filehelper" and send_or_recv == "0+[收到]" and data_type == "1":
                        output = [{"role": "system", "content": "清清是一个程序员，你是清清的助理，你代替他对聊天对话代为回复"}]
                        recent_messages = collection.find({
                            "data.data_type": "1",
                            "data.from_wxid": wx_id
                        }).sort("timestamp", -1).limit(5)

                        for history in reversed(list(recent_messages)):
                            print(history)
                            if history["data"]["send_or_recv"] == "1+[Demo]":
                                output.append(
                                    {"role": "assistant",
                                     "content": history["data"]["msg"]})
                            else:
                                output.append(
                                    {"role": "user",
                                     "content": history["data"]["msg"]})
                        query = {"from_wxid": wx_id, "question": output}
                        queue_intent.put(query)


            except KafkaException as e:
                print(f"Kafka error: {e}")
                time.sleep(3)

    except Exception as e:
        print(f"response_handler响应处理线程异常: {e}")
    finally:
        kafka_client.close()
        mongo_client.close()
        print("response_handler响应处理线程已关闭")

def response_producer():
    """线程：从response_handler线程获得消息并分发给模型并发回给kafka"""
    bot = ChatBot()
    kafka_client = KafkaClient(
        broker=config.BROKER,
        group_id=config.GROUP_OUT_ID,
        request_topic=config.REQUEST_TOPIC,
        response_topic=config.RESPONSE_TOPIC
    )
    try:
        while not stop_event.is_set():
            try:
                query = queue_intent.get(timeout=1)
                if query is None:
                    continue

                print("query receive:", query["question"], query["from_wxid"])
                output = {"from_wxid": query["from_wxid"], "reply": ''}
                output["reply"] = bot.reply(query["question"])
                kafka_client.send_message(output)
                print("send to kafka:", output)
            except Empty:
                continue
    except Exception as e:
        print(f"message_dealer消息处理线程异常: {e}")
    finally:
        kafka_client.close()
        print("message_dealer消息处理线程已关闭")

if __name__ == "__main__":

    response_thread = threading.Thread(target=request_handler)
    intent_thread = threading.Thread(target=response_producer)

    response_thread.start()
    intent_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("退出中...")
        stop_event.set()  # 通知所有线程退出
        queue_recved_message.put(None)  # 终止消息线程
        queue_kafka.put(None)  # 终止 Kafka 线程
        queue_intent.put(None)

        response_thread.join()
        intent_thread.join()
        print("所有线程已安全退出")