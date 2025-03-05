from confluent_kafka import Consumer, KafkaException, KafkaError, Producer
import ai_aide_de_camp.config as config
import json


class KafkaClient:
    def __init__(self, broker, group_id, request_topic, response_topic):
        # Kafka 配置
        self.broker = broker
        self.group_id = group_id
        self.request_topic = request_topic
        self.response_topic = response_topic

        # 创建 Producer
        self.producer = Producer({
            'bootstrap.servers': self.broker
        })

        # 创建 Consumer
        self.consumer = Consumer({
            'bootstrap.servers': self.broker,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        })
        self.consumer.subscribe([self.response_topic])

    def send_message(self, message):
        """发送消息到 kafka 模型"""

        def acked(err, msg):
            if err:
                print(f"Failed to deliver message: {err}")
            else:
                print(f"Message produced: {msg.key()}")

        try:
            self.producer.produce(
                self.request_topic,
                value=json.dumps(message).encode('utf-8'),
                callback=acked
            )
            self.producer.poll(1)  # 触发回调
        except Exception as e:
            print(f"Failed to send message: {e}")

    def receive_response(self):
        """监听 kafka的响应"""
        msg = self.consumer.poll(timeout=1)
        if msg is None:
            return None
        if msg.error():
            raise KafkaException(msg.error())

            # 解析消息
        response = json.loads(msg.value().decode('utf-8'))
        print(f"Received response: {response}")
        return response

    def close(self):
        """关闭 Producer 和 Consumer"""
        self.producer.flush()
        self.consumer.close()

if __name__ == "__main__":
    BROKER = "localhost:9092"
    GROUP_ID = "test-group"
    REQUEST_TOPIC = "winchat"
    RESPONSE_TOPIC = "winchat"

    # 初始化 KafkaClient
    client = KafkaClient(BROKER, GROUP_ID, REQUEST_TOPIC, RESPONSE_TOPIC)

    test_message = {
        "user_id": "12345",
        "intent": "get_weather",
        "location": "New York"
    }

    #print("发送消息...")
    #client.send_message(test_message)

    # 监听并打印接收到的消息
    #print("正在监听 Kafka 消息...")
    #response = client.receive_response()

    #print(f"收到响应: {response}")
    # 关闭连接
    client.close()
