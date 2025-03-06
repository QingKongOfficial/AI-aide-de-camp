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
import ai_aide_de_camp.config as config

import json
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer


class IntentToMilvus:
    def __init__(self, host='localhost', port='19530', collection_name='intent_collection'):
        """
        初始化IntentToMilvus类

        参数:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 要创建的集合名称
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(config.SB_FINETUNED_PATH)  # 多语言模型，支持中文
        self.vector_dim = self.encoder.get_sentence_embedding_dimension()

        # 连接到Milvus服务器
        self.connect_to_milvus()

    def connect_to_milvus(self):
        """连接到Milvus服务器"""
        try:
            connections.connect(host=self.host, port=self.port)
            print(f"成功连接到Milvus服务器: {self.host}:{self.port}")
        except Exception as e:
            print(f"连接Milvus服务器失败: {e}")
            raise

    def create_collection(self):
        """创建Milvus集合，如果已经存在则先删除"""
        # 如果集合已存在，则删除
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"已删除原有集合: {self.collection_name}")

        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="example", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]

        # 创建集合
        schema = CollectionSchema(fields=fields, description="Intent and examples collection")
        collection = Collection(name=self.collection_name, schema=schema)

        # 创建IVF_FLAT索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embeddings", index_params=index_params)
        print(f"已创建集合: {self.collection_name}，并为embeddings字段创建索引")
        return collection

    def encode_text(self, text):
        """将文本编码为向量"""
        return self.encoder.encode(text)

    def load_intents_from_json(self, json_file_path):
        """从JSON文件加载意图数据"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取JSON文件失败: {e}")
            raise

    def insert_intents(self, json_file_path):
        """将意图数据插入到Milvus集合中"""
        # 加载数据
        intents_data = self.load_intents_from_json(json_file_path)

        # 创建集合
        collection = self.create_collection()

        # 准备插入数据
        intent_list = []
        example_list = []
        embedding_list = []

        # 处理每个意图及其示例
        for intent_obj in intents_data:
            intent_name = intent_obj['intent']
            examples = intent_obj['examples']

            for example in examples:
                intent_list.append(intent_name)
                example_list.append(example)
                # 对示例文本进行编码
                embedding = self.encode_text(example)
                embedding_list.append(embedding)

        # 插入数据
        data = [
            intent_list,
            example_list,
            embedding_list
        ]

        # 插入Milvus集合
        collection.insert(data)
        collection.flush()  # 确保数据写入
        print(f"已成功插入 {len(intent_list)} 条意图示例到Milvus集合中")

        # 加载集合以便搜索
        collection.load()
        count = collection.num_entities
        print(f"集合中共有 {count} 条记录")

        return collection

    def search_similar_intent(self, query_text, top_k=3):
        """搜索相似意图

        参数:
            query_text: 查询文本
            top_k: 返回结果数量

        返回:
            匹配的意图及相似度
        """
        # 编码查询文本
        query_embedding = self.encode_text(query_text).tolist()

        # 加载集合
        collection = Collection(self.collection_name)
        collection.load()

        # 执行向量搜索
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embeddings",
            param=search_params,
            limit=top_k,
            output_fields=["intent", "example"]
        )

        # 处理搜索结果
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append({
                    "intent": hit.entity.get("intent"),
                    "example": hit.entity.get("example"),
                    "similarity": hit.distance
                })

        return search_results

    def close(self):
        """关闭Milvus连接"""
        connections.disconnect(alias='default')
        print("已关闭与Milvus的连接")


# 使用示例
if __name__ == "__main__":
    # 初始化IntentToMilvus类
    intent_milvus = IntentToMilvus(host='localhost', port='19530')

    # 插入数据
    intent_milvus.insert_intents(config.INTENT_DATA)

    # 执行测试查询
    test_query = "你能告诉我北京今天的温度吗"
    results = intent_milvus.search_similar_intent(test_query, top_k=2)

    print("\n测试查询结果:")
    for idx, result in enumerate(results):
        print(f"{idx + 1}. 意图: {result['intent']}, 示例: {result['example']}, 相似度: {result['similarity']:.4f}")

    # 关闭连接
    intent_milvus.close()