
# Ai aide-de-camp
![version](https://img.shields.io/badge/version-1.0-blue)
![license](https://img.shields.io/badge/license-AGPL--3.0-red)

## 📌 项目简介

该项目使用Kafka接入聊天软件，使用意图识别模型初步判断用户意图。并通过Milvus实现RAG来增强意图识别的准确性和拓展性。
再将用户的输入传递到对应的回应模型（目前使用本地部署llm），同时使用MongoDB实现llm的对话记忆和日志记录。最终通过Kafka返回响应内容。

## 🛠️ 主要特性
- 高吞吐的消息处理 (Kafka)
- 高效的数据存储 (MongoDB)
-  向量搜索和 AI 处理 (Milvus)
- 多线程实时数据处理能力
- 可扩展的架构

该项目可同时适用于AI聊天，推荐系统，大数据分析的生产场景。

## License
This project is licensed under [AGPL-3.0 license].  
The following components use Apache-2.0 License:
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)  
  Copyright 2024 UKPLab. Licensed under Apache-2.0.