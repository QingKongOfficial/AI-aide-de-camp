import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from ai_aide_de_camp.intent_detection.sentence_bert.dataloader import IntentDataset
import ai_aide_de_camp.config as config
from pymilvus import connections, Collection, utility

class IntentPredictor:
    def __init__(self, model_path, intents_path, milvus_cfg):
        """
        Args:
            model_path (str): Path to the trained SentenceBERT model.
            intents_path (str): Path to the intents' data.
            milvus_cfg (dict): Milvus配置信息，包含host、port、collection_name等
        """
        # 原有初始化逻辑
        self.model = SentenceTransformer(model_path)
        self.intents_path = intents_path
        self.intent_dataset = IntentDataset(intents_path)
        self.train_embeddings = None
        self.prepare_train_embeddings()

        # 新增 Milvus 连接
        connections.connect(
            host=milvus_cfg["host"],
            port=milvus_cfg["port"]
        )
        self.milvus_collection = Collection(milvus_cfg["collection_name"])
        self.milvus_collection.load()

        # 配置阈值（根据实际场景调整）
        self.confidence_threshold = 0.7

    def rag_search(self, query_embedding, top_k=3):
        """使用 Milvus 进行 RAG 检索"""
        # 将 Tensor 转换为 numpy array
        query_embedding = query_embedding.cpu().numpy().squeeze()

        # 定义搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16}
        }

        # 执行搜索
        results = self.milvus_collection.search(
            data=[query_embedding],
            anns_field="embedding",  # 假设 Milvus 集合中存储向量的字段名
            param=search_params,
            limit=top_k,
            output_fields=["intent", "text"]  # 假设 Milvus 集合中存储的元数据字段
        )

        return results[0]  # 返回第一个查询的结果（因为只查询了一个句子）

    def prepare_train_embeddings(self):
        """Precompute sentence embeddings for training data."""
        sentences = self.intent_dataset.sentences
        self.train_embeddings = self.model.encode(sentences, convert_to_tensor=True)

    def predict(self, sentence):
        """增强版预测（含 RAG 流程）"""
        # 原始编码和相似度计算
        input_embedding = self.model.encode([sentence], convert_to_tensor=True)
        cos_scores = util.cos_sim(input_embedding, self.train_embeddings)
        top_score = torch.max(cos_scores).item()
        top_idx = int(np.argmax(cos_scores.cpu().numpy()))
        original_intent = self.intent_dataset.intent_mapping[top_idx]

        # 如果置信度低于阈值，触发 RAG
        if top_score < self.confidence_threshold:
            rag_results = self.rag_search(input_embedding)

            # 从 RAG 结果中提取意图（示例：选择出现频率最高的意图）
            rag_intents = [hit.entity.get("intent") for hit in rag_results]
            if rag_intents:
                predicted_intent = max(set(rag_intents), key=rag_intents.count)
            else:
                predicted_intent = "unknown"  # 处理无结果情况
        else:
            predicted_intent = original_intent

        return predicted_intent

    def predict_batch(self, sentences):
        """Predict intents for a batch of sentences."""
        input_embeddings = self.model.encode(sentences, convert_to_tensor=True)

        # Calculate cosine similarities between the input sentences and all training sentences
        cos_scores = util.cos_sim(input_embeddings, self.train_embeddings)

        # Get the top index for each input sentence
        top_idxs = np.argmax(cos_scores.cpu().numpy(), axis=1)

        # Get the corresponding intents from the intent mapping
        predicted_intents = [self.intent_dataset.intent_mapping[idx] for idx in top_idxs]

        return predicted_intents


if __name__ == "__main__":
    milvus_cfg = {
        "host": "localhost",
        "port": "19530",
        "collection_name": "intent_rag_db"
    }

    predictor = IntentPredictor(
        model_path=config.SB_FINETUNED_PATH,
        intents_path=config.INTENT_DATA,
        milvus_cfg=milvus_cfg
    )

    # 测试预测
    test_sentence = "如何申请信用卡的延期还款？"
    print(predictor.predict(test_sentence))