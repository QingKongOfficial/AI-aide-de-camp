import json
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util
from itertools import combinations
import numpy as np
from ai_aide_de_camp.intent_detection.sentence_bert.dataloader import IntentDataset
import ai_aide_de_camp.config as config


class IntentPredictor:
    def __init__(self, model_path, intents_path):
        """
        Args:
            model_path (str): Path to the trained SentenceBERT model.
            intents_path (str): Path to the intents' data.
        """
        self.model = SentenceTransformer(model_path)
        self.intents_path = intents_path
        self.intent_dataset = IntentDataset(intents_path)
        self.train_embeddings = None
        self.prepare_train_embeddings()

    def prepare_train_embeddings(self):
        """Precompute sentence embeddings for training data."""
        sentences = self.intent_dataset.sentences
        self.train_embeddings = self.model.encode(sentences, convert_to_tensor=True)

    def predict(self, sentence):
        """Predict the intent of a single sentence."""
        # Encode the input sentence
        input_embedding = self.model.encode([sentence], convert_to_tensor=True)

        # Calculate cosine similarities between the input sentence and all training sentences
        cos_scores = util.cos_sim(input_embedding, self.train_embeddings)

        # Get the index of the highest similarity score
        top_idx = int(np.argmax(cos_scores.cpu().numpy()))

        # Get the corresponding intent from the intent mapping
        predicted_intent = self.intent_dataset.intent_mapping[top_idx]

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
    # 初始化 IntentPredictor
    model_path = config.SB_FINETUNED_PATH
    intents_path = config.INTENT_DATA
    predictor = IntentPredictor(model_path, intents_path)

    # 单个句子预测
    sentence = "你好啊，今天怎么样"
    predicted_intent = predictor.predict(sentence)
    print(f"Predicted intent: {predicted_intent}")

    # 批量预测
    sentences = ["怎么把我的文字转语音", "你帮我看看怎么裁剪这个图片"]
    predicted_intents = predictor.predict_batch(sentences)
    print(f"Predicted intents: {predicted_intents}")