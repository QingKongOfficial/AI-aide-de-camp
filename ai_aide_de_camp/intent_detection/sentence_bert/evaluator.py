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
from sklearn.metrics import accuracy_score, classification_report
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import ai_aide_de_camp.config as config
from ai_aide_de_camp.intent_detection.sentence_bert.dataloader import DataLoaderHelper
from ai_aide_de_camp.intent_detection.sentence_bert.predictor import IntentPredictor


class Evaluator:
    def __init__(self, model_path, intents_path):
        """
        Initialize the Evaluator

        Args:
            model_path (str): Path to the fine-tuned model
            intents_path (str): Path to the intents.json file
        """
        self.model = SentenceTransformer(model_path)  # 加载训练好的模型
        self.intents_path = intents_path
        self.intent_dataset = self._load_data()  # 加载测试数据
        self.test_sentences = self.intent_dataset['sentences']
        self.true_labels = self.intent_dataset['labels']
        self.train_embeddings = None
        self._prepare_train_embeddings()

    def _load_data(self):
        """Load and prepare test data from intents.json"""
        with open(self.intents_path, "r", encoding="utf-8") as f:
            intents = json.load(f)

        sentences = []
        true_labels = []

        # Prepare sentences and true labels for evaluation
        for intent in intents:
            examples = intent["examples"]
            for example in examples:
                sentences.append(example)
                true_labels.append(intent["intent"])

        return {"sentences": sentences, "labels": true_labels}

    def _prepare_train_embeddings(self):
        """Precompute sentence embeddings for training data (from the same intents)"""
        sentences = self.intent_dataset['sentences']
        self.train_embeddings = self.model.encode(sentences, convert_to_tensor=True)

    def _predict(self, sentence):
        """Predict the intent of a single sentence using cosine similarity."""
        input_embedding = self.model.encode([sentence], convert_to_tensor=True)

        # Calculate cosine similarity between the input sentence and all training sentences
        cos_scores = util.cos_sim(input_embedding, self.train_embeddings)

        # Get the index of the highest similarity score
        top_idx = int(np.argmax(cos_scores.cpu().numpy()))

        # Return the corresponding intent from the intent mapping
        return self.intent_dataset['labels'][top_idx]

    def evaluate(self):
        """Evaluate the fine-tuned model on the test set."""
        predicted_labels = []
        # Predict intents for all test sentences
        for sentence in self.test_sentences:
            predicted_label = self._predict(sentence)
            predicted_labels.append(predicted_label)

        # Calculate evaluation metrics
        accuracy = accuracy_score(self.true_labels, predicted_labels)
        report = classification_report(self.true_labels, predicted_labels)

        # Print evaluation results
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Classification Report:\n{report}")




if __name__ == "__main__":
    # 初始化 Evaluator
    model_path = config.SB_FINETUNED_PATH
    intents_path = config.INTENT_DATA
    evaluator = Evaluator(model_path, intents_path)

    # 评估模型
    evaluator.evaluate()