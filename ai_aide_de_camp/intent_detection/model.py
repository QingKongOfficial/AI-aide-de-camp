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
# -*- coding: utf-8 -*-
import os
import json
from tqdm.auto import tqdm
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, util, SentencesDataset
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple
import json
from itertools import combinations
from torch.utils.data import DataLoader
import torch
import faiss
from sentence_transformers import SentenceTransformer, InputExample, losses
from ai_aide_de_camp import config


class IntentDetectionModel:
    def __init__(self, pretrained_model_path, finetuned_model_path, intents_path):
        """
        Initialize the SentenceBERT trainer

        Args:
            pretrained_model_path (str): Path to the pretrained model
            finetuned_model_path (str): Path to save the finetuned model
            intents_path (str): Path to the intents.json file
        """
        self.pretrained_model_path = pretrained_model_path
        self.finetuned_model_path = finetuned_model_path
        self.intents_path = intents_path
        self.model = None
        self.index = None
        self.sentences = []
        self.intent_mapping = {}

    def load_training_data(self):
        """Load and prepare training data from intents.json"""
        with open(self.intents_path, "r", encoding="utf-8") as f:
            intents = json.load(f)

        train_examples = []
        # Keep track of sentences and their intents for FAISS
        sentence_id = 0

        for intent in intents:
            examples = intent["examples"]
            # Store sentences and their intent mappings
            for example in examples:
                self.sentences.append(example)
                self.intent_mapping[sentence_id] = intent["intent"]
                sentence_id += 1

            # Generate all pairwise combinations for training
            for s1, s2 in combinations(examples, 2):
                train_examples.append(InputExample(texts=[s1, s2]))

        return train_examples

    def fine_tune(self, batch_size=8, num_epochs=1, warmup_steps=10):
        """
        Fine-tune the SentenceBERT model

        Args:
            batch_size (int): Training batch size
            num_epochs (int): Number of training epochs
            warmup_steps (int): Number of warmup steps
        """
        # Load model
        self.model = SentenceTransformer(self.pretrained_model_path)

        # Prepare training data
        train_examples = self.load_training_data()
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        # Define loss function
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps
        )

        # Save the fine-tuned model
        self.model.save(self.finetuned_model_path)
        print(f"Fine-tuned model saved to {self.finetuned_model_path}")


if __name__ == "__main__":
    pass
