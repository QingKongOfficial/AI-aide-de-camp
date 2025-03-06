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
from sentence_transformers import SentenceTransformer, InputExample, losses, util, SentencesDataset
import ai_aide_de_camp.config as config
from ai_aide_de_camp.intent_detection.sentence_bert.dataloader import DataLoaderHelper

class SentenceBertTrainer:
    def __init__(self, base_model_path, finetuned_model_path, intents_path):
        """
        Initialize the SentenceBERT trainer

        Args:
            base_model_path (str): Path to the pretrained model
            finetuned_model_path (str): Path to save the finetuned model
            intents_path (str): Path to the intents.json file
        """
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.intents_path = intents_path
        self.model = None
        self.index = None
        self.sentences = []
        self.intent_mapping = {}

    def fine_tune(self, batch_size=8, num_epochs=1, warmup_steps=10):
        """
        Fine-tune the SentenceBERT model

        Args:
            batch_size (int): Training batch size
            num_epochs (int): Number of training epochs
            warmup_steps (int): Number of warmup steps
        """
        # Load model
        self.model = SentenceTransformer(self.base_model_path)

        # Instantiate DataLoaderHelper to get the dataloader
        data_loader_helper = DataLoaderHelper(self.intents_path, batch_size)
        train_dataloader = data_loader_helper.get_dataloader()

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

    # Initialize the trainer
    trainer = SentenceBertTrainer(
        base_model_path=config.SB_BASE_PATH,
        finetuned_model_path=config.SB_FINETUNED_PATH,
        intents_path=config.INTENT_DATA
    )

    # Fine-tune the model
    trainer.fine_tune(batch_size=8, num_epochs=1, warmup_steps=10)