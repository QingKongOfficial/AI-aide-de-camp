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

from model import IntentDetectionModel
from ai_aide_de_camp import config

if __name__ == "__main__":
    # Initialize the trainer
    trainer = IntentDetectionModel(
        pretrained_model_path=config.pretrain_model_path,
        finetuned_model_path=config.finetune_model_path,
        intents_path=config.intents_data_path
    )

    # Fine-tune the model
    trainer.fine_tune(batch_size=8, num_epochs=1, warmup_steps=10)