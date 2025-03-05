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

from sentence_transformers import SentenceTransformer
import faiss
import os
from ai_aide_de_camp import config
import numpy as np

model = SentenceTransformer(config.SB_BASE_PATH)

user_input = "帮我生成声音"

user_input_embedding = model.encode([user_input])[0]
print(user_input_embedding)

api_data = [
    {"intent": "image_processing", "examples": ["裁剪图片", "调整图片大小"], "api_call": {"function": "crop_image", "parameters": {"width": 100, "height": 100}}},
    {"intent": "text_to_audio", "examples": ["转换文本为语音", "生成语音"], "api_call": {"function": "text_to_speech", "parameters": {"language": "zh", "text": "你好"}}},
    {"intent": "chatting", "examples": []}
]

api_embeddings = [model.encode([example])[0] for item in api_data for example in item["examples"]]
api_embeddings = np.array(api_embeddings).astype('float32')

index = faiss.IndexFlatL2(api_embeddings.shape[1])
index.add(api_embeddings)


k = 1
distances, indices = index.search(np.array([user_input_embedding]).astype('float32'), k)

matching_api = api_data[indices[0][0]]
print(f"匹配的 API：{matching_api['api_call']}")