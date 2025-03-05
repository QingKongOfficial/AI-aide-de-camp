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

SB_FINETUNED_PATH = "A:/WORK/Git/AI-aide-de-camp/ai_aide_de_camp/intent_detection/sentence_bert/models/sentence_bert_finetuned"  #finetuned model path
SB_BASE_PATH = "A:/WORK/Git/AI-aide-de-camp/ai_aide_de_camp/intent_detection/sentence_bert/models/paraphrase-multilingual-mpnet-base-v2"
INTENT_DATA = "A:/WORK/Git/AI-aide-de-camp/ai_aide_de_camp/intent_detection/sentence_bert/train_data/intents.json"


BROKER = "localhost:9092"
GROUP_IN_ID = "ai-aide-de-camp-input"
GROUP_OUT_ID = "ai-aide-de-camp-output"
REQUEST_TOPIC = "ai-aide-de-camp-To-winchat"
RESPONSE_TOPIC = "winchat-To-ai-aide-de-camp"

CHAT_BOT_URL = "http://localhost:1234/v1/chat/completions"
CHAT_BOT_MODEL = "publisher/repository/causallm_7b.q6_k.gguf"

CHAT_DB_URL = r"mongodb://localhost:27017/"
CHAT_DB_NAME = r"chat_db"  #对话记录
WINCHAT_LOG = r"winchat_log"  #日志记录