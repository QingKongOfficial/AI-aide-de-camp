from torch.utils.data import Dataset, DataLoader
import json
from sentence_transformers import InputExample
from itertools import combinations


class IntentDataset(Dataset):
    def __init__(self, intents_path):
        """
        Args:
            intents_path (str): Path to the intents.json file
        """
        self.intents_path = intents_path
        self.sentences = []
        self.intent_mapping = {}
        self.train_examples = []

        self.load_data()

    def load_data(self):
        """Load and prepare training data from intents.json"""
        with open(self.intents_path, "r", encoding="utf-8") as f:
            intents = json.load(f)

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
                self.train_examples.append(InputExample(texts=[s1, s2]))

    def __len__(self):
        """Return the number of examples in the dataset"""
        return len(self.train_examples)

    def __getitem__(self, idx):
        """Return an example by index"""
        return self.train_examples[idx]


class DataLoaderHelper:
    def __init__(self, intents_path, batch_size=8, shuffle=True):
        """
        Args:
            intents_path (str): Path to the intents.json file
            batch_size (int): Batch size for the DataLoader
            shuffle (bool): Whether to shuffle the data
        """
        self.intents_path = intents_path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self):
        """Create and return a DataLoader instance"""
        # Instantiate the dataset
        dataset = IntentDataset(self.intents_path)

        # Create DataLoader instance
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return dataloader