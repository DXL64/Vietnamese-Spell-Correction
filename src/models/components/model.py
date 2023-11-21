from src.models.components.transformer import TransformerWithTR
from src.models.components.collator import *
from transformers import AutoTokenizer
import transformers
from src.models.components.tokenizer import TokenAligner
from src.data.components.vocab import Vocab
import pyrootutils
import rootutils
import os

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

root = rootutils.find_root(search_from=__file__, indicator=".project-root")

class ModelWrapper:

    def __init__(self, model_name, vocab_dataset: str, device):
        rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

        root = rootutils.find_root(search_from=__file__, indicator=".project-root")
        
        self.model_name = model_name

        vocab_path = str(root / "data" / vocab_dataset / f"{vocab_dataset}.vocab.pkl")
        self.vocab = Vocab("vi")
        self.vocab.load_vocab_dict(vocab_path)
        if model_name == "tfmwtr":
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word-base")
            self.tokenAligner = TokenAligner(self.tokenizer, self.vocab)
            self.bart = transformers.MBartForConditionalGeneration.from_pretrained("vinai/bartpho-word-base").to(device)
            self.model = TransformerWithTR(self.bart, self.tokenizer.pad_token_id, device)
            self.collator = DataCollatorForCharacterTransformer(self.tokenAligner)
            # self.model.resize_to[ken_embeddings(self.tokenAligner)
        else:
            raise(Exception(f"Model {model_name} isn't implemented!"))
        