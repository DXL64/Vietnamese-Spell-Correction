import sys
sys.path.append("..")
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer



class TokenAligner():
    def __init__(self, tokenizer: AutoTokenizer, vocab: Vocab = None):
        self.tokenizer = tokenizer
        self.vocab = vocab
    
    """
    params:
        text  ----  str
    """
    def _char_tokenize(self, text):
        characters = list(text)
        tokens = [ token + "@@" if i < len(characters) - 1 and characters[i + 1] != " " else token for i, token in enumerate(characters)]
        tokens = [token for token in tokens if token not in [" @@", " "]]
        print(tokens)
        encoded = self.tokenizer.encode_plus(tokens, return_tensors = "pt")
        token_ids = encoded['input_ids'].squeeze(0)
        attn_mask = encoded['attention_mask'].squeeze(0)
        return tokens, token_ids, attn_mask
    
    def char_tokenize(self, batch_texts):
        doc = dict()
        doc['tokens'] = []
        doc['token_ids'] = []
        doc['attention_mask'] = []
        for text in batch_texts:
            tokens, token_ids, attn_mask = self._char_tokenize(text)
            doc['tokens'].append(tokens)
            doc['token_ids'].append(token_ids)
            doc['attention_mask'].append(attn_mask)
        return doc

    def tokenize_for_transformer_with_tokenization(self, batch_noised_text, batch_label_texts = None):
        docs = self.char_tokenize(batch_noised_text)
        batch_srcs = docs['token_ids']
        batch_attention_masks = docs['attention_mask']

        batch_attention_masks = pad_sequence(batch_attention_masks , 
            batch_first=True, padding_value=0)

        print(batch_srcs)
        batch_srcs = pad_sequence(batch_srcs , 
            batch_first=True, padding_value=self.tokenizer.pad_token_id)
        print(batch_srcs)
            
        if batch_label_texts != None:
            batch_lengths = [len(self.tokenizer.tokenize(text)) for text in batch_label_texts]
            batch_tgts = self.tokenizer.batch_encode_plus(batch_label_texts, max_length = 512, 
                    truncation = True, padding=True, return_tensors="pt")['input_ids']
            return batch_srcs, batch_tgts, batch_lengths, batch_attention_masks
        
        return batch_srcs, batch_attention_masks
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
    dataset = "dxl"
    vocab_path = f'../../../data/{dataset}/{dataset}.vocab.pkl'
    vocab = Vocab("vi")
    vocab.load_vocab_dict(vocab_path)
    token_aligner = TokenAligner(tokenizer, vocab)
    text = "Ô kìa, ai như cô thắm, con bác năm ở xa mới về "
    # text = ["Ô kìa ai như cô thắm con bác năm ở xa mới về "]
    # text = "chúng tôi là những nghiên cứu viên đấy"
    data = token_aligner._char_tokenize(text)[1]
    print(token_aligner.decode(data))
    