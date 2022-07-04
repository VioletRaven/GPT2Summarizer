import nltk
import string
from transformers import GPT2Tokenizer
nltk.download('stopwords')

class cleaner_d2v:
    @staticmethod
    def add_special_tokens(model):
        """ Returns GPT2 tokenizer after adding separator and padding tokens """
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>'}
        num_add_toks = tokenizer.add_special_tokens(special_tokens)
        return tokenizer
