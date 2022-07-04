import torch
from transformers import GPT2LMHeadModel
import argparse
from utils_new import add_special_tokens, sample_seq

def load_model(model_name, path):
	tokenizer = add_special_tokens(model_name)
	model = GPT2LMHeadModel.from_pretrained(model_name)
	model.resize_token_embeddings(len(tokenizer))
	model.load_state_dict(torch.load(path))
	model.eval()
	return model, tokenizer

def GPT2_summarizer(model, text, device, tokenizer, length, temperature, top_k, top_p):
	context = tokenizer(text, return_tensors='np')
	dictionary = {}
	dictionary['article'] = context['input_ids'][0]
	dictionary['sum_idx'] = len(context['input_ids'][0])
	generated_text = sample_seq(model, dictionary['article'], length, device, temperature, top_k, top_p)
	generated_text = generated_text[0,dictionary['sum_idx']:].tolist()
	text = tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
	text = tokenizer.convert_tokens_to_string(text)
	return text

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", default='GroNLP/gpt2-small-italian', type=str, help="Model name to use")
	parser.add_argument("--text", type=str, required=True, help="Text to pass for summarization")
	parser.add_argument("--saved_model", default='model_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.bin', type=str, required=True, help="Saved model path")
	parser.add_argument("--length", default=70, type=int, required=False, help="Number of words to compose the phrase")
	parser.add_argument("--temperature", default=0.2, type=float, required=False, help="Temperature is a degree of randomness of the predictions," 
	"lower temperature provides smoother summaries but increases computation time. A temperature of 1 means random probabilities for the next word, >1 probabilities will take less likely words and use words out of context, <1 more context-related words and more words depending on real training data")
	parser.add_argument("--top_k", default=20, type=int, required=False, help="Top K sampling to sort out by probability and zero-ing out those below the k'th token")
	parser.add_argument("--top_p", default=0.7, type=float, required=False, help="Top P sampling taking one of top N sample and removing those with probability less than P")

	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model, tokenizer = load_model(model_name = args.model_name, path = args.saved_model)
	text = GPT2_summarizer(model = model, text = args.text, tokenizer = tokenizer, device = device, length=args.length, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
	print(text)

if __name__ == '__main__':
	main()

#parameters for my machine
#model_name = 'GroNLP/gpt2-small-italian'
#PATH = r'C:\Users\Mario\Desktop\Summarization_project\weights\config_new\model_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.bin'
#news = "Sarebbe stato molto facile per l'uomo estrarre la freccia dalla carne del malcapitato, eppure questo si rivelò complicato e fatale. La freccia aveva infatti penetrato troppo a fondo nella gamba e aveva provocato una terribile emorragia."
#temperature of 0.15/0.20 and top_k of 20 is the gold standard for summarization tasks