import torch
from transformers import AutoTokenizer, AutoModel,logging
logging.set_verbosity_error()

def bert_enc(text):

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to('cuda')

    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to('cuda')

    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]

    sentence_embedding = last_hidden_states[0][0]

    normalized_embeddings = torch.nn.functional.normalize(sentence_embedding, p=2, dim=0)

    return normalized_embeddings




