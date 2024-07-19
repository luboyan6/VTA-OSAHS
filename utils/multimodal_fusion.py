from clinical_BERT import bert_enc
from cross_attention import CrossAttention
import torch

def data_fusion(json_dataset):
    q_t = []
    k_v_t = []
    label = []

    for index, val in enumerate(json_dataset):
        vision_embedding = val['image']
        language_embedding = bert_enc(val['text'])

        q_t.append(vision_embedding)
        k_v_t.append(language_embedding)
        label.append(int(val['label']))

    batch_size = len(label)
    seq_len = 1
    d_model = 768

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Concatenate tensors and move to device
    q_t_tensor = torch.cat(q_t, dim=0).view(batch_size, seq_len, d_model).to(device)
    k_v_t_tensor = torch.cat(k_v_t, dim=0).view(batch_size, seq_len, d_model).to(device)
    value_tensor = torch.cat(k_v_t, dim=0).view(batch_size, seq_len, d_model).to(device)

    # Initialize CrossAttention module
    cross_atten = CrossAttention(d_model=d_model, dropout=0.3).to(device)

    # Perform cross-attention
    output = cross_atten(query=q_t_tensor, key=k_v_t_tensor, value=value_tensor).to(device)

    dataset = []
    for index, item in enumerate(output):
        # Concatenate output tensor with label and append to dataset
        item_with_label = torch.cat((item.cpu(), torch.tensor(label[index]).view(1, 1)), 1)
        dataset.append(item_with_label)

    return dataset





