from transformers import BigBirdModel, BigBirdTokenizer
import torch

# Load BigBird
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
model = BigBirdModel.from_pretrained('google/bigbird-roberta-base')
model.eval()

# Create a random sparse attention mask for demonstration
def make_attention_mask(input_ids, global_attention_indices):
    # global_attention_indices: list of token positions to attend globally
    mask = torch.zeros_like(input_ids)
    mask[:, global_attention_indices] = 1
    return mask


def process_with_bigbird(text: str, block_size: int = 64, num_global: int = 1):
    tokens = tokenizer(text, return_tensors='pt', max_length=4096, truncation=True)
    input_ids = tokens.input_ids
    # Set the first `num_global` tokens (e.g., CLS) to global attention
    global_indices = list(range(num_global))
    global_mask = make_attention_mask(input_ids, global_indices)

    # Forward pass with sparse attention
    outputs = model(
        **tokens,
        attention_mask=None,
        global_attention_mask=global_mask,
        block_size=block_size
    )
    last_hidden = outputs.last_hidden_state  # [B, T, D]
    return last_hidden

# Example usage
text = "[Your very long document here...]"
hidden_states = process_with_bigbird(text, block_size=128, num_global=2)
print(hidden_states.shape)




# Pseudo-code: HuggingFace does not provide a built-in compressive model,
# so this illustrates the loop:     Compressive Transformer (High-Level Sketch)

# def process_with_compressive(model, tokenizer, text, segment_len=512, comp_ratio=0.5):
#     input_ids = tokenizer(text, return_tensors='pt').input_ids
#     short_mem = None
#     comp_mem = None
#     outputs_all = []

#     for idx in range(0, input_ids.size(1), segment_len):
#         segment = input_ids[:, idx: idx+segment_len]
#         outputs = model(segment, mems=short_mem)
#         # Update short-term memory
#         new_short = outputs.mems  # List of K/V states
#         # Compress oldest part of short_mem into comp_mem
#         if short_mem is not None:
#             oldest = short_mem[:int(len(short_mem)*comp_ratio)]
#             comp_mem = compress(oldest, into=comp_mem)
#         short_mem = new_short
#         outputs_all.append(outputs.last_hidden_state)

#     # Merge all hidden states as final
#     final = torch.cat(outputs_all, dim=1)
#     return final
