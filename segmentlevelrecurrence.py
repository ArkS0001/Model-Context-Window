from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
import torch

# 1. Load model and tokenizer
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
model.eval()

def process_long_input(text: str, segment_len: int = 512):
    """
    Processes `text` in chunks of `segment_len`, maintaining memory (K/V cache).
    Returns concatenated logits tensor.
    """
    # Tokenize full document
    tokens = tokenizer.encode(text, return_tensors='pt')  # shape: [1, N]
    mems = None
    all_logits = []

    # Process each segment sequentially
    for start_idx in range(0, tokens.size(1), segment_len):
        segment = tokens[:, start_idx:start_idx + segment_len]
        # Forward pass with memory
        outputs = model(segment, mems=mems, use_cache=True)
        logits, mems = outputs.logits, outputs.mem
        all_logits.append(logits)

    # Concatenate logits across segments (B, T, V)
    full_logits = torch.cat(all_logits, dim=1)
    return full_logits

# Example usage
if __name__ == '__main__':
    long_text = """
    [Your very long document here...]
    """
    logits = process_long_input(long_text, segment_len=1024)
    # Convert logits to probabilities or next-token predictions as needed
    print(logits.shape)  # e.g. [1, total_tokens, vocab_size]
