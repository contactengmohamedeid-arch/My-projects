# GPT-0.0
Learning implementation of a GPT model, built while following Andrej Karpathy's lessons. Not original research.

`Practical/GPT Arch.py` is one heavily commented file: GPT-2 vocabulary tokenizer (50,257 tokens via tiktoken), embeddings, positional encoding, transformer blocks, and a data pipeline with WikiText-103 from Hugging Face `datasets`, plus training-speed measurement and loss plots.
`how-to-train-your-gpt` is a link to another repository, not code stored here.

**Run:** `pip install torch tiktoken datasets numpy matplotlib`, then `python "Practical/GPT Arch.py"`.
