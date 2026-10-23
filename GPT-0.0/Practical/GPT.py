# ===== WHAT: Standard Python libraries =====
import math              # WHY: sqrt(), sin(), cos() for positional encoding math
import time              # WHY: measure training speed (tokens per second)
import os                # WHY: create directories, save/load model checkpoint files
from dataclasses import dataclass  # WHY: clean config class — no messy dictionaries

# ===== WHAT: NumPy — the CPU array library =====
import numpy as np       # WHY: fast numerical operations on CPU arrays
                         #      (mostly used for quick data checks, not heavy lifting)

# ===== WHAT: PyTorch — the neural network framework =====
import torch             # WHY: core library — tensors, GPU support, autograd
import torch.nn as nn               # WHY: neural network building blocks:
                                     #      Linear (dense layers), Embedding (lookup tables),
                                     #      Dropout (regularization), ModuleList (stacking layers)
import torch.nn.functional as F     # WHY: stateless functions used inside forward():
                                     #      softmax (convert to probabilities),
                                     #      cross_entropy (measure prediction error),
                                     #      silu (SwiGLU activation function)
from torch.utils.data import Dataset, DataLoader  # WHY: efficient data pipeline
#                                  Dataset = define how to load one sample
#                                  DataLoader = batch them, shuffle, prefetch

# ===== WHAT: tiktoken — OpenAI's fast BPE tokenizer =====
import tiktoken          # WHY: same Byte Pair Encoding tokenizer as GPT-3.5/GPT-4
                         #      Written in Rust, ~100x faster than pure Python tokenizers
                         #      Handles 50K+ vocabulary efficiently

# ===== WHAT: HuggingFace datasets — download training text =====
from datasets import load_dataset    # WHY: one line to download WikiText-103
                                     #      Handles caching (only downloads once),
                                     #      streaming (for datasets too big for disk),
                                     #      and format conversion automatically

# ===== WHAT: matplotlib — plot loss curves =====
import matplotlib.pyplot as plt      # WHY: visualize training progress
                                     #      Is the loss going down? Is it plateauing?
                                     #      A picture is worth 1,000 log lines



# ===== WHAT: Quick verification =====
# WHY: Always test your environment before writing 500 lines of code.
#      A missing import now saves hours of debugging later.
print("All imports ready!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")

#############################################################################################################################################################

@dataclass
class TokenizerConfig:
    """
    WHAT: Keeps all tokenizer settings in one place.
    WHY: Like a recipe card — consistent across the whole project.
         Change one value and everything updates automatically.
    """
    name: str = "gpt2"                # WHAT: use GPT-2's pretrained BPE tokenizer
                                       # WHY: same BPE as GPT-3/4 — 50K merges,
                                       #      battle-tested on billions of documents,
                                       #      and already trained (no weeks of work)
    vocab_size: int = 50257           # WHAT: total number of unique tokens
                                       # WHY: 50,257 is the exact GPT-2 vocabulary size
                                       #      (50,000 merges + 256 byte tokens + 1 EOS)
                                       #      This is the "goldilocks" number —
                                       #      big enough for rare subwords,
                                       #      small enough for fast matrix operations



class SimpleTokenizer:
    """
    WHAT: Wraps tiktoken to give us a friendly, consistent interface.
    WHY: tiktoken's raw API is low-level (you need to specify
         allowed_special every call). This wrapper makes encode/decode
         trivial — just call .encode("hello") and get tokens back.
         
         It also handles the EOS token consistently so we never
         accidentally forget to add it during training data prep.
    """

    def __init__(self, config: TokenizerConfig = None):
            """
            WHAT: Initialize the tokenizer with GPT-2's BPE vocabulary.
            WHY: We use a pretrained tokenizer because:
                1. Training a tokenizer from scratch takes weeks of CPU time
                2. GPT-2's tokenizer is open-source, fast, and well-tested
                3. Using the same tokenizer as production models means our
                    code works identically to how GPT-3 tokenizes
            """
            self.config = config or TokenizerConfig()

            # WHAT: Load the GPT-2 encoding from tiktoken
            # WHY: tiktoken stores pretrained BPE merge tables.
            #      get_encoding("gpt2") loads the exact 50K merges
            #      that GPT-2 was trained with.
            self.enc = tiktoken.get_encoding(self.config.name)

            # WHAT: Define and encode the End-of-Sequence token
            # WHY: <|endoftext|> is the special token that marks boundaries
            #      between documents. During training, we insert it between
            #      every document so the model learns where one text ends
            #      and another begins.
            self.eos_token = "<|endoftext|>"       # The string representation
            self.eos_token_id = self.enc.encode(    # Convert to its token ID
                self.eos_token,
                allowed_special={self.eos_token}    # WHY: tiktoken blocks special tokens
                                                    #      by default for safety. We must
                                                    #      explicitly allow EOS encoding.
            )[0]  # [0] because encode() returns a list — we want the single ID
    def encode(self, text: str) -> list[int]:
        """
        WHAT: Turn text into a list of integer token IDs.
        WHY: Neural networks only eat numbers. Raw strings like
             "Hello world" mean nothing to matrix multiplication.

        Example: "Hello world" -> [15496, 995]

        Under the hood: tiktoken splits the text into subword pieces
        using the pretrained BPE merge table, then looks up each
        piece's ID in the vocabulary.
        """
        # WHAT: Use tiktoken's fast C/Rust-based encoder
        # WHY: tiktoken is written in Rust, not Python.
        #      It can tokenize hundreds of MB of text per second.
        #      A pure Python BPE tokenizer would be 100x slower.
        return self.enc.encode(text, allowed_special={self.eos_token})

    def decode(self, ids: list[int]) -> str:
        """
        WHAT: Turn token IDs back into human-readable text.
        WHY: After the model generates a sequence of token IDs
             during inference, we need to convert them back to
             text so humans can read the output.

        Example: [15496, 995] -> "Hello world"
        """
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        """
        WHAT: How many unique tokens exist in the vocabulary.
        WHY: This number determines the size of our model's output
             layer — the final Linear layer must have vocab_size
             outputs (one score for each possible next token).
             
             50,257 means the model chooses from 50,257 possibilities
             every time it predicts the next word.
        """
        return self.config.vocab_size

#############################################################################################################################################################

# ===== WHAT: Quick self-test =====
# WHY: Always test each component in isolation before combining.
#      "Does the tokenizer work?" is a 5-second check that saves
#      hours of debugging a misbehaving training loop.
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()

    # Test 1: Basic text
    test_text = "The cat sat on the mat."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Test 1 — Basic:")
    print(f"  Original: '{test_text}'")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  '{decoded}'")
    print(f"  Match:    {test_text == decoded}")

    # Test 2: EOS token
    eos = tokenizer.encode(tokenizer.eos_token)
    print(f"\nTest 2 — EOS token:")
    print(f"  String: '{tokenizer.eos_token}'")
    print(f"  Token ID: {tokenizer.eos_token_id}")
    print(f"  Encode result: {eos}")

    # Test 3: Rare/unseen word
    rare = tokenizer.encode("antidisestablishmentarianism")
    decoded_rare = tokenizer.decode(rare)
    print(f"\nTest 3 — Rare word:")
    print(f"  Encoded: {rare}")
    print(f"  Pieces:  {[tokenizer.decode([t]) for t in rare]}")
    print(f"  Decoded: '{decoded_rare}'")

    # Test 4: Emoji/Unicode
    emoji = tokenizer.encode("Hello 😊 world")
    print(f"\nTest 4 — Emoji:")
    print(f"  Encoded: {emoji}")
    print(f"  Decoded: '{tokenizer.decode(emoji)}'")

    print(f"\n  Vocab size: {tokenizer.vocab_size:,}")


#############################################################################################################################################################


class Embedding(nn.Module):
    """
    WHAT: Converts token IDs into dense vectors (embeddings).
    WHY: A neural network can't do meaningful math on integer IDs
         like [9246, 6734]. It needs continuous numbers in vectors.

         Think of it as a giant lookup table:
         Row 9246 -> vector of 768 floats (the "meaning" of "cat")
         Row 6734 -> vector of 768 floats (the "meaning" of "sat")

         This table is LEARNED. Initially random, backpropagation
         gradually moves related tokens closer together in the
         768-dimensional space.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        WHAT: Create the embedding table (a learnable matrix).

        Args:
            vocab_size: How many unique tokens exist (50,257 for GPT-2)
            d_model:    Size of each embedding vector.

        Examples by model scale:
            GPT-2 small:  vocab=50257, d_model=768   → table is 50257 × 768
            GPT-2 medium: vocab=50257, d_model=1024  → table is 50257 × 1024
            GPT-3 small:  vocab=50257, d_model=4096  → table is 50257 × 4096
            GPT-3 large:  vocab=50257, d_model=12288 → table is 50257 × 12288

        WHY: The embedding dimension determines how much "space"
             each word has to express its meaning. Bigger d_model =
             more nuanced meanings can be captured, at the cost of
             more parameters and slower training.
        """
        super().__init__()

        # WHAT: The actual embedding weights — a [vocab_size, d_model] matrix
        # WHY: nn.Embedding is an optimized lookup table. When you pass
        #      a tensor of token IDs, it returns the corresponding rows.
        #      It's backed by a standard weight matrix, so gradients
        #      flow through it just like any nn.Linear layer.
        #
        #      Internally, nn.Embedding is essentially:
        #      def forward(self, x):
        #          return self.weight[x]  # index into the weight matrix
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        WHAT: Look up embeddings for each token ID in the input.

        Input shape:  [batch_size, seq_len]    — each cell is a token ID
        Output shape: [batch_size, seq_len, d_model] — each cell is a vector

        Example walkthrough:
            Input:  [[464, 3797]]              # ["The", "cat"]
            Step 1: Look up row 464 → [768 floats] for "The"
                    Look up row 3797 → [768 floats] for "cat"
            Step 2: Scale by sqrt(768) ≈ 27.7
            Output: [[[v0..v767], [v0..v767]]] # 2 vectors of 768 numbers

        WHY each dimension:
            batch_size = how many sequences we process at once (parallelism)
            seq_len    = how many tokens per sequence (context window)
            d_model    = how rich each token's representation is (expressiveness)
        """
        # WHAT: Index into the embedding matrix
        # WHY: For each token ID, return its row. This is an O(1)
        #      lookup operation — very fast, even for 50K+ vocabulary.
        embeddings = self.embed(x)  # [batch, seq_len, d_model]

        # WHAT: Return embeddings unchanged
        # WHY: We use RoPE for position encoding. RoPE rotates rather
        #      than adds, so no scaling is needed. LLaMA and Mistral
        #      follow this same convention.
        return embeddings
    