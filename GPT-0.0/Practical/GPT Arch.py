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

#############################################################################################################################################################

class RotaryPositionalEmbedding(nn.Module):
    """
    WHAT: Rotary Position Embeddings (RoPE).
    WHY: Instead of ADDING position info to embeddings,
         we ROTATE Q and K vectors by position-dependent angles.
         The dot product q_i · k_j then depends ONLY on (j-i),
         which is exactly what attention should care about.

         Paper: "RoFormer" (Su et al., 2021)
         Used in: LLaMA 1/2/3, Mistral, Mixtral, Qwen 1/2, Gemma

         How it works at a glance:
         1. For each pair of dimensions (0,1), (2,3), (4,5), ...
         2. Rotate by angle = position * frequency
         3. Lower dims rotate fast (local position)
            Higher dims rotate slow (global position)
         4. The dot product naturally depends on relative distance
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        WHAT: Precompute rotation frequencies for fast lookup.

        Args:
            d_model:     Head dimension (e.g., 64 for GPT-2). Must be even.
            max_seq_len: Precompute angles for positions 0..max_seq_len-1.
            theta:       Base frequency. 10000 is standard. Controls the
                         spread between fast and slow rotation frequencies.
        """
        super().__init__()

        # WHAT: Verify d_model is even (must have pairs to rotate)
        assert d_model % 2 == 0, (
            f"d_model ({d_model}) must be even for RoPE. "
            f"Each pair of dimensions needs a partner to rotate with."
        )

        # WHAT: Create dimension indices: [0, 2, 4, ..., d_model-2]
        # WHY: Each pair (2i, 2i+1) gets the same rotation frequency.
        #      We only need half the indices because pairs share.
        dim_indices = torch.arange(0, d_model, 2).float()

        # WHAT: Compute rotation frequencies
        # WHY: theta_i = 1 / (theta ^ (2i / d_model))
        #
        #      i=0:  1 / 10000^(0/64)      = 1.0      → fast rotation (local)
        #      i=30: 1 / 10000^(60/64)     ≈ 0.0001   → slow rotation (global)
        #
        #      This multi-scale approach means some dimensions
        #      capture local word order while others capture
        #      long-range position relationships.
        inv_freq = 1.0 / (theta ** (dim_indices / d_model))

        # WHAT: Precompute angles for all positions
        # WHY: Computing cos/sin during training is expensive.
        #      Precomputing them once and caching is 100x faster.
        positions = torch.arange(max_seq_len).float()     # [0, 1, 2, ..., 2047]

        # WHAT: Outer product: each position x each frequency
        #       freqs[p, i] = p * inv_freq[i] = angle for position p, dim pair i
        #       Shape: [max_seq_len, d_model/2]
        freqs = torch.outer(positions, inv_freq)

        # WHAT: Duplicate to full dimension
        # WHY: Each dim pair (2i, 2i+1) gets the same angle,
        #      so we copy each angle: [θ0, θ1, θ2, ...] -> [θ0, θ0, θ1, θ1, ...]
        emb = freqs.repeat_interleave(2, dim=-1)         # [max_seq_len, d_model]

        # WHAT: Cache cos and sin for all positions
        # WHY: register_buffer means these move with model.to(device)
        #      and are saved with model.state_dict(), but are NOT
        #      trainable parameters (no gradients needed).
        self.register_buffer("cos_cached", emb.cos())   # cos of each angle
        self.register_buffer("sin_cached", emb.sin())   # sin of each angle


    @staticmethod
    def rotate_half(x):
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        return torch.stack([-x_odd, x_even], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        WHAT: Apply RoPE to queries or keys.

        Input:  [batch, num_heads, seq_len, head_dim]
                x can be either Q or K (NOT V — values don't need position)
        Output: Same shape, rotated by position-dependent angles

        WHY applied only to Q and K:
        The attention score = Q_i · K_j controls WHICH values to attend to.
        We want this score to depend on relative position.
        The VALUE vectors carry content — position is irrelevant for the
        content itself. Position only matters for deciding which tokens
        to pay attention TO.
        """
        # WHAT: Extract cos and sin for current sequence length
        # WHY: If seq_len=512 but max_seq_len=2048, we only need
        #      the first 512 rows of the cached cos/sin tables.
        cos = self.cos_cached[:seq_len]   # [seq_len, head_dim]
        sin = self.sin_cached[:seq_len]   # [seq_len, head_dim]

        # WHAT: Add batch and head dimensions for broadcasting
        # WHY: cos/sin are [seq_len, head_dim]. We need them to
        #      multiply with x [batch, heads, seq_len, head_dim].
        #      unsqueeze(0).unsqueeze(0) adds dims at positions 0 and 1:
        #      [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        #      Now they broadcast correctly over batch and heads.
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # WHAT: Execute rotation: x_rotated = x*cos(θ) + rotate_half(x)*sin(θ)
        # WHY: This is mathematically equivalent to applying a 2D rotation
        #      matrix to each pair of dimensions, but implemented in pure
        #      element-wise operations — much faster and parallelizable.
        return (x * cos) + (self.rotate_half(x) * sin)

#############################################################################################################################################################

class MultiHeadAttention(nn.Module):
    """
    WHAT: Multi-Head Self-Attention with RoPE and causal masking.

    WHY: Transformers would be useless without attention. This is the
         mechanism that lets each token "look at" every other token and
         decide how much each matters for understanding the current context.

         Each attention head:
         1. Projects input into Query, Key, Value spaces
         2. Computes Q·K^T / sqrt(d_k) → how well each query matches each key
         3. Applies causal mask → no peeking at future tokens
         4. Softmax → converts scores to a probability distribution
         5. Weighted sum of Values → builds context-aware representation

         Doing this with multiple heads in parallel lets each head
         specialize in different linguistic patterns.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model:   Total embedding dimension (e.g., 768 for GPT-2 small)
            num_heads: Number of parallel attention heads (e.g., 12)
            dropout:   Probability of randomly zeroing attention weights

        WHY: d_model must be divisible by num_heads because each head
             operates on d_model/num_heads dimensions (64 for GPT-2 small).
             This split-then-concat strategy lets heads specialize while
             keeping total parameter count the same as a single large head.
        """
        super().__init__()

        # WHAT: Validate that heads evenly divide the model dimension
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
            f"This ensures each head has equal dimension."
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 768/12 = 64 dimensions per head
                                               # WHY: 64 is the "sweet spot" —
                                               # enough to capture meaning,
                                               # small enough for efficient compute

        # ===== QKV Projection =====
        # WHAT: One big linear layer that projects input to Q, K, V simultaneously
        # WHY:  3 separate Linear(768→768) layers = 3 matrix multiplies.
        #       One combined Linear(768→2304) = 1 bigger matrix multiply.
        #       On GPU, 1 big operation is much faster than 3 small ones
        #       due to better parallelism and fewer kernel launches.
        #       Shape: [d_model, 3 * d_model] = [768, 2304]
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # ===== Output Projection =====
        # WHAT: Project concatenated head outputs back to d_model
        # WHY:  After concatenation: [batch, seq, d_model] but each head's
        #       output was computed independently. This linear layer MIXES
        #       information across heads, letting them communicate.
        #       Without it, heads would stay isolated — like 12 experts
        #       who never talk to each other.
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # ===== RoPE (Rotary Position Embeddings) =====
        # WHAT: Applies rotation-based position encoding to Q and K only
        # WHY:  RoPE encodes position into the Q and K vectors so that
        #       the dot product Q·K naturally depends on RELATIVE position.
        #       We apply to the head_dim (not d_model) because each head
        #       needs its own position info in its subspace.
        #       V does NOT get RoPE because values carry content, not
        #       position — position is only relevant for deciding
        #       WHICH values to attend to, not the values themselves.
        self.rotary = RotaryPositionalEmbedding(self.head_dim)

        # ===== Dropout =====
        # WHAT: Randomly zero out attention weights during training
        # WHY:  Without dropout, the model can become overconfident —
        #       one token always dominates attention, ignoring other
        #       potentially useful context. Dropout forces the model
        #       to learn redundant attention patterns (backup plans).
        self.attn_dropout = nn.Dropout(dropout)   # Applied to attention weights
        self.resid_dropout = nn.Dropout(dropout)  # Applied to final output

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        WHAT: Compute multi-head self-attention.

        Input:  x    [batch, seq_len, d_model]  — token embeddings
                mask [batch, 1, seq, seq]       — causal mask (1=visible, 0=masked)

        Output:      [batch, seq_len, d_model]  — context-aware representations

        The forward pass has 8 steps, each critical:
        """
        batch_size, seq_len, _ = x.shape

        # ===== STEP 1: Project input to Q, K, V — all at once =====
        # WHAT: Linearly transform input into query, key, value spaces
        # WHY:  Combined projection is faster on GPU than 3 separate ones.
        #       After this: [batch, seq, 3*d_model] where the last dim
        #       has Q values first, then K values, then V values.
        qkv = self.qkv_proj(x)               # [batch, seq, 3 * d_model]

        # ===== STEP 2: Reshape to expose the head dimension =====
        # WHAT: Split the 3*d_model into separate Q,K,V and separate heads
        # WHY:  We need shape [batch, num_heads, seq, head_dim] for
        #       parallel computation. The reshape + permute does this
        #       in two efficient operations without data copies.
        #
        # Transform: [batch, seq, 3, heads, head_dim]
        # Then permute: [3, batch, heads, seq, head_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # [3, batch, heads, seq, head_dim]

        # WHAT: Unpack the three projections
        q = qkv[0]  # Query:  [batch, heads, seq, head_dim] — "what I'm looking for"
        k = qkv[1]  # Key:    [batch, heads, seq, head_dim] — "what I offer to match"
        v = qkv[2]  # Value:  [batch, heads, seq, head_dim] — "my actual content"

        # ===== STEP 3: Apply Rotary Position Embeddings =====
        # WHAT: Rotate Q and K by position-dependent angles
        # WHY:  After rotation, the dot product q_i · k_j depends on
        #       cos(i-j) and sin(i-j) — the RELATIVE distance between
        #       tokens i and j. This is what we want: attention should
        #       care about "how far apart are these tokens?" not
        #       "what are their absolute positions?"
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)

        # ===== STEP 4: Compute attention scores (Q · K^T) =====
        # WHAT: For each query token, compute dot product with every key token
        # WHY:  Dot product measures cosine similarity (if vectors normalized).
        #       Higher dot product = query "wants" what key "offers".
        #
        #       Shape: [batch, heads, query_seq, key_seq]
        #       attn_scores[b, h, i, j] = how much token i attends to token j
        #
        #       DIVIDE BY sqrt(head_dim): critical for stable training.
        #       Without this, the variance of dot products grows with d_k,
        #       making softmax too "peaky" → gradients vanish → model dies.
        #       See Part 4 above for the mathematical derivation.
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ===== STEP 5: Apply causal mask — no peeking at future tokens =====
        # WHAT: Set attention scores to future tokens to -infinity
        # WHY:  During training, the model must predict token[i+1] from
        #       tokens[0..i]. If token[i] can see token[i+1], it's like
        #       seeing the answer before the question — cheating.
        #
        #       -infinity → e^(-inf) = 0.0 after softmax = zero attention
        #
        #       The mask is lower-triangular:
        #       Token 0 → sees [0]        (itself only)
        #       Token 1 → sees [0, 1]     (itself + previous)
        #       Token 2 → sees [0, 1, 2]  (itself + all previous)
        #       Token 3 → sees [0, 1, 2, 3]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # ===== STEP 6: Softmax — scores become attention weights =====
        # WHAT: Convert raw scores to a probability distribution over keys
        # WHY:  softmax(scores)[j] = e^score[j] / sum(e^score[k] for k in all keys)
        #       This makes all weights:
        #       - Positive (e^x > 0 always)
        #       - Sum to 1.0 (proper probability distribution)
        #       - Differentiable (we can compute gradients through it)
        #
        #       The softmax is applied over the LAST dimension (dim=-1),
        #       which is the "key" dimension — so each query gets a
        #       distribution over all keys it can see.
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # ===== STEP 7: Weighted sum of values =====
        # WHAT: Mix the value vectors according to attention weights
        # WHY:  This is WHERE attention actually happens. Each query
        #       token gets a NEW vector that is a weighted blend of
        #       all visible value vectors.
        #
        #       High attention to token j → V_j has large influence
        #       Low attention to token j → V_j has small influence
        #
        #       The result is "context-aware" — each token now "knows"
        #       about the other relevant tokens in the sequence.
        #
        #       [batch, heads, seq, head_dim] @ [batch, heads, seq, head_dim]
        #       → [batch, heads, seq, head_dim]
        attn_output = attn_weights @ v

        # ===== STEP 8: Merge heads and project =====
        # WHAT: Combine all head outputs into one d_model vector per token
        # WHY:  Currently: [batch, heads, seq, head_dim]
        #       Need:       [batch, seq, d_model]
        #
        #       Transpose swaps heads and sequence:
        #       [batch, seq, heads, head_dim]
        #       Reshape flattens heads×head_dim:
        #       [batch, seq, d_model]
        #
        #       The final linear projection lets information flow between
        #       heads — each head's discoveries can now influence the
        #       combined representation.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        output = self.out_proj(attn_output)   # Mix across heads
        output = self.resid_dropout(output)   # Regularization

        return output

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    WHAT: Create a causal (lower triangular) attention mask.
    WHY:  Prevents tokens from attending to future tokens during training.

    Visual for seq_len=6:
        [[✓, ✗, ✗, ✗, ✗, ✗],     Token 0 (first word)
         [✓, ✓, ✗, ✗, ✗, ✗],     Token 1
         [✓, ✓, ✓, ✗, ✗, ✗],     Token 2
         [✓, ✓, ✓, ✓, ✗, ✗],     Token 3
         [✓, ✓, ✓, ✓, ✓, ✗],     Token 4
         [✓, ✓, ✓, ✓, ✓, ✓]]     Token 5 (last word — sees everything)

    ✓ = position is visible (1.0)
    ✗ = position is masked (0.0, becomes -inf in attention)

    Reshaped to [1, 1, seq_len, seq_len] for broadcasting over:
    - batch dimension (all batches use same mask)
    - head dimension (all heads use same mask — heads CAN'T see future)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)

#############################################################################################################################################################

class RMSNorm(nn.Module):
    """
    WHAT: Root Mean Square Layer Normalization.
    WHY: Normalizes each token's representation so its magnitude is ~1.0.
         Prevents values from growing/shrinking across deep networks.

         Used in: LLaMA 1/2/3, Mistral, Gemma, Qwen
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # WHAT: Learnable scale per dimension
        # WHY: After forcing RMS=1, the model can learn to amplify
        #      important dimensions and dampen unimportant ones.
        #      Starts at 1.0 (no change initially).
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps  # WHY: prevents division by zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # WHAT: Compute 1/sqrt(mean(x²))
        # WHY: rsqrt is 1/sqrt — computed as a single CUDA kernel
        #      for speed. The mean is over the last dimension (d_model).
        #      keepdim=True preserves the dimension for broadcasting.
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # WHAT: Normalize then learnable-scale
        return x * rms * self.weight

class SwiGLU(nn.Module):
    """
    WHAT: SwiGLU — gated version of Swish activation.
    WHY: The "gate" (right side of multiplication) learns to
         selectively pass or block information — like a faucet.

         Standard FFN:  output = W2(ReLU(W1(x)))
         SwiGLU FFN:    output = W3(SiLU(W1(x)) * (W2(x)))
                                   ^^^^^^^^      ^^^^^^
                                   values        gate

         The gate multiplies values: if gate ≈ 0, block info.
                                     if gate ≈ 1, pass info.
                                     if gate ≈ 0.5, partial pass.

         This gating mechanism is what makes SwiGLU outperform
         ReLU and GELU — the model learns WHERE to apply non-linearity.

         Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
         Used in: LLaMA 1/2/3, PaLM, Gemini
    """

    def __init__(self, d_model: int, expansion_factor: int = 4):
        super().__init__()

        # WHAT: Hidden dim is 4x input/output — the "expansion" bottleneck
        # WHY: Expand→process→contract is more expressive than same-size.
        #      784 → 3072 → 784 lets the FFN learn ~4x more complex patterns.
        hidden_dim = expansion_factor * d_model

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)   # Projects to values
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)   # Projects to gates
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)   # Projects back

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # WHAT: SiLU(w1(x)) are the values, w2(x) are the gates
        # WHY: SiLU (also called Swish) = x * sigmoid(x)
        #      It's smooth (unlike ReLU which has a sharp corner at 0),
        #      which makes gradients flow better during training.
        #      Gate multiplies values element-wise, selectively passing info.
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    """
    WHAT: One complete Transformer layer (attention + FFN with residuals).
    WHY: Stack N of these to build a deep language model.

         Architecture (Pre-Norm):
         ┌─────────────────────────────────────┐
         │ x = x + Attention(RMSNorm(x), mask) │  ← Mix information BETWEEN tokens
         │ x = x + SwiGLU(RMSNorm(x))          │  ← Process information WITHIN tokens
         └─────────────────────────────────────┘

         Each sublayer: normalize FIRST (pre-norm), then compute,
         then ADD back the original (residual connection).

         Without residuals: deep networks can't train (vanishing gradients)
         Without pre-norm: training is unstable at large depths
         Without FFN: no non-linear processing per token
         Without attention: no information mixing between tokens
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # WHAT: First normalization — before attention
        # WHY: Pre-norm: clean, well-scaled input → stable attention computation
        self.norm1 = RMSNorm(d_model)

        # WHAT: Multi-head self-attention with RoPE and causal masking
        # WHY: The core mechanism that lets tokens "talk to" each other
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # WHAT: Second normalization — before FFN
        # WHY: FFN expects normalized input for consistent behavior across layers
        self.norm2 = RMSNorm(d_model)

        # WHAT: SwiGLU feed-forward network
        # WHY: Non-linear processing per token. Without this, stacking more
        #      attention layers would be no more powerful than one layer.
        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass: norm → sublayer → add residual.
        Executed twice: once for attention, once for FFN.
        """

        # ===== SUB-LAYER 1: Self-Attention with residual =====
        # WHAT: x = x + Attention(Norm(x))
        # WHY: The model learns what CHANGES (the delta) to make to x,
        #      not what to replace x with entirely. This is easier to learn.
        #      If attention can't improve things, it can output near-zero.
        x = x + self.attention(self.norm1(x), mask)

        # ===== SUB-LAYER 2: Feed-Forward with residual =====
        # WHAT: x = x + FFN(Norm(x))
        # WHY: Same residual pattern. After mixing information via attention,
        #      each token "thinks" independently via the FFN.
        #      Attention = group discussion. FFN = private reflection.
        x = x + self.ffn(self.norm2(x))

        return x

#############################################################################################################################################################
@dataclass
class GPTConfig:
    """
    WHAT: All hyperparameters in one place.
    WHY: Changing model size is one line. No hunting through code.
    """
    # ===== Architecture =====
    vocab_size: int = 50257        # WHAT: 50,257 unique tokens in GPT-2 vocabulary
    d_model: int = 768             # WHAT: Each token becomes a 768-dim vector
                                   # WHY: Bigger = more nuanced meanings, more compute
    num_heads: int = 12            # WHAT: 12 attention heads (12 × 64 = 768)
    num_layers: int = 12           # WHAT: 12 transformer blocks stacked
                                   # WHY: Deeper = better reasoning, harder to train
    max_seq_len: int = 1024        # WHAT: Max tokens model can process at once

    # ===== Regularization (prevent overfitting) =====
    dropout: float = 0.1           # WHAT: Randomly disable 10% of neurons during training
    embd_dropout: float = 0.1      # WHAT: Dropout applied right after embedding lookup

    # ===== Training =====
    learning_rate: float = 3e-4    # WHAT: Step size for weight updates
    weight_decay: float = 0.1      # WHAT: Penalize large weights (L2 regularization)
    warmup_steps: int = 2000       # WHAT: Gradually increase LR for first 2000 steps
    max_steps: int = 100000        # WHAT: Total training iterations
    batch_size: int = 8            # WHAT: Sequences processed per GPU step
    grad_accum_steps: int = 4      # WHAT: Accumulate gradient steps (effective batch = 8×4 = 32)
    betas: tuple = (0.9, 0.95)    # WHAT: AdamW momentum coefficients
    eps: float = 1e-8              # WHAT: Small constant preventing division by zero

    def __post_init__(self):
        """Validate configuration consistency."""
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )

#############################################################################################################################################################

class GPT(nn.Module):
    """
    WHAT: A complete decoder-only Transformer language model.
    WHY: This single class combines everything we've built:
         embedding → N× transformer blocks → output projection.

         "Decoder-only" means it generates text left-to-right
         (causal/autoregressive), without an encoder that would
         look at the full sequence.

         This is the same architecture family as:
         GPT-2 (12 layers, 768 dims), GPT-3 (96 layers, 12288 dims),
         LLaMA 3 (32-80 layers), Mistral (32 layers)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ===== 1. TOKEN EMBEDDING =====
        # WHAT: Lookup table: token ID → dense vector
        # WHY: Converts integers (IDs) into the continuous vectors
        #      that neural networks can work with.
        #      Shape: [50257, 768] — one row per vocabulary token
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # WHAT: Dropout applied to embeddings
        # WHY: Early dropout prevents the model from overfitting to
        #      specific embedding values during training
        self.embd_dropout = nn.Dropout(config.embd_dropout)

        # ===== 2. TRANSFORMER BLOCKS =====
        # WHAT: Stack of N identical transformer layers
        # WHY: nn.ModuleList registers each block so PyTorch tracks
        #      their parameters for training. A regular Python list
        #      would NOT be tracked!
        #
        #      Each block: RMSNorm → Attention(+residual) → RMSNorm → FFN(+residual)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])

        # ===== 3. FINAL NORMALIZATION =====
        # WHAT: One last RMSNorm before the output head
        # WHY: The output of the last transformer block is raw (unnormalized).
        #      We normalize before projecting to vocabulary so the
        #      LM head gets clean, well-scaled inputs.
        self.final_norm = RMSNorm(config.d_model)

        # ===== 4. LM HEAD (output projection) =====
        # WHAT: Linear projection: d_model → vocab_size
        # WHY: Transforms each token's 768-dim "understanding"
        #      into a 50257-dim vector of scores — one score per
        #      possible next token.
        #
        #      logits[b, t, v] = "score for token v being the next
        #                         word after position t in batch b"
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ===== 5. WEIGHT TYING =====
        # WHAT: Share weight matrix between embedding and LM head
        # WHY: The embedding maps token → vector. The LM head maps
        #      vector → token. These are INVERSE operations!
        #
        #      Sharing weights has three benefits:
        #      1. Parameter efficiency: saves 50257×768 = 38.6M params
        #         (30% of the total for GPT-2 small!)
        #      2. Better regularization: the shared matrix gets
        #         gradient signals from both directions, improving
        #         the quality of token representations
        #      3. Theoretical elegance: input and output tokens
        #         live in the same semantic space
        #
        #      How it works: setting self.lm_head.weight to point to
        #      the SAME tensor as self.token_embedding.weight means
        #      PyTorch uses the same memory for both.
        self.token_embedding.weight = self.lm_head.weight

        # ===== 6. WEIGHT INITIALIZATION =====
        # WHAT: Initialize all weights with Normal(0, 0.02)
        # WHY: Starting from the right distribution is critical.
        #      Too small → gradients vanish, model never learns.
        #      Too large → activations saturate, gradients explode.
        #      0.02 std gives values mostly in [-0.04, 0.04],
        #      which is the sweet spot for Transformers.
        self.apply(self._init_weights)
        print(f"GPT initialized with {self.get_num_params():,} parameters")

    def _init_weights(self, module: nn.Module):
        """Initialize weights using the GPT-2 scheme."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Count total trainable parameters (weights + biases)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None
    ) -> tuple:
        """
        WHAT: Process a batch of token sequences through the GPT model.

        Args:
            input_ids: [batch_size, seq_len] — token IDs for each sequence
            targets:   [batch_size, seq_len] — same tokens, used for loss
                       (the model predicts input_ids[t+1] from input_ids[t])

        Returns:
            logits: [batch, seq_len, vocab_size] — raw prediction scores
            loss:   scalar — cross-entropy (None if targets not provided)

        The dataset already provides shifted targets, so we compute the
        loss on all positions directly (no manual slicing needed).
        """
        batch_size, seq_len = input_ids.shape

        # ===== 1. EMBED TOKENS =====
        # Input:  [batch, seq] token IDs
        # Output: [batch, seq, d_model] continuous vectors
        x = self.token_embedding(input_ids)
        x = self.embd_dropout(x)

        # ===== 2. CREATE CAUSAL MASK =====
        # WHAT: Lower-triangular mask: token i can only see tokens 0..i
        # WHY: Without this, the model would "cheat" by looking at
        #      future tokens when predicting the next one.
        mask = create_causal_mask(seq_len, input_ids.device)

        # ===== 3. TRANSFORMER LAYERS =====
        # WHAT: Pass through all N transformer blocks sequentially
        # WHY: Each layer refines the representations. Early layers
        #      capture syntax. Later layers capture semantics.
        for layer in self.layers:
            x = layer(x, mask)

        # ===== 4. FINAL NORMALIZATION =====
        x = self.final_norm(x)

        # ===== 5. PROJECT TO VOCABULARY =====
        # WHAT: Convert from d_model-dim "understanding" to vocab_size-dim scores
        # WHY: Each position gets a score for every possible next token.
        #
        # Example: logits[0, 3, 2603] = 9.2 means:
        #   "For batch 0, position 3, the score for token 2603 ('mat') is 9.2"
        #   Higher score = model thinks this token is more likely.
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        # ===== 6. COMPUTE LOSS (training only) =====
        loss = None
        if targets is not None:
            # WHAT: Flatten logits and targets, then compute cross-entropy.
            # WHY: The dataset already provides shifted targets, so we
            #      compute the loss on all positions directly.
            logits_flat = logits.contiguous().view(
                -1, self.config.vocab_size
            )
            targets_flat = targets.contiguous().view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

#############################################################################################################################################################

class TextDataset(Dataset):
    """
    WHAT: Prepares text data by splitting into training chunks.
    WHY: The model learns to predict the next token. Each chunk
         provides input-target pairs for next-token prediction.

         Each sample: input[t] and target[t+1] for all positions t.
         This is called "teacher forcing" — we show the correct
         answer for every position during training.
    """

    def __init__(self, texts: list[str], tokenizer, max_seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # ===== Concatenate all texts with EOS separators =====
        # WHY: EOS prevents the model from learning false connections
        #      between unrelated documents.
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)  # Document boundary marker

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens in dataset: {len(self.tokens):,}")

    def __len__(self) -> int:
        """Number of chunks. Each uses max_seq_len+1 tokens."""
        return (len(self.tokens) - 1) // self.max_seq_len

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns (input_ids, target_ids) for one chunk.
        Target is shifted by 1 position:

        tokens:    [The,  cat,  sat,  on,   the,  mat,  EOS,  The,  dog,  ...]
        idx=0:     [The,  cat,  sat,  on,   the]     ← input_ids
                   [cat,  sat,  on,   the,  mat]     ← target_ids (shifted)
        """
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1 : end + 1]
        return input_ids, target_ids
