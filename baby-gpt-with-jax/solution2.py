import os
import math
import tiktoken
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple
from time import time
import pickle
from functools import partial

from helpers import save_object, load_object

# Force JAX to use the CPU and expose available CPU cores to JAX
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_num_cpu_devices', max(1, os.cpu_count() or 1))

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
class Config:
    # --- Local File Path ---
    # dataset_path = "TinyStoriesV2-GPT4-valid.txt"  # Set your local file path here
    dataset_path = "TinyStories-1000.txt"  # Set your local file path here
    model_path = "baby_gpt_model_params.pkl"  # Path to save/load model parameters
    tokenizer_path = "baby_gpt_tokenizer.pkl"  # Path to save/load tokenizer
    loss_results_path = "loss_training_results.pkl"  # Path to save training loss results

    # --- Architecture Settings ---
    vocab_size = 50257  # Standard GPT-2 vocab size (tiktoken)
    max_len = 256       # Reduced context length for CPU efficiency
    embed_dim = 256     # Small embedding size
    num_heads = 4       # Number of attention heads
    num_layers = 4      # Number of transformer layers
    lr = 5e-4           # Learning rate
    batch_size = 8      # Small batch size for CPU
    steps = 5_000        # Total training iterations
    # steps = 200

    # --- Early Stopping Settings ---
    early_stopping = True
    early_stop_patience = 200
    early_stop_min_delta = 1e-4
    early_stop_eval_interval = 100
    val_split_ratio = 0.05
    val_seed = 7
    use_data_parallelism = True
    
    # --- Advanced Sampling Settings ---
    temperature = 0.2   # Controls randomness (lower = more deterministic)
    top_k = 50          # Limits sampling to the top K highest-probability tokens (0 to disable)
    top_p = 0.9         # Nucleus sampling: keeps top tokens adding up to cumulative prob P (0.0 to disable)

# Model Parameter Containers
class LayerParams(NamedTuple):
    q_proj: jnp.ndarray; k_proj: jnp.ndarray; v_proj: jnp.ndarray; out_proj: jnp.ndarray
    ln1_w: jnp.ndarray; ln1_b: jnp.ndarray
    fc1: jnp.ndarray; fc2: jnp.ndarray
    ln2_w: jnp.ndarray; ln2_b: jnp.ndarray

class ModelParams(NamedTuple):
    token_emb: jnp.ndarray
    pos_emb: jnp.ndarray
    layers: list[LayerParams]
    ln_f_w: jnp.ndarray
    ln_f_b: jnp.ndarray
    
# ==========================================
# Donate parameters to improve performance
# ==========================================
# @partial(
#     jax.jit,
#     donate_argnums=(0,1),
#     static_argnames=("num_heads",),
# )
    
# ==========================================
# Define the mask for causal attention
# ==========================================
# CAUSAL_MASK = jnp.tril(
#     jnp.ones(
#         (Config.max_len, Config.max_len),        
#     )
# )
CAUSAL_MASK = jnp.tril(
    jnp.ones((Config.max_len, Config.max_len), dtype=bool)
)

# ==========================================
# 2. LOCAL DATA LOADING
# ==========================================
def get_data_loader(file_path, batch_size, max_len):
    """Loads tokenized text from a local path and yields training/validation batches."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find local file at '{file_path}'. "
            f"Please verify the path or update Config.dataset_path."
        )

    enc = tiktoken.get_encoding("gpt2")

    print(f"Reading local dataset from: {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    print("Tokenizing entire dataset into RAM...")
    tokens = enc.encode_ordinary(data)
    tokens_np = jnp.array(tokens, dtype=jnp.int32)
    num_tokens = len(tokens_np)
    print(f"Dataset loaded. Total tokens found: {num_tokens}")

    val_size = max(1, int(num_tokens * Config.val_split_ratio))
    if val_size >= num_tokens - max_len - 1:
        val_size = max(1, min(16, num_tokens // 10))

    # Keep the validation split deterministic and lightweight by using a held-out suffix.
    split_idx = num_tokens - val_size - max_len - 1
    if split_idx <= 0:
        raise ValueError("Dataset is too small for the configured validation split and sequence length.")

    train_slice = tokens_np[: split_idx + max_len + 1]
    val_slice = tokens_np[split_idx:]

    def build_batches(token_stream):
        def batch_generator():
            key = jax.random.PRNGKey(42)
            while True:
                key, subkey = jax.random.split(key)
                idx = jax.random.randint(subkey, (batch_size,), 0, len(token_stream) - max_len - 1)
                
                # Original implementation
                # idx_list = idx.tolist()
                # x = jnp.stack([token_stream[i : i + max_len] for i in idx_list])
                # y = jnp.stack([token_stream[i + 1 : i + max_len + 1] for i in idx_list])
                
                # Better (vectorized) implementation
                offsets = jnp.arange(max_len)
                x = token_stream[idx[:, None] + offsets]
                y = token_stream[idx[:, None] + offsets + 1]
                
                yield x, y

        return batch_generator()

    train_loader = build_batches(train_slice)
    val_loader = build_batches(val_slice)

    return train_loader, val_loader, enc

# ==========================================
# 3. INITIALIZATION
# ==========================================
def init_params(key, config):
    """Initializes weights using standard normal scaled distributions."""
    k1, k2, *layer_keys = jax.random.split(key, 2 + config.num_layers)
    
    # Embedding matrices
    token_emb = jax.random.normal(k1, (config.vocab_size, config.embed_dim)) * 0.02
    pos_emb = jax.random.normal(k2, (config.max_len, config.embed_dim)) * 0.02
    
    layers = []
    for l_key in layer_keys:
        k_q, k_k, k_v, k_o, k_f1, k_f2 = jax.random.split(l_key, 6)
        
        # Attention weights
        q = jax.random.normal(k_q, (config.embed_dim, config.embed_dim)) * (1.0 / math.sqrt(config.embed_dim))
        k = jax.random.normal(k_k, (config.embed_dim, config.embed_dim)) * (1.0 / math.sqrt(config.embed_dim))
        v = jax.random.normal(k_v, (config.embed_dim, config.embed_dim)) * (1.0 / math.sqrt(config.embed_dim))
        out = jax.random.normal(k_o, (config.embed_dim, config.embed_dim)) * (1.0 / math.sqrt(config.embed_dim))
        
        # MLP Weights
        fc1 = jax.random.normal(k_f1, (config.embed_dim, config.embed_dim * 4)) * (1.0 / math.sqrt(config.embed_dim))
        fc2 = jax.random.normal(k_f2, (config.embed_dim * 4, config.embed_dim)) * (1.0 / math.sqrt(config.embed_dim * 4))
        
        layers.append(LayerParams(
            q_proj=q, k_proj=k, v_proj=v, out_proj=out,
            ln1_w=jnp.ones(config.embed_dim), ln1_b=jnp.zeros(config.embed_dim),
            fc1=fc1, fc2=fc2,
            ln2_w=jnp.ones(config.embed_dim), ln2_b=jnp.zeros(config.embed_dim)
        ))
        
    return ModelParams(
        token_emb=token_emb, pos_emb=pos_emb, layers=layers,
        ln_f_w=jnp.ones(config.embed_dim), ln_f_b=jnp.zeros(config.embed_dim)
    )

# ==========================================
# 4. MODEL FORWARD PASS
# ==========================================
def layer_norm(x, w, b, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return w * (x - mean) / jnp.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.pow(x, 3))))

def causal_attention(x, params: LayerParams, num_heads):
    seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    
    q = (x @ params.q_proj).reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
    k = (x @ params.k_proj).reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
    v = (x @ params.v_proj).reshape(seq_len, num_heads, head_dim).swapaxes(0, 1)
    
    # scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(head_dim)
    scale = 1.0 / jnp.sqrt(head_dim)
    scores = (q @ k.swapaxes(-1,-2)) * scale
    
    # Original code
    # mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    # scores = jnp.where(mask == 1.0, scores, -1e9)
    
    # New code - will improve performance
    mask = CAUSAL_MASK[:seq_len, :seq_len]
    scores = jnp.where(mask, scores, -jnp.inf)
    
    attn_weights = jax.nn.softmax(scores, axis=-1)
    context = (attn_weights @ v).swapaxes(0, 1).reshape(seq_len, embed_dim)
    
    return context @ params.out_proj

def forward(params: ModelParams, x):
    seq_len = x.shape[0]
    h = params.token_emb[x] + params.pos_emb[:seq_len]
    
    for layer in params.layers:
        h = h + causal_attention(layer_norm(h, layer.ln1_w, layer.ln1_b), layer, Config.num_heads)
        h = h + gelu(layer_norm(h, layer.ln2_w, layer.ln2_b) @ layer.fc1) @ layer.fc2
        
    h = layer_norm(h, params.ln_f_w, params.ln_f_b)
    return h @ params.token_emb.T

batched_forward = jax.jit(
    jax.vmap(
        forward,
        in_axes=(None,0),
    )
)

# ==========================================
# 5. LOSS & TRAINING STEP
# ==========================================
def loss_fn(params, x, y):
    logits = batched_forward(params, x)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    y_flat = y.reshape(-1)
    
    # one_hot = jax.nn.one_hot(y_flat, logits_flat.shape[-1])
    # loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits_flat, axis=-1), axis=-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_flat,
        y_flat,
    )
    return jnp.mean(loss)

def make_train_step(tx):

    #@partial(jax.jit, static_argnames=("num_heads",))
    @partial(jax.jit, donate_argnums=(0,1),)
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(
            params,
            x,
            y,            
        )

        updates, opt_state = tx.update(
            grads,
            opt_state,
            params,
        )

        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    return train_step


# @partial(jax.jit, static_argnames=("num_heads", "tx"))
# def train_step(params, opt_state, x, y, num_heads, tx):
#     loss, grads = jax.value_and_grad(loss_fn)(params, x, y, num_heads)

#     updates, opt_state = tx.update(
#         grads,
#         opt_state,
#         params,
#     )

#     params = optax.apply_updates(params, updates)

#     return params, opt_state, loss

# def train_step(params, opt_state, x, y, num_heads, tx):
#     loss, grads = jax.value_and_grad(loss_fn)(params, x, y, num_heads)
#     updates, opt_state = tx.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, loss


def train_step_sharded(params, opt_state, x_batch, y_batch, num_heads, tx):
    """Shard a batch across local CPU devices and average the per-device gradients."""
    num_devices = len(jax.local_devices())
    if not Config.use_data_parallelism or num_devices <= 1:
        return train_step(params, opt_state, x_batch, y_batch, num_heads, tx)

    shard_count = min(num_devices, x_batch.shape[0])
    if shard_count <= 1:
        return train_step(params, opt_state, x_batch, y_batch, num_heads, tx)

    per_shard = x_batch.shape[0] // shard_count
    if per_shard == 0:
        return train_step(params, opt_state, x_batch, y_batch, num_heads, tx)

    batch_to_use = per_shard * shard_count
    x_batch = x_batch[:batch_to_use]
    y_batch = y_batch[:batch_to_use]

    x_shards = [x_batch[i * per_shard : (i + 1) * per_shard] for i in range(shard_count)]
    y_shards = [y_batch[i * per_shard : (i + 1) * per_shard] for i in range(shard_count)]

    devices = jax.local_devices()[:shard_count]
    x_sharded = jax.device_put_sharded(x_shards, devices)
    y_sharded = jax.device_put_sharded(y_shards, devices)

    def shard_loss_and_grads(params, x_shard, y_shard, num_heads):
        return jax.value_and_grad(loss_fn)(params, x_shard, y_shard, num_heads)

    pmap_loss_and_grads = jax.pmap(
        shard_loss_and_grads,
        in_axes=(None, 0, 0, None),
        static_broadcasted_argnums=(3,),
    )

    losses, grads = pmap_loss_and_grads(params, x_sharded, y_sharded, num_heads)
    grads = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads)
    loss = jnp.mean(losses)

    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ==========================================
# 6. CLEANER TEXT SAMPLING LOGIC
# ==========================================
@jax.jit(static_argnames=('top_k', 'top_p'))
def apply_sampling(logits, temperature, top_k, top_p):
    """Applies temperature scaling, Top-K filter, and Top-P filter to logits."""
    # 1. Apply Temperature Scaling
    logits = logits / jnp.where(temperature > 0.0, temperature, 1.0)
    
    # Sort logits descending to find threshold positions
    sorted_indices = jnp.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    
    # 2. Apply Top-K Filtering
    if top_k > 0:
        k_mask = jnp.arange(logits.shape[-1]) >= top_k
        sorted_logits = jnp.where(k_mask, -1e9, sorted_logits)
        
    # 3. Apply Top-P (Nucleus) Filtering
    if top_p > 0.0 and top_p < 1.0:
        probs = jax.nn.softmax(sorted_logits, axis=-1)
        cum_probs = jnp.cumsum(probs)
        # Shift mask right by 1 element to ensure we keep the token that exceeds top_p
        p_mask = cum_probs > top_p
        p_mask = jnp.roll(p_mask, 1, axis=-1).at[0].set(False)
        sorted_logits = jnp.where(p_mask, -1e9, sorted_logits)
        
    # Put sorted/filtered values back into original order
    inverse_indices = jnp.argsort(sorted_indices)
    filtered_logits = sorted_logits[inverse_indices]
    return filtered_logits

def generate(params, tokenizer, prompt, max_gen_len=60, temperature=Config.temperature):
    jit_forward = jax.jit(forward)
    tokens = tokenizer.encode(prompt)
    
    for i in range(max_gen_len):
        x = jnp.array(tokens[-Config.max_len:], dtype=jnp.int32)
        raw_logits = jit_forward(params, x,)[-1, :]
        
        # Apply filters to clean predictions
        processed_logits = apply_sampling(
            raw_logits, temperature, Config.top_k, Config.top_p
        )
        
        # Draw final token using a changing random seed
        next_token = jax.random.categorical(jax.random.PRNGKey(42 + i), processed_logits)
        tokens.append(int(next_token))
        
        if next_token == tokenizer.eot_token:
            break
            
    return tokenizer.decode(tokens)



# Include functions to save and load model parameters if needed
def save_model_params(params, file_path):
    """Save model parameters safely by transferring to host and pickling.

    JAX DeviceArrays must be moved to host (numpy) before pickling/saving.
    """
    
    host_params = jax.device_get(params)
    with open(file_path, 'wb') as f:
        pickle.dump(host_params, f)


def load_model_params(file_path):
    """Load model parameters previously saved with `save_model_params`.
    """
    return load_object(file_path)

# Include functions to save and load the tokenizer if needed
def save_tokenizer(tokenizer, file_path):
    """Save the tokenizer object to a file using pickle."""
    save_object(tokenizer, file_path)

def load_tokenizer(file_path):
    """Load the tokenizer object from a file using pickle."""
    return load_object(file_path)

def train_model():
    # Load dataset locally
    data_iter, val_iter, tokenizer = get_data_loader(Config.dataset_path, Config.batch_size, Config.max_len)
    
    # Initialize parameters and optimizer
    print("Initializing weights...")
    init_key = jax.random.PRNGKey(0)
    params = init_params(init_key, Config)
    
    tx = optax.adamw(learning_rate=Config.lr)
    opt_state = tx.init(params)
    train_step = make_train_step(tx)
    
    # Training Loop
    print("Starting training loop on CPU...")
    steps = []
    losses = []
    validation_losses = []
    best_params = jax.tree.map(lambda x: x.copy(), params)
    best_opt_state = opt_state
    best_loss = float("inf")
    patience_counter = 0
    best_step = 0
    start_time = time()

    for step in range(1, Config.steps + 1):
        x_batch, y_batch = next(data_iter)
        
        params, opt_state, loss_val = train_step(params, opt_state, x_batch, y_batch)
               
        loss_val.block_until_ready()
        train_loss = float(loss_val)
        
        steps.append(step)
        losses.append(train_loss)

        if Config.early_stopping and step % Config.early_stop_eval_interval == 0 or step == 1:
            val_x, val_y = next(val_iter)
            
            # val_loss = float(loss_fn(params, val_x, val_y))
            val_loss = loss_fn(params, val_x, val_y)
            val_loss.block_until_ready()
            val_loss = float(val_loss)
            validation_losses.append(val_loss)

            if val_loss < best_loss - Config.early_stop_min_delta:
                best_loss = val_loss
                best_params = jax.tree.map(lambda x: x.copy(), params)
                best_opt_state = opt_state
                best_step = step
                patience_counter = 0
            else:
                patience_counter += 1

            if Config.early_stopping and patience_counter >= Config.early_stop_patience:
                print(
                    f"Early stopping at step {step}: validation loss has not improved for {patience_counter} checks."
                )
                break

        if step % 100 == 0 or step == 1:
            print(f"Step {step}/{Config.steps} | Train Loss: {train_loss:.4f}")

    end_time = time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    loss_results = {
        "steps": steps,
        "train_losses": losses,
        "validation_losses": validation_losses,
        "best_step": best_step,
        "best_validation_loss": best_loss,
    }
    save_object(loss_results, Config.loss_results_path)
    print(f"Training completed in {elapsed_time_seconds:.2f} seconds.")
    print(f"Training completed in {elapsed_time_minutes:.2f} minutes.")
    if best_step > 0:
        params = best_params
        opt_state = best_opt_state
    
    save_model_params(best_params, Config.model_path)
    save_tokenizer(tokenizer, Config.tokenizer_path)

    return params, tokenizer, loss_results

def generate_text_from_prompt(params, tokenizer, prompt, max_gen_len=60, temperature=Config.temperature):
    # Test Story Generation
    start_time = time()    
    print("\nStart inference!\n\nGenerating a sample story:")
    prompt = "Once upon a time, in a warm and sunny place"
    generated_story = generate(params, tokenizer, prompt, max_gen_len=max_gen_len, temperature=temperature)
    print("-" * 40)
    print(generated_story)
    print("-" * 40)
    print("\n\n\n")
    end_time = time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    print(f"Training completed in {elapsed_time_seconds:.2f} seconds.")
    print(f"Training completed in {elapsed_time_minutes:.2f} minutes.")
    

def main(is_train=True):
    if is_train:
        # Train the model
        params, tokenizer, loss_results = train_model()
    else:
        if not os.path.exists(Config.model_path) or not os.path.exists(Config.tokenizer_path):
            raise FileNotFoundError(
                f"Model parameters or tokenizer not found. "
                f"Please ensure '{Config.model_path}' and '{Config.tokenizer_path}' exist."
            )
        params = load_model_params(Config.model_path)
        tokenizer = load_tokenizer(Config.tokenizer_path)
        loss_results = load_object(Config.loss_results_path)

    # Generate text from a prompt
    generate_text_from_prompt(params, tokenizer, "Once upon a time, a little", max_gen_len=60, temperature=0.1)

# ==========================================
# 7. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    main(is_train=True)  # Set to False to skip training and load existing model
    
