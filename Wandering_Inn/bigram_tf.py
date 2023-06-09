import tensorflow as tf
import tensorflow_probability as tfp

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200

# with open('./files/input.txt', 'r', encoding='utf-8') as f:
with open('./files/preprocessed_wandering_inn.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
print(chars)
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = tf.convert_to_tensor(encode(text), dtype=tf.float32)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = tf.random.uniform((batch_size,), 0, len(data) - block_size, dtype=tf.dtypes.int32, seed=1337)
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out

# super simple bigram model
class BigramLanguageModel(tf.keras.Model):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, vocab_size)

    def call(self, idx):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1) # (B, C)
            # sample from the distribution
            sampler = tfp.distributions.Sample(probs, sample_shape=1) # (B, 1)
            idx_next = sampler.sample()
            # append sampled index to the running sequence
            idx = tf.concat((idx, idx_next), axis=1) # (B, T+1)
        return idx

loss_metric = tf.keras.losses.SparseCategoricalCrossentropy()

# class Perplexity(tf.keras.losses.SparseCategoricalCrossentropy):
#     def __init__(self, *args, name='perplexity', **kwargs):
#         super().__init__(*args, name=name, **kwargs)
    
#     def call(self, *args, **kwargs):
#         ## TODO: Implement perplexity (hint: use the superclass)
#         scce = super().call(*args, **kwargs)
#         return tf.math.exp(tf.reduce_mean(scce))

model = BigramLanguageModel(vocab_size) 
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(0.005), 
    loss=loss_metric, 
    # metrics=[Perplexity()],
)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    xb, yb = get_batch('train')
    xv, yv = get_batch('val')
    if iter % eval_interval == 0:
        model.fit(xb, yb, epochs=1, validation_data=(xv,yv), verbose=1)
        # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    model.fit(xb, yb, epochs=1, validation_data=(xv,yv), verbose=1)