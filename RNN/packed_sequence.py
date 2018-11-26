import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# Random word from random word generator
data = ['hello world',
	    'midnight',
	    'calculation',
	    'path',
        'short circuit']

# Make dictionary
char_set = list(set(char for seq in data for char in seq)) # Get all characters
char2idx = {c: i for i, c in enumerate(char_set)} # Constuct character to index dictionary
print(char_set)
print(len(char_set))

# Convert character to index and
# Make list of tensors
X = [torch.LongTensor([char2idx[char] for char in seq]) for seq in data]

# Check converted result
for sequence in X:
  print(sequence.size(), sequence)
  
# Make length tensor 
lengths = torch.LongTensor([len(seq) for seq in X])
print(lengths)
  
# Make a Tensor of shape (Batch x Maximum_Sequence_Length)
X = pad_sequence(X, batch_first=True) # X is now padded sequence
print(X)
print(X.shape)

# Sort by descending lengths
lengths, sorted_idx = torch.sort(lengths, descending=True)
X = X[sorted_idx]
print(sorted_idx)
print(X)

# Convert padded Tensor to packed sequence
packed_X = pack_padded_sequence(X, lengths, batch_first=True)# X is now packed sequence
print(packed_X)

# declare RNN
rnn = torch.nn.RNN(len(char_set), 30, batch_first=True)
rnn(packed_X)
