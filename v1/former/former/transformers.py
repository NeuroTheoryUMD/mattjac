import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock

from .util import d

# transformer for mapping stimulus (1-D image) stimuli to neural responses
class NTransformer1D(nn.Module):
    """
    Transformer for predicting neural activity.
    (adapted from the CTransformer classifier)
    """

    def __init__(self, stim, heads, depth, seq_length, num_neurons, max_pool=True, dropout=0.0, wide=False):
        """
        :param stim: Stimulus dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_neurons: Number of neurons
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_neurons, self.max_pool = num_neurons, max_pool

        # TODO: do we need to use the position embedding
        #       to tell the model where in time the stimulus is?
        # TODO: can we encode sine and cosine oscillations like in the
        #       original paper?
        self.pos_embedding = nn.Embedding(embedding_dim=stim, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(stimuli=stim, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        # TODO: output num_neurons needs to be shape [10, 100, 11]
        self.torobs = nn.Linear(stim, num_neurons)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of 1D stimuli.
        :return: predicted log-probability vectors for each stimulus based on the preceding tokens.
        """
        
        b, t, e = x.shape
        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        
        
        x = x + positions
        x = self.do(x)

        x = self.tblocks(x)
        
        # TODO: x and the robs output need to be lined up
        #x_view0 = x.view(b*t, e)
        #print('x_view0', x_view0.shape)
        #x_view1 = x_view0.view(b, t, e)
        #print('x_view1', x_view1.shape)

        # target size (torch.Size([10, 100, 11])) 
        # that is different to the input size (torch.Size([10, 100, 1100]))
        
        x = self.torobs(x)

        return F.log_softmax(x, dim=2)


class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, attention_type='default'):

        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=(seq_length * 2 - 1 if attention_type=='relative' else seq_length))

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, attention_type=attention_type, pos_embedding=self.pos_embedding))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(stimuli=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()
        print('b, t, e', b, t, e)

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        print('x', x.shape)
        print('tokens', tokens.shape)
        print('positions', positions.shape)
        x = tokens + positions
        print("x'", x.shape)
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)

