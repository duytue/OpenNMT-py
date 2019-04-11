"""Define RNN-based encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory
from onmt.modules import context_gate_factory, GlobalAttention

class CGURNNEncoder(EncoderBase):

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(CGURNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        # ================================================
        self.bidirectional = bidirectional
        self.selfatt = True

        # ================================================
        # The paper use True
        self.swish = True
        if self.swish:
            self.sw1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding=0), nn.BatchNorm1d(hidden_size), nn.ReLU())
            self.sw3 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(hidden_size),
                                     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(hidden_size))
            self.sw33 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(hidden_size),
                                      nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(hidden_size),
                                      nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(hidden_size))
            self.linear = nn.Sequential(nn.Linear(2*hidden_size, 2*hidden_size), nn.GLU(), nn.Dropout(dropout))
            self.filter_linear = nn.Linear(3*hidden_size, hidden_size)
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()

        # ================================================
        # Coverage penalty is proved to be as equally good as coverage attention,
        # So we don't use it here
        coverage_attn=False
        # The attention type to use: 
        # "dotprod or general (Luong) or MLP (Bahdanau)"
        attn_type = "general"
        # ["softmax", "sparsemax"]
        attn_func = "softmax"

        self.attn = GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type, attn_func=attn_func
            )

        # if config.selfatt:
        #     if config.attention == 'None':
        #         self.attention = None
        #     elif config.attention == 'bahdanau':
        #         self.attention = models.bahdanau_attention(hidden_size, config.emb_size, config.pool_size)
        #     elif config.attention == 'luong':
        #         self.attention = models.luong_attention(hidden_size, config.emb_size, config.pool_size)
        #     elif config.attention == 'luong_gate':
        #         self.attention = models.luong_gate_attention(hidden_size, config.emb_size)

        # ================================================
        # if config.cell == 'gru':
        #     self.rnn = nn.GRU(input_size=embeddings.embedding_size, hidden_size=hidden_size,
        #                       num_layers=config.enc_num_layers, dropout=config.dropout,
        #                       bidirectional=config.bidirectional)
        # else:
        #     self.rnn = nn.LSTM(input_size=embeddings.embedding_size, hidden_size=hidden_size,
        #                        num_layers=config.enc_num_layers, dropout=config.dropout,
        #                        bidirectional=config.bidirectional)

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout,
            embeddings,
            opt.bridge)

    def forward(self, src, lengths=None):
        self._check_args(src, lengths)

        # ================================================
        # Handle embeddings

        # embs = pack(self.embedding(inputs), lengths)
        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)
        
        # ================================================
        # Forward RNN

        # outputs, state = self.rnn(embs)
        memory_bank, encoder_final = self.rnn(packed_emb)

        # ================================================
        # Unpack output

        # outputs = unpack(outputs)[0]
        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        # ================================================
        # CNN bottleneck

        if self.bidirectional:
            if self.swish:
                memory_bank = self.linear(memory_bank)
            else:
                memory_bank = memory_bank[:,:,:self.hidden_size] + memory_bank[:,:,self.hidden_size:]

        if self.swish:
            memory_bank = memory_bank.transpose(0,1).transpose(1,2)
            conv1 = self.sw1(memory_bank)
            conv3 = self.sw3(memory_bank)
            conv33 = self.sw33(memory_bank)
            conv = torch.cat((conv1, conv3, conv33), 1)
            conv = self.filter_linear(conv.transpose(1,2))
            if self.selfatt:
                conv = conv.transpose(0,1)
                memory_bank = memory_bank.transpose(1,2).transpose(0,1)
            else:
                gate = self.sigmoid(conv)
                memory_bank = memory_bank * gate.transpose(1,2)
                memory_bank = memory_bank.transpose(1,2).transpose(0,1)

        # ================================================
        # Attention

        # context = memory_bank.tranpose(0, 1)
        # memory_bank, encoder_final = self.attn(
        #         context.contiguous(),
        #         memory_bank.transpose(0, 1),
        #         memory_lengths=lengths
        # )

        # if self.selfatt:
        #     self.attention.init_context(context=conv)
        #     out_attn, weights = self.attention(conv, selfatt=True)
        #     gate = self.sigmoid(out_attn)
        #     memory_bank = memory_bank * gate

        # if self.config.cell == 'gru':
        #     state = state[:self.config.dec_num_layers]
        # else:
        #     state = (state[0][::2], state[1][::2])

        # ================================================
        # Bridge

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

        return encoder_final, memory_bank, lengths

        # return outputs, state

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout,
            embeddings,
            opt.bridge)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
