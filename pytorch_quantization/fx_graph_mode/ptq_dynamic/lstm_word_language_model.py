# imports
import os
from io import open
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Definition
class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden


def init_hidden(lstm_model, bsz):
    # get the weight tensor and create hidden layer in the same device
    weight = lstm_model.encoder.weight
    # get weight from quantized model
    if not isinstance(weight, torch.Tensor):
        weight = weight()
    device = weight.device
    nlayers = lstm_model.rnn.num_layers
    nhid = lstm_model.rnn.hidden_size
    return (torch.zeros(nlayers, bsz, nhid, device=device),
            torch.zeros(nlayers, bsz, nhid, device=device))


# Load Text Data
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

model_data_filepath = 'data/'

corpus = Corpus(model_data_filepath + 'wikitext-2')

ntokens = len(corpus.dictionary)

# Load Pretrained Model
model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

model.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu'),
        weights_only=True
        )
    )

model.eval()
print(model)

bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

# create test data set
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    return data.view(bsz, -1).t().contiguous()

test_data = batchify(corpus.test, eval_batch_size)
example_inputs = (next(iter(test_data))[0])

# Evaluation functions
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history."""

  if isinstance(h, torch.Tensor):
      return h.detach()
  else:
      return tuple(repackage_hidden(v) for v in h)

def evaluate(model_, data_source):
    print('test data size: ', data_source.size(0))
    # Turn on evaluation mode which disables dropout.
    model_.eval()
    total_loss = 0.
    hidden = init_hidden(model_, eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model_(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)



# Now we can dynamically quantize the model. We can use the same function as post training static quantization but with a dynamic qconfig.
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, QConfigMapping

# Full docs for supported qconfig for floating point modules/ops can be found in 
# `quantization docs <https://pytorch.org/docs/stable/quantization.html#module-torch.quantization>`_
# Full docs for `QConfigMapping <https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping>`_
qconfig_mapping = (QConfigMapping()
    .set_object_type(nn.Embedding, float_qparams_weight_only_qconfig)
    .set_object_type(nn.LSTM, default_dynamic_qconfig)
    .set_object_type(nn.Linear, default_dynamic_qconfig)
)
# Load model to create the original model because quantization api changes the model inplace and we want
# to keep the original model for future comparison


model_to_quantize = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

model_to_quantize.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu')
        )
    )

model_to_quantize.eval()

# For dynamically quantized objects, we didn’t do anything in prepare_fx for modules, 
# but will insert observers for weight for dynamically quantizable forunctionals and torch ops. 
# We also fuse the modules like Conv + Bn, Linear + ReLU.

# In convert we’ll convert the float modules to dynamically quantized modules 
# and convert float ops to dynamically quantized ops.
# We can see in the example model, nn.Embedding, nn.Linear and nn.LSTM are dynamically quantized.

prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
print("prepared model:", prepared_model)
quantized_model = convert_fx(prepared_model)
print("quantized model", quantized_model)

# compare the size and runtime of the quantized model.
# There is a 4x size reduction because we quantized all the weights in the model (nn.Embedding, nn.Linear and nn.LSTM) 
# from float (4 bytes) to quantized int (1 byte).
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)


# There is a roughly 2x speedup for this model. Also note that the speedup may vary depending on model, 
# device, build, input batch sizes, threading etc.
torch.set_num_threads(1)

def time_model_evaluation(model, test_data):
    s = time.time()
    loss = evaluate(model, test_data)
    elapsed = time.time() - s
    print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

time_model_evaluation(model, test_data)
time_model_evaluation(quantized_model, test_data)



################################################## output ##################################################
# (base) root@autodl-container-9db611b252-b759e1c0:~/autodl-tmp/workspace/infer_system/pytorch_quantization/fx_graph_mode/ptq_dynamic# python3 lstm_word_language_model.py 
# /root/miniconda3/lib/python3.8/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
#   return self.fget.__get__(instance, owner)()
# LSTMModel(
#   (drop): Dropout(p=0.5, inplace=False)
#   (encoder): Embedding(33278, 512)
#   (rnn): LSTM(512, 256, num_layers=5, dropout=0.5)
#   (decoder): Linear(in_features=256, out_features=33278, bias=True)
# )
# prepared model: GraphModule(
#   (encoder): Embedding(33278, 512)
#   (drop): Dropout(p=0.5, inplace=False)
#   (activation_post_process_0): PlaceholderObserver()
#   (rnn): LSTM(512, 256, num_layers=5, dropout=0.5)
#   (activation_post_process_1): PlaceholderObserver()
#   (decoder): Linear(in_features=256, out_features=33278, bias=True)
# )



# def forward(self, input, hidden):
#     input_1 = input
#     encoder = self.encoder(input_1);  input_1 = None
#     drop = self.drop(encoder);  encoder = None
#     activation_post_process_0 = self.activation_post_process_0(drop);  drop = None
#     rnn = self.rnn(activation_post_process_0, hidden);  activation_post_process_0 = hidden = None
#     getitem = rnn[0]
#     getitem_1 = rnn[1];  rnn = None
#     drop_1 = self.drop(getitem);  getitem = None
#     activation_post_process_1 = self.activation_post_process_1(drop_1);  drop_1 = None
#     decoder = self.decoder(activation_post_process_1);  activation_post_process_1 = None
#     return (decoder, getitem_1)
    
# # To see more debug info, please use `graph_module.print_readable()`
# /root/miniconda3/lib/python3.8/site-packages/torch/ao/nn/quantized/reference/modules/rnn.py:320: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
#   torch.tensor(weight_qparams["scale"], dtype=torch.float, device=device))
# /root/miniconda3/lib/python3.8/site-packages/torch/ao/nn/quantized/reference/modules/rnn.py:323: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
#   torch.tensor(weight_qparams["zero_point"], dtype=torch.int, device=device))
# quantized model GraphModule(
#   (encoder): QuantizedEmbedding(num_embeddings=33278, embedding_dim=512, dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams)
#   (drop): Dropout(p=0.5, inplace=False)
#   (rnn): DynamicQuantizedLSTM(512, 256, num_layers=5, dropout=0.5)
#   (decoder): DynamicQuantizedLinear(in_features=256, out_features=33278, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
# )



# def forward(self, input, hidden):
#     input_1 = input
#     encoder = self.encoder(input_1);  input_1 = None
#     drop = self.drop(encoder);  encoder = None
#     rnn = self.rnn(drop, hidden);  drop = hidden = None
#     getitem = rnn[0]
#     getitem_1 = rnn[1];  rnn = None
#     drop_1 = self.drop(getitem);  getitem = None
#     decoder = self.decoder(drop_1);  drop_1 = None
#     return (decoder, getitem_1)
    
# # To see more debug info, please use `graph_module.print_readable()`
# Size (MB): 113.943637
# Size (MB): 28.889917
# loss: 5.167
# elapsed time (seconds): 197.1
# loss: 5.167
# elapsed time (seconds): 84.3


