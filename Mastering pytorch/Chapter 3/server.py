import os
import json
import numpy as np
from flask import Flask, request
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

class CNNModel(nn.Module):
    def __init__(self, embedding_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNNModel, self).__init__()
        resnet = models.resnet152(weights=True)
        modules_list = list(resnet.children())[:-1]
        self.resnet_module = nn.Sequential(*modules_list)
        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)
    
    def forward(self, input_images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            resnet_features = self.resnet_module(input_images)
            resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
            final_features = self.batch_norm(self.linear_layer(resnet_features))
        return final_features
        

        


class LSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20):
        """Set the hyper-parameters and build the layers."""
        super(LSTMModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_features, capts, lens):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embedding_layer(capts)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        lstm_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        hidden_variables, _ = self.lstm_layer(lstm_input)
        model_outputs = self.linear_layer(hidden_variables[0])
        return model_outputs
    
    def sample(self, input_features, lstm_states=None):
        """Generate captions for given image features using greedy search."""
        sampled_indices = []
        lstm_inputs = input_features.unsqueeze(1)
        for i in range(self.max_seq_len):
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs, lstm_states)          # hiddens: (batch_size, 1, hidden_size)
            model_outputs = self.linear_layer(hidden_variables.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted_outputs = model_outputs.max(1)                        # predicted: (batch_size)
            sampled_indices.append(predicted_outputs)
            lstm_inputs = self.embedding_layer(predicted_outputs)                       # inputs: (batch_size, embed_size)
            lstm_inputs = lstm_inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_indices = torch.stack(sampled_indices, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_indices

class Vocab(object):
    """Simple vocabulary wrapper"""
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0
    
    def __call__(self, token):
        if not token in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]
    def __len__(self):
        return len(self.w2i)
    def add_token(self, token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1
with open('./data_dir/vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

encoder = CNNModel(256)
decoder = LSTMModel(256, 512, len(vocabulary), 1)

ENCODER_PATH_TO_MODEL = './models_dir/encoder-2-3000.ckpt'
DECODER_PATH_TO_MODEL = './models_dir/decoder-2-3000.ckpt'
encoder.load_state_dict(torch.load(ENCODER_PATH_TO_MODEL, map_location='cpu'))
decoder.load_state_dict(torch.load(DECODER_PATH_TO_MODEL, map_location='cpu'))
encoder.eval()
decoder.eval()



def run_model(input_tensor):
    
    with torch.no_grad():
        features = encoder(input_tensor.unsqueeze(0))
        sampled_indices = decoder.sample(features)
        sampled_indices = sampled_indices[0].cpu().numpy() 
    return sampled_indices

def get_predicted_sentence(indices):
    predicted_caption = []
    for token_index in indices:
        word = vocabulary.i2w[token_index]
        predicted_caption.append(word)
        if word == '<end>':
            break
    predicted_sentence = ' '.join(predicted_caption)
    return str(predicted_sentence)

app = Flask(__name__)

@app.route('/image_caption', methods=['POST'])
def image_caption():
    data = request.files['data'].read()
    md = json.load(request.files['metadata'])
    input_array = np.frombuffer(data, dtype=np.float32)
    input_image_tensor = torch.from_numpy(input_array).view(md['dims'])
    output = run_model(input_image_tensor)
    final_output = get_predicted_sentence(output)
    return final_output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8885)