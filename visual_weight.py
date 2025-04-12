import argparse
import pickle
import matplotlib.pyplot as plt

def visual(W, name):
    plt.imshow(W * 10, cmap='seismic', vmin=-0.5, vmax=0.5)
    plt.title(name)
    plt.axis('off')
    plt.show()

# load model
parser = argparse.ArgumentParser()

parser.add_argument('--model_path', '-p', type=str, default='./saved_models/best_model/models/best_model.pickle')
args = parser.parse_args()

model_path = args.model_path

with open(model_path, 'rb') as f:
    param_list = pickle.load(f, encoding='bytes')

for idx, layer in enumerate(param_list[2:]):
    name = f'Weight of the {idx+1} Layer'
    visual(layer['W'], name)

