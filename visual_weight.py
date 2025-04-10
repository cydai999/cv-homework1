import pickle
import matplotlib.pyplot as plt

def visual(W, name):
    plt.imshow(W)
    plt.title(name)
    plt.axis('off')
    plt.show()

model_path = './saved_models/2025-04-10-22-26/models/best_model.pickle'

with open(model_path, 'rb') as f:
    param_list = pickle.load(f, encoding='bytes')

for idx, layer in enumerate(param_list[2:]):
    name = f'Weight of the {idx+1} Layer'
    visual(layer['W'], name)

