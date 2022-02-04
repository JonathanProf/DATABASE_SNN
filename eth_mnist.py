import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.network import *
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)

"""
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=False)

args = parser.parse_args()
"""

seed = int(0)
n_neurons = int(400)
n_epochs = int(1)
n_test = int(10000)
n_train = int(60000)
n_workers = int(-1)
exc = float(22.5)
inh = float(120)
theta_plus = float(0.05)
time = int(64)
dt = int(1.0)
intensity = float(128)
progress_interval = int(10)
update_interval = int(250)
train = False
plot = True
gpu = False

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

#network = network.load('./network400.pt')

PATH_NET_PARAM = "./SpikingNeuralNetwork/BD/window" + str(time) + "ms/BD" + str(n_neurons) + "_" + str(time) + "ms/"

# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Build network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

network.load_state_dict( torch.load( PATH_NET_PARAM + 'network' + str(n_neurons) + 'N_' + str(time) + 'ms_CPU_.pt' , map_location="cpu") )
# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt)
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt)
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = torch.load( PATH_NET_PARAM + 'assignments' + str(n_neurons) + 'N_' + str(time) + 'ms_CPU_.pt',"cpu")
proportions = torch.load( PATH_NET_PARAM + 'proportions' + str(n_neurons) + 'N_' + str(time) + 'ms_CPU_.pt',"cpu")

'''
inputDataToFile = network.X_to_Ae.w.numpy()
filenameData = './SpikingNeuralNetwork/BD/BD400/XeAe.csv'
inputDataToFile.tofile( filenameData , sep = ',' )

inputDataToFile = network.Ae.theta.numpy()
filenameData = './SpikingNeuralNetwork/BD/BD400/theta.csv'
inputDataToFile.tofile( filenameData , sep = ',' )

inputDataToFile = assignments.numpy()

filenameData = './SpikingNeuralNetwork/BD/BD400/assignments.csv'
inputDataToFile.tofile( filenameData , sep = ',' )

inputDataToFile = proportions.numpy()
filenameData = './SpikingNeuralNetwork/BD/BD400/proportions.csv'
inputDataToFile.tofile( filenameData , sep = ',' )
'''
# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

epoch = 1

contSample = int(0)

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    
    
    inputNetwork = inputs['X'][:,0,0,:,:].numpy()
    filename_InSamp = './SpikingNeuralNetwork/BD/inputSamples/{0:05d}_inputSpikesPoisson'.format(step+1)+'.csv'
    inputNetwork.tofile( filename_InSamp , sep = ',' )
    
    
    '''
    print('{0:=>30}'.format(''))
    print("\nstep:" , step+1, " sample: ", batch['encoded_label'])
    print('{0:=>30}'.format(''))
    '''
    network.run(inputs=inputs, time=time, input_time_dim=1)
    
    '''
    filename_Index = './SpikingNeuralNetwork/BD/inputSamples/vectorIndexresultsBindsnet.csv'
    with open( filename_Index , 'a', encoding='utf-8' ) as f:
        f.write( '\n' )
    '''
    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()
    '''
    print("")
    print(torch.nonzero( spike_record[0] ))
    '''
    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)
    
    
    with open( './SpikingNeuralNetwork/BD/classification/labelsBindsnetIn_' + str(n_neurons) + 'N_' + str(time) + 'ms_CPU_DELL.csv', 'a', encoding='utf-8' ) as f:
        f.write( str( int(label_tensor) ) +'\n' )
    
    
    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )
    
    
    with open( './SpikingNeuralNetwork/BD/classification/labelsBindsnetOut_' + str(n_neurons) + 'N_' + str(time) + 'ms_CPU_DELL.csv', 'a', encoding='utf-8' ) as f:
        f.write( str( int(all_activity_pred) ) +'\n' )
    
    
    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))


print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")