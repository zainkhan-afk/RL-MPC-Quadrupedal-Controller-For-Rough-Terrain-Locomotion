import numpy as np

def sigmoid(z):
    return 1 / (1+np.exp(-z))


def scale_footstep_timing(act):
    act[0] = act[0]*16.0+4.0  # horizon: 4-20
    for i in range(8):
        act[i+1] = act[i+1]*act[0]  # offset, duration: 0-horizon

    return act

def convert_to_list(act):
    if(type(act) == type(np.array([1]))):
        act = act.tolist()
    if(type(act[0]) == np.float64):
        act = [act[j].item() for j in range(9)]
    act = [round(act[j]) for j in range(9)]

    return act