import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "depth":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(
                                        nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Flatten()
                                    )
                
                # extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += 288
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 64), 
                                                nn.ReLU())
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            features = extractor(observations[key])
            encoded_tensor_list.append(features)
        
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        combined = th.cat(encoded_tensor_list, dim=1)
        return combined

if __name__ == "__main__":
    import numpy as np 
    depth_space = gym.spaces.Box(0.0, 1.0, shape=(1, 40, 40), dtype='float32')
    vector_space = gym.spaces.Box(-np.inf, np.inf, shape=(16, ), dtype='float32')
        
    observation_space = gym.spaces.Dict({"depth": depth_space, "vector": vector_space})
    extractor = CustomCombinedExtractor(observation_space)


    ip = {"depth": th.zeros((1,1,40,40)), 
    "vector": th.zeros((1,16))}
    extractor(ip)