# RL-MPC-Quadrupedal-Controller-For-Rough-Terrain-Locomotion

The MPC controller is based on the paper titled [Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control](https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf) and it's implementation is taken from [here](https://github.com/Derek-TH-Wang/quadruped_ctrl).

## Running the code

1. First clone the repository:

    ```
    git clone 
    ```


2. Compile the MPC controller using:

    ```
    cd quadruped_ctrl/
    mkdir build
    cd build
    cmake ..
    make -j 4
    ```
    This creates a `.so` file inside the build directory named `libquadruped_ctrl.so`.

3. In order to train the robot or to run robot on a pretrained model, run the `main.py` file.
    ```
    python main.py
    ```

## Configs

There are two different configs that determine the behavior of the simulation. The first one is `config/training_params.yaml`. This file contains the training parameters.

The second config file is `quadruped_ctrl/config/quadruped_ctrl_config.yaml`. This file contains all the simulation related configurations. The values of interest in this config are: `visualization` and `terrain`.

The `visualization` should be set to `False` when the display needs to be disabled. This is useful during training.

The `terrain` option specifies the terrain the robot will walk on. The two options are `plane` and `hybrid`.

