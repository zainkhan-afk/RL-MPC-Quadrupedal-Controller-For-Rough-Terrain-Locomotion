import numpy as np
from utils import sigmoid, scale_footstep_timing, convert_to_list

class GaitCorrection:
    def __init__(self):
        self.prev_values = np.array([0]*9)
        self.current_values = np.array([0]*9)
        self.num_steps = 20
        self.m = self.calculate_gradient()
        self.current_step = 0

    def calculate_gradient(self):
        if self.num_steps == 1:
            return self.current_values - self.prev_values
        return (self.current_values - self.prev_values) / (self.num_steps - 1)

    def step(self):
        if self.current_step < self.num_steps:
            gait_params = self.m * self.current_step + self.prev_values
            self.current_step += 1

            gait_params = scale_footstep_timing(gait_params)
            gait_params = convert_to_list(gait_params)

            # gait_params[0] = gait_params[0]*16.0+4.0  # horizon: 4-20
            # for i in range(8):
            #     gait_params[i+1] = gait_params[i+1]*gait_params[0]  # offset, duration: 0-horizon
            
            

            # if(type(gait_params) == type(np.array([1]))):
            #     gait_params = gait_params.tolist()
            # if(type(gait_params[0]) == np.float64):
            #     gait_params = [gait_params[j].item() for j in range(9)]
            # gait_params = [round(gait_params[j]) for j in range(9)]

            
            return gait_params
        return None

    def reset(self):
        self.prev_values = np.array([0]*9)
        self.current_values = np.array([0]*9)
        self.m = self.calculate_gradient()
        self.current_step = 0

    def set_gait_params(self, gait_params):
        self.prev_values = self.current_values
        self.current_values = gait_params

        # num_stps = self.prev_values[0]*16 + 4
        # print(round(num_stps), end = " ")
        # for i in range(8):
        #     print(round(self.prev_values[i+1] * num_stps), end = " ")
        
        # print(" | ", end = "")
        
        # num_stps = self.current_values[0]*16 + 4
        # print(round(num_stps), end = " ")
        # for i in range(8):
        #     print(round(self.current_values[i+1] * num_stps), end = " ")
        # print()

        
        self.m = self.calculate_gradient()
        
        self.current_step = 0

if __name__ == "__main__":
    gc = GaitCorrection()
    act = np.array([20,20,20,20,20,20,20,20,20]) # stand
    gc.set_gait_params(act)

    act = np.array([14,0,7,7,0,7,7,7,7]) # trot14
    gc.set_gait_params(act)

    while True:
        gait_params = gc.step()
        if gait_params is None:
            break
        print("During:", gait_params)

    act = np.array([10,0,2,7,9,5,5,5,5]) # gallop10
    gc.set_gait_params(act)

    while True:
        gait_params = gc.step()
        if gait_params is None:
            break
        print("During:", gait_params)
