import time
import yaml
import random
import ctypes
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from copy import deepcopy
from gait_correction import GaitCorrection
import os
from utils import sigmoid, scale_footstep_timing, convert_to_list


# if  40 < i < 50 or 60 < i < 70 or 80 < i < 90 or 100 < i < 120 \
#     or 130 < i < 150 or 160 < i < 170 or 190 < i < 210 \
#     or 220 < i < 230 or 240 < i < 250 or 270 < i < 290 \
#     or 300 < i < 310 or 325 < i < 340 or 355 < i < 365 \
#     or 380 < i < 390 or 410 < i < 420 or 430 < i < 440 \
#     or 450 < i < 460 or 470 < i < 480 or 490 < i < 500 \
#     or 510 < i < 520 or 530 < i < 540 or 560 < i < 660:
#     height = random.uniform(0, heightPerturbationRange)
# else:
#     height = 0

def convert_type(input):
    ctypes_map = {int: ctypes.c_int,
                  float: ctypes.c_double,
                  str: ctypes.c_char_p
                  }
    input_type = type(input)
    if input_type is list:
        length = len(input)
        if length == 0:
            print("convert type failed...input is "+input)
            return 0
        else:
            arr = (ctypes_map[type(input[0])] * length)()
            for i in range(length):
                arr[i] = bytes(
                    input[i], encoding="utf-8") if (type(input[0]) is str) else input[i]
            return arr
    else:
        if input_type in ctypes_map:
            return ctypes_map[input_type](bytes(input, encoding="utf-8") if type(input) is str else input)
        else:
            print("convert type failed...input is "+input)
            return 0


class StructPointer(ctypes.Structure):
    _fields_ = [("eff", ctypes.c_double * 12)]


class HybridGaitRobot(object):

    def __init__(self, action_repeat=1000):
        self.action_repeat = action_repeat
        self.gait_correction = GaitCorrection()

        self.stand_act = np.array([1]*9) # stand
        self.gait_correction.set_gait_params(self.stand_act)

        self.ground = 0
        self.quadruped = 0
        self.sim_gravity = [0.0, 0.0, -9.8]

        self.init_pos = [0, 0, 0.3]
        self.motor_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.tau = [0]*12
        self.imu_data = [0]*10
        self.leg_data = [0]*24
        
        self.target_base_vel = [0]*3
        self.target_current_vel = [0]*3
        self.prev_target_base_vel = [0]*3
        self.target_vel_m = [0]*3
        self.num_vel_transition_steps = 20
        self.curr_vel_transition_step = 0
        
        self.base_vel = [0]*3
        self.base_position = self.init_pos
        self._last_base_pos = self.init_pos
        self._robot_dist = 0
        self.get_last_vel = [0]*3
        self._last_time = 0

        with open('quadruped_ctrl/config/quadruped_ctrl_config.yaml') as f:
            quadruped_param = yaml.safe_load(f)
            params = quadruped_param['simulation']

        self.visualization = params['visualization']
        self.terrain = params['terrain']
        self.lateralFriction = params['lateralFriction']
        self.spinningFriction = params['spinningFriction']
        self.freq = params['freq']
        self.stand_kp = params['stand_kp']
        self.stand_kd = params['stand_kd']
        self.joint_kp = params['joint_kp']
        self.joint_kd = params['joint_kd']
        self.evaluating = params["evaluating"]

        self.gait_corr = params["gait_corr"]


        so_file = 'quadruped_ctrl/build/libquadruped_ctrl.so'
        self.cpp_gait_ctrller = ctypes.cdll.LoadLibrary(so_file)
        self.cpp_gait_ctrller.get_prf_foot_coor.restype = ctypes.c_double
        self.cpp_gait_ctrller.toque_calculator.restype = ctypes.POINTER(
            StructPointer)

        self.contact_dict = {2:"fr", 11:"hl", 5:"fl", 8:"hr"}
        self.contact_timings = {"fr":[], "hl":[], "fl":[], "hr":[]}
        self.euler_rotation = [0, 0, 0]
        self.velocity_list = []
        self.x_dist_list = []
        self.rpy_list = []

        self.init_simulator()

    def make_stairs(self):
        planeShape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
        self.ground = p.createMultiBody(0, planeShape)
        # p.resetBasePositionAndOrientation(self.ground, [0, 0, 0], [0, 0.0872, 0, 0.9962])
        p.resetBasePositionAndOrientation(
            self.ground, [0, 0, 0], [0, 0, 0, 1])
        
        box_ids = []
        num_boxes = 10
        max_step_height = 0.1
        single_step_size = max_step_height / (num_boxes * 0.5)
        # many box
        for i in range(num_boxes):
            if i < num_boxes//2:
                box_ids.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.09, 1.0, (i+1)*single_step_size]))
            else:
                box_ids.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.09, 1.0, (num_boxes * 0.5 * single_step_size) - (i - num_boxes//2)*single_step_size]))
            p.createMultiBody(1000, box_ids[-1], basePosition=[1.0 + i*0.2, 0.0, 0.0])
            p.changeDynamics(box_ids[-1], -1, lateralFriction=self.lateralFriction)

    def init_simulator(self):
        if self.visualization:
            p.connect(p.GUI, options = "--mp4=output/movie.mp4 --mp4fps=60")  # or p.DIRECT for non-graphical version
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(self.sim_gravity[0],
                     self.sim_gravity[1], self.sim_gravity[2])
        p.setTimeStep(1.0/self.freq)
        p.resetDebugVisualizerCamera(0.01, 45, -30, [1, -1, 1])

        heightPerturbationRange = 0.06
        numHeightfieldRows = 1024
        numHeightfieldColumns = 128
        if self.terrain == "plane":
            planeShape = p.createCollisionShape(shapeType=p.GEOM_PLANE)
            self.ground = p.createMultiBody(0, planeShape)
            p.resetBasePositionAndOrientation(
                self.ground, [0, 0, 0], [0, 0, 0, 1])

        elif self.terrain == "hybrid":
            heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns
            perturb = False
            perturb_from = 0
            perturb_to = 0
            perturb_anchor = 0
            for i in range(int(numHeightfieldRows/2)):
                for j in range(int(numHeightfieldColumns/2)):
                    if i > 20  and i%20 == 0 and not perturb:
                        chance = random.random()

                        if chance > 0.7:
                            perturb = True
                            perturb_from = random.randint(3, 10)
                            perturb_to = random.randint(3, 10)
                            perturb_anchor = i + random.randint(0, 5)

                    if perturb:
                        if perturb_anchor-perturb_from < i < perturb_anchor+perturb_to:
                            height = random.uniform(0, heightPerturbationRange)

                        if i >= perturb_anchor + perturb_to:
                            perturb = False

                    else:
                        height = 0
                    
                    heightfieldData[2*i+2*j*numHeightfieldRows] = height
                    heightfieldData[2*i+1+2*j*numHeightfieldRows] = height
                    heightfieldData[2*i+(2*j+1)*numHeightfieldRows] = height
                    heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows] = height
            # terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.1, .1, 1], heightfieldTextureScaling=(
            #     numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
            
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.1, .1, 1], heightfieldData=heightfieldData, 
                                                  numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
            self.ground = p.createMultiBody(0, terrainShape)
            # textureId = p.loadTexture("checker_grid.jpg")


            tex_data_gray = np.array(heightfieldData)

            tex_data_gray = tex_data_gray.reshape((numHeightfieldColumns, numHeightfieldRows))

            tex_data_gray = tex_data_gray / tex_data_gray.max() * 255

            tex_data_color = np.zeros((tex_data_gray.shape[0], tex_data_gray.shape[1], 4))

            tex_data_color = tex_data_color.astype("uint8")

            tex_data_color[:, :, 0] = 0
            tex_data_color[:, :, 1] = 100
            tex_data_color[:, :, 2] = 0
            tex_data_color[:, :, 3] = 255

            tex_data_color[tex_data_gray > 0, 0] = 100
            tex_data_color[tex_data_gray > 0, 1] = 100
            tex_data_color[tex_data_gray > 0, 2] = 100
            tex_data_color[tex_data_gray > 0, 3] = 255


            # tex_data_color[:, :, 0] = tex_data_gray

            tex_data_color = cv2.flip(tex_data_color, cv2.ROTATE_90_COUNTERCLOCKWISE)

            H, W, _ = tex_data_color.shape

            tex_data_color = cv2.resize(tex_data_color, (int(W*2), int(H*2)))

            cv2.imwrite("land_tex.png", tex_data_color)

            textureId = p.loadTexture("land_tex.png")
            p.changeVisualShape(self.ground, -1, textureUniqueId=textureId)

            p.resetBasePositionAndOrientation(
                self.ground, [45, 0, 0], [0, 0, 0, 1])
        

        p.changeDynamics(self.ground, -1, lateralFriction=self.lateralFriction)
        self.quadruped = p.loadURDF("hybrid_gait/quadruped_ctrl/model//mini_cheetah.urdf", self.init_pos,
                                    useFixedBase=False)
        p.changeDynamics(self.quadruped, 3, lateralFriction=self.lateralFriction, 
                         spinningFriction=self.spinningFriction)
        p.changeDynamics(self.quadruped, 7, lateralFriction=self.lateralFriction, 
                         spinningFriction=self.spinningFriction)
        p.changeDynamics(self.quadruped, 11, lateralFriction=self.lateralFriction, 
                         spinningFriction=self.spinningFriction)
        p.changeDynamics(self.quadruped, 15, lateralFriction=self.lateralFriction, 
                         spinningFriction=self.spinningFriction)
        jointIds = []
        for j in range(p.getNumJoints(self.quadruped)):
            p.getJointInfo(self.quadruped, j)
            jointIds.append(j)

    def get_camera_image(self):
        positions_from = []
        positions_to = []

        W = 40
        H = 40

        terrain_W = 2
        terrain_H = 2

        scale_x = terrain_W / W
        scale_y = terrain_H / H

        x = -W//2
        y = -H//2
        for i in range(1600):
            x += 1
            if (x >= W//2):
                y += 1
                x = -W//2
            positions_from.append(np.array([self.base_position[0] + x * scale_x, 
                                            self.base_position[1] + y * scale_y, self.base_position[2] - 0.02]))
            positions_to.append(np.array([self.base_position[0] + x * scale_x, 
                                          self.base_position[1] + y * scale_y, -10]))



        img = np.zeros((1, 40, 40)).astype("float64")
        height_map = p.rayTestBatch(positions_from, positions_to)

        for i in range(len(height_map)):
            x = i%W
            y = i // W
            z_val = height_map[i][3][2]
            # print(height_map[i][3])
            img[0, y, x] = z_val

        # print(img.min(), img.max())
        img = (img.max() - img) / (img.max() - img.min())
        # img = img.astype("uint8")

        # img = cv2.resize(img, (400, 400))
        # cv2.imshow("img", cv2.resize(img.reshape((40,40)), (200, 200)))
        # cv2.waitKey(1)

        return img


    def step_desired_velocity(self):
        if self.curr_vel_transition_step < self.num_vel_transition_steps:
            current_vel = [self.target_vel_m[i] * self.curr_vel_transition_step + self.prev_target_base_vel[i] for i in range(3)]
            self.curr_vel_transition_step += 1
            return current_vel
        return None

    def clear_data(self):
        self.contact_timings = {"fr":[], "hl":[], "fl":[], "hr":[]}
        self.euler_rotation = [0, 0, 0]
        self.velocity_list = []
        self.x_dist_list = []
        self.rpy_list = []

    def write_data(self, data_path, file_postfix = ""):
        if self.evaluating:
            if len(self.contact_timings["fr"]) != 0:
                print("Reseting Robot")
                path = os.path.join(data_path, f"contact_timings_{file_postfix}.csv")
                f = open(path, "w")
                f.write("fr,hl,fl,hr\n")
                for i in range(len(self.contact_timings["fr"])):
                    line = f"{self.contact_timings['fr'][i]},{self.contact_timings['hl'][i]},{self.contact_timings['fl'][i]},{self.contact_timings['hr'][i]}\n"
                    f.write(line)
                f.close()

                path = os.path.join(data_path, f"velocity_{file_postfix}.csv")
                f = open(path, "w")
                f.write("velocity\n")
                for vel in self.velocity_list:
                    f.write(f"{vel}\n")

                f.close()

                path = os.path.join(data_path, f"x_dist_{file_postfix}.csv")
                f = open(path, "w")
                f.write("xDist\n")
                for xd in self.x_dist_list:
                    f.write(f"{xd}\n")

                f.close()


                path = os.path.join(data_path, f"rpy_{file_postfix}.csv")
                f = open(path, "w")
                f.write("R,P,Y\n")
                for r,p,y in self.rpy_list:
                    f.write(f"{r},{p},{y}\n")

                f.close()


            


        self.contact_timings = {"fr":[], "hl":[], "fl":[], "hr":[]}
        self.velocity_list = []
        self.x_dist_list = []


    def log_data(self):
        if self.evaluating:
            p.performCollisionDetection()
            contact_pts = p.getContactPoints(bodyA = self.ground, bodyB = 1)
            contact_feet_indices = []
            for contact_pt in contact_pts:
                foot_idx = contact_pt[4]
                
                if foot_idx in self.contact_dict:
                    contact_feet_indices.append(self.contact_dict[foot_idx])

            for key in self.contact_timings:
                if key in contact_feet_indices:
                    self.contact_timings[key].append(1)
                else:
                    self.contact_timings[key].append(0)

    def reset_robot(self):
        # self.write_data()
        init_pos = [0.0, -0.8, 1.7, 0.0, -0.8, 1.7, 0.0, -0.8, 1.7, 0.0, -0.8, 1.7,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p.resetBasePositionAndOrientation(
            self.quadruped, self.init_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        for j in range(12):
            p.resetJointState(
                self.quadruped, self.motor_id_list[j], init_pos[j], init_pos[j+12])

        self.cpp_gait_ctrller.init_controller(convert_type(
            self.freq), convert_type([self.stand_kp, self.stand_kd, self.joint_kp, self.joint_kd]))

        for _ in range(10):
            self._get_data_from_sim()
            self.cpp_gait_ctrller.pre_work(convert_type(
                self.imu_data), convert_type(self.leg_data))
            p.stepSimulation()

        for j in range(12):
            force = 0
            p.setJointMotorControl2(
                self.quadruped, j, p.VELOCITY_CONTROL, force=force)

        self.cpp_gait_ctrller.set_robot_mode(convert_type(1))
        for _ in range(1000):
            self._run()
            p.stepSimulation
        self.cpp_gait_ctrller.set_robot_mode(convert_type(2))

        obs = np.array([0.0]*16)
        # return obs

        img = self.get_camera_image()


        self.target_base_vel = [0]*3
        self.target_current_vel = [0]*3
        self.prev_target_base_vel = [0]*3
        self.target_vel_m = [0]*3
        self.curr_vel_transition_step = 0

        self.gait_correction.reset()
        self.gait_correction.set_gait_params(self.stand_act)

        return obs, img


    def step(self, gait_param):
        obs = np.array([0.0]*16)
        self._robot_dist = 0

        # for i in range(len(gait_param)):
        #     gait_param[i] = gait_param[i].item()
        if self.gait_corr:
            self.gait_correction.set_gait_params(gait_param)
        else:
            gait_param = scale_footstep_timing(gait_param)
            gait_param = convert_to_list(gait_param)
            self.cpp_gait_ctrller.set_gait_param(convert_type(gait_param))
        
        num_repeat = 0
        while num_repeat < self.action_repeat:
            # imgW, imgH = 800, 400
            # img = p.getCameraImage(imgW, imgH, renderer=p.ER_BULLET_HARDWARE_OPENGL, flags = p.ER_NO_SEGMENTATION_MASK)
            # rgbBuffer = np.reshape(img[2], (imgH, imgW, 4))
            # bgrBuffer = cv2.cvtColor(rgbBuffer, cv2.COLOR_RGB2BGR)
            # cv2.imshow("i", bgrBuffer)
            # cv2.waitKey(1)
            
            self.log_data()
            # time.sleep(0.5)

            if self.gait_corr:
                current_vel = self.step_desired_velocity()
                if current_vel is not None:
                    if sum([abs(self.target_current_vel[i] - current_vel[i]) for i in range(3)]) > 1e-3:
                        # print(current_vel)
                        self.target_current_vel = deepcopy(current_vel)
                        self.cpp_gait_ctrller.set_robot_vel(convert_type(current_vel))
                        
                corr_gait_param = self.gait_correction.step()
                if corr_gait_param is not None:
                    print(corr_gait_param)
                    self.cpp_gait_ctrller.set_gait_param(convert_type(corr_gait_param))

            self._run()
            obs = self._get_obs(obs)
            
            self.velocity_list.append(self.base_vel[0])
            self.x_dist_list.append(self.base_position[0])
            self.rpy_list.append([self.euler_rotation[i] for i in range(3)])
            
            self._robot_dist += np.sqrt(((self.base_position[0] - self._last_base_pos[0])**2 +
                                         (self.base_position[1] - self._last_base_pos[1])**2))
            self._last_base_pos = self.base_position

            if self.visualization:
                time.sleep(1.0/self.freq)

            robot_safe = self.cpp_gait_ctrller.get_safety_check()
            if not robot_safe:
                break

            num_repeat += 1

        if(num_repeat) == 0:
            obs = np.array([0.0]*16)
        
        else:
            obs[3:-2] /= (num_repeat + 1)  # average obs per step
            # obs[-2] /= self._robot_dist  # energy consumption per meter
            obs[-2] /= (num_repeat + 1)  # avg force in step
            obs[-1] = float(num_repeat+1) / self.action_repeat
        
        img = self.get_camera_image()
        # img = np.zeros((40, 40))
        
        return obs, img, robot_safe

    def set_vel(self, target_base_vel):
        self.curr_vel_transition_step = 0
        self.prev_target_base_vel = deepcopy(self.target_base_vel)
        self.target_base_vel = target_base_vel
        
        self.target_vel_m = [(self.target_base_vel[i] - self.prev_target_base_vel[i]) / (self.num_vel_transition_steps - 1) for i in range(3)]
        
        if not self.gait_corr:
            self.cpp_gait_ctrller.set_robot_vel(convert_type(self.target_base_vel))

    def _cal_energy_consumption(self):
        engergy = 0
        for i in range(12):
            engergy += np.abs(self.tau[i] *
                              self.leg_data[12+i]) * (1.0 / self.freq)
        engergy /= 1000.0
        engergy_consumption = engergy

        return engergy_consumption.item()

    def _cal_force_sum(self):
        force = np.sum(np.array(self.tau)**2) / 1000.0
        return force

    def _get_obs(self, obs):
        base_acc = np.array(self.imu_data[0:3]) + np.array(self.sim_gravity)
        rpy = p.getEulerFromQuaternion(self.imu_data[3:7])
        rpy_rate = self.imu_data[7:10]
        avg_foot_x = self.cpp_gait_ctrller.get_prf_foot_coor()
        # energy = self._cal_energy_consumption()
        force = self._cal_force_sum()

        obs[0:3] = np.abs(np.array(self.target_current_vel[0:3]))  # target linear_xy, angular_z vel
        obs[3:5] += np.abs(np.array(self.base_vel[0:2]))  # real linear xy vel
        obs[5] += np.abs(self.imu_data[9])  # real angular_z vel
        obs[6:9] += np.abs(base_acc)  # linear acc
        obs[9:11] += np.abs(np.array(rpy[0:2]))  # rp
        obs[11:13] += np.abs(np.array(rpy_rate[0:2]))  # rp_rate
        # obs[13:25] = np.abs(np.array())  # joint pos
        obs[13] += np.abs(np.array(avg_foot_x)) # 4 foot avg coor in robot frame 
        # obs[14] += np.abs(np.array(energy))  # energy
        obs[14] += np.abs(np.array(force))  # force

        return obs

    def _run(self):
        # get data from simulator
        self._get_data_from_sim()

        # call cpp function to calculate mpc tau
        tau = self.cpp_gait_ctrller.toque_calculator(convert_type(
            self.imu_data), convert_type(self.leg_data))

        for i in range(12):
            self.tau[i] = tau.contents.eff[i]

        # set tau to simulator
        p.setJointMotorControlArray(bodyUniqueId=self.quadruped,
                                    jointIndices=self.motor_id_list,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=self.tau)

        # reset visual cam
        p.resetDebugVisualizerCamera(1, 25, -30, self.base_position)
        # p.resetDebugVisualizerCamera(5, 0, -30, self.base_position)

        p.stepSimulation()

        return

    def _get_data_from_sim(self):
        get_matrix = []
        get_velocity = []
        get_invert = []

        self.base_position, base_orientation = p.getBasePositionAndOrientation(
            self.quadruped)
        self.euler_rotation = p.getEulerFromQuaternion(base_orientation)
        get_velocity = p.getBaseVelocity(self.quadruped)

        get_invert = p.invertTransform(self.base_position, base_orientation)
        get_matrix = p.getMatrixFromQuaternion(get_invert[1])

        # IMU ori
        self.imu_data[3] = base_orientation[0]
        self.imu_data[4] = base_orientation[1]
        self.imu_data[5] = base_orientation[2]
        self.imu_data[6] = base_orientation[3]
        # IMU rpy_rate
        self.imu_data[7] = get_matrix[0] * get_velocity[1][0] + get_matrix[1] * \
            get_velocity[1][1] + get_matrix[2] * get_velocity[1][2]
        self.imu_data[8] = get_matrix[3] * get_velocity[1][0] + get_matrix[4] * \
            get_velocity[1][1] + get_matrix[5] * get_velocity[1][2]
        self.imu_data[9] = get_matrix[6] * get_velocity[1][0] + get_matrix[7] * \
            get_velocity[1][1] + get_matrix[8] * get_velocity[1][2]
        # IMU acc
        linear_X = (get_velocity[0][0] - self.get_last_vel[0]) * self.freq
        linear_Y = (get_velocity[0][1] - self.get_last_vel[1]) * self.freq
        linear_Z = 9.8 + (get_velocity[0][2] -
                          self.get_last_vel[2]) * self.freq
        self.imu_data[0] = get_matrix[0] * linear_X + \
            get_matrix[1] * linear_Y + get_matrix[2] * linear_Z
        self.imu_data[1] = get_matrix[3] * linear_X + \
            get_matrix[4] * linear_Y + get_matrix[5] * linear_Z
        self.imu_data[2] = get_matrix[6] * linear_X + \
            get_matrix[7] * linear_Y + get_matrix[8] * linear_Z

        self.base_vel[0] = get_matrix[0] * get_velocity[0][0] + get_matrix[1] * \
            get_velocity[0][1] + get_matrix[2] * get_velocity[0][2]
        self.base_vel[1] = get_matrix[3] * get_velocity[0][0] + get_matrix[4] * \
            get_velocity[0][1] + get_matrix[5] * get_velocity[0][2]
        self.base_vel[2] = get_matrix[6] * get_velocity[0][0] + get_matrix[7] * \
            get_velocity[0][1] + get_matrix[8] * get_velocity[0][2]

        # joint data
        joint_state = p.getJointStates(self.quadruped, self.motor_id_list)
        self.leg_data[0:12] = [joint_state[0][0], joint_state[1][0], joint_state[2][0],
                               joint_state[3][0], joint_state[4][0], joint_state[5][0],
                               joint_state[6][0], joint_state[7][0], joint_state[8][0],
                               joint_state[9][0], joint_state[10][0], joint_state[11][0]]

        self.leg_data[12:24] = [joint_state[0][1], joint_state[1][1], joint_state[2][1],
                                joint_state[3][1], joint_state[4][1], joint_state[5][1],
                                joint_state[6][1], joint_state[7][1], joint_state[8][1],
                                joint_state[9][1], joint_state[10][1], joint_state[11][1]]
        com_velocity = [get_velocity[0][0],
                        get_velocity[0][1], get_velocity[0][2]]
        self.get_last_vel.clear()
        self.get_last_vel = com_velocity

        return


if __name__ == "__main__":
    act = np.array([20,20,20,20,20,20,20,20,20]) # stand
    act = np.array([20,0,10,5,15,15,15,15,15]) # walk20
    # act = np.array([10,0,2,7,9,5,5,5,5]) # gallop10

    # act = np.array([16,0,8,4,12,12,12,12,12]) # walk16
    # act = np.array([16,0,8,8,0,12,12,12,12]) # ???
    # act = np.array([12,0,6,3,9,9,9,9,9]) # walk10

    act = act.tolist()

    hb = HybridGaitRobot()
    hb.reset_robot()
    # hb.set_vel([0.19, 0, 0])
    while True:
        hb.step(act)