import gym
import math

import numpy as np

import dogfight_client as df
from Parameters import *
from Manevra_Modlari.manevra_modlari import *
from Harfang_Modulu.harfang_modulu import *
# from Enerji_Hesaplamalari.energy_calculations import *
import random
import csv
import pandas as pd


class HarfangEnv(gym.Env):
    """
    Aciklama
    RL tabanli Hiearsik yapili  Fuzeden kacma algoritmasi 基于RL的Hiearsik结构导弹规避算法

    Master---> LSTM+SAC 主人：长短期记忆网络+sac
    Slave----> SAC      奴隶：sac
    
    Observation  (Duzenlenecek) 观察（待组织）
    Type : Box(2)
    Aircraft_Start_x_Pos : Location info : min = -2000  : max = +2000 飞机
    Aircraft_Start_y_Pos : Location info : min = -2000  : max = +2000
    Aircraft_Start_z_Pos : Location info : min = +2000  : max = +6000
    Rocket_Start_x_Pos   : Location info : min = -5000  : max = +5000 火箭
    Rocket_Start_y_Pos   : Location info : min = -5000  : max = +5000
    Rocket_Start_z_Pos   : Location info : min = +2000  : max = +6000
    (Duzenlenecek)



    Action    动作
    Type : Box(3)
    Rudder    :  min = -1 : max = +1 主舵
    Aileron   :  min = -1 : max = +1 副翼
    Elevator  :  min = -1 : max = +1 电梯


    Reward (Duzenlenecek) 奖励
    -Fuzenin gorus acisina bagli olarak (WEZ), reward tanimlanabilir  根据引信（WEZ）的观察角度，可以定义为奖励
    -Belli bir mesafe altinda aralarindaki mesafeye bagli ceza tanimlanabilir 在一定的距离下，可以根据它们之间的距离来定义一个惩罚。
    -Fuzenin ucaga carpmasi ile episode bitip yuksek ceza verilebilir 如果导弹击中了飞机，剧情可能会结束，并可能被处以高额罚款。

    Termination 终止条件
    -Fuzenin ucaga carpmasi 引信撞上飞机
    -Fuzenin ucagi iskalamasi 导弹错过飞机
    -Step sayisi asilirsa 如果超过了步骤数
    

    """

    def __init__(self):

        self.n_items = 1
        self.obs_space_min_x = -3000.0     # 观察空间大小
        self.obs_space_max_x = +3000.0
        self.obs_space_min_y = -3000.0
        self.obs_space_max_y = +3000.0
        self.obs_space_min_z = -0.0
        self.obs_space_max_z = +6000.0

        self.start_space_aircraft_min_x = + 2000.0   # 里面黑色框
        self.start_space_aircraft_max_x = - 2000.0
        self.start_space_aircraft_min_y = + 2000.0
        self.start_space_aircraft_max_y = - 2000.0
        self.start_space_aircraft_min_z = + 2000.0
        self.start_space_aircraft_max_z = + 6000.0

        self.start_space_rocket_min_x = + 5000.0   # 飞机在这个范围初始位置生成？
        self.start_space_rocket_max_x = - 5000.0
        self.start_space_rocket_min_y = + 5000.0
        self.start_space_rocket_max_y = - 5000.0
        self.start_space_rocket_min_z = + 2000.0
        self.start_space_rocket_max_z = + 6000.0

        self.end_space_min_x = -10000.0  # 外面黑色框
        self.end_space_max_x = +10000.0
        self.end_space_min_y = -10000.0
        self.end_space_max_y = +10000.0
        self.end_space_min_z = 0.0
        self.end_space_max_z = +10000.0

        self.value = None

        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]),
                                           dtype=np.float32)  # 动作空间范围相当于定义出了3个-1~1范围
        # np.array创建矩阵，列表-->矩阵
        # box  每个维度有不相同的约束 运行出：Box([-1. -1. -1.], [1. 1. 1.], (3,), float64)
        self.maxStepSize = 2000  # 最大步数2000
        self.observation_space = gym.spaces.Box(low=np.array([self.obs_space_min_x, self.obs_space_min_y]),
                                                high=np.array([self.obs_space_max_x, self.obs_space_max_y]),
                                                dtype=np.float32)
        # box  每个维度有不相同的约束 运行出：Box([-10000. -10000.], [10000. 10000.], (2,), float32)
        self.starting_area_aircraft = gym.spaces.Box(low=np.array(
            [self.start_space_aircraft_min_x, self.start_space_aircraft_min_y, self.start_space_aircraft_min_z]),
                                                     high=np.array([self.start_space_aircraft_max_x,
                                                                    self.start_space_aircraft_max_y,
                                                                    self.start_space_aircraft_max_z]), dtype=np.float32)
        # Box([-2000. -2000. 2000.], [2000. 2000. 6000.], (3,), float32)
        self.starting_area_rocket = gym.spaces.Box(
            low=np.array([self.start_space_rocket_min_x, self.start_space_rocket_min_y, self.start_space_rocket_min_z]),
            high=np.array(
                [self.start_space_rocket_max_x, self.start_space_rocket_max_y, self.start_space_rocket_max_z]),
            dtype=np.float32)
        self.location_aircraft = self.starting_area_aircraft.sample()  # 随机抽取[-1,1]
        self.location_rocket = self.starting_area_rocket.sample()
        self.done = False
        self.steps = 1
        self.Threshold_Hit = 200
        self.Threshold_Escape = 50000
        self.Missile_Loc = None
        self.loc_diff = 0
        self.Plane_Linear_Speed = 0
        self.Plane_Altitude = 0

        self.Plane_Destroyed = False
        self.Plane_Wreck = False
        self.Plane_Crashed = False
        self.Oppo_Destroyed = False
        self.Oppo_Wreck = False
        self.Oppo_Crashed = False
        self.Oppo_TargetID = None
        self.Oppo_Locked = False

        self.states = None
        self.Plane_ID_oppo = "ennemy_1"  # 敌机ID
        self.Plane_ID_ally = "ally_1"  # 我方ID

        self.Aircraft_Loc = None
        self.reward_ally = 0  # 我方奖励（ally友军）
        self.reward_oppo = 0  # 敌方奖励（opponent 敌手）

        self.Ally_target_locked = None
        self.Oppo_target_locked = None
        # 奖励
        self.reward = 0
        self.HA_Kutle = 9000  # kg
        # self.Pos_X_ally = random.randint(-2000, 3500)  # ally 盟友
        # self.Pos_Y_ally = random.randint(2000, 3500)
        # self.Pos_Z_ally = random.randint(-2000, 5000)
        self.Pos_X_ally = 1000 # ally 盟友
        self.Pos_Y_ally = 1000
        self.Pos_Z_ally = 1000
        self.Pos_X_oppo = 3000
        self.Pos_Y_oppo = 3000
        self.Pos_Z_oppo = -3000
        self.done_ally = False
        self.done_oppo = False
        self.Plane_Irtifa = 0  # 海拔高度
        self.Oppo_Irtifa = 0
        self.ally_alt = 0
        self.oppo_alt = 0
        self.plane_heading = 0
        self.plane_heading_2 = 0

        self.Ally_target_out_of_range = None
        self.Oppo_target_out_of_range = None

    def reset(self):  # 一回合结束后重置环境
        self.done = False
        self.steps = 1  # 本次回合的第一步
        state_ally, _, _ = self._get_observation(self.Plane_ID_ally, self.Plane_ID_oppo)
        # 返回的是飞机位置，飞机欧拉角，飞机推力，敌机推力，敌机俯仰角，敌机翻滚角，目标角度 组成的矩阵
        # 未返回的是我方飞机和敌方飞机是否被摧毁，残骸，坠毁信息和全部位置信息，以及敌机TargetID，
        # state_oppo = state_ally  # self._get_observation(self.Plane_ID_ally, self.Plane_ID_oppo)
        _, _, state_oppo = self._get_observation(self.Plane_ID_ally, self.Plane_ID_oppo)
        self._reset_machine(self.Plane_ID_ally, self.Plane_ID_oppo)

        return state_ally, state_oppo

    def step(self, action_ally):
        """
        输入为本机动作
        返回是本机状态，奖励值，是否终止等
        """
        #print("动作：",action_ally)
        self._apply_action(self.Plane_ID_ally, action_ally) # 完成了欧拉角计算，设置推力，更新场景
        #简单训练敌机
        action_oppo = np.array([0, 0, 0])
        self._apply_action2(self.Plane_ID_oppo, action_oppo)
        state_ally, _, _ = self._get_observation(self.Plane_ID_ally, self.Plane_ID_oppo)  # 获取飞机位置，飞机欧拉角，飞机推力，敌机推力，敌机俯仰角，敌机翻滚角，目标角度
        _, _, state_oppo = self._get_observation(self.Plane_ID_ally, self.Plane_ID_oppo)
        self._get_loc_diff()  # 获取的是飞机与（0，4000，0）之间的距离
        self._get_reward()  # 计算奖励值
        self._get_termination()
        state = self.states
        return state_ally, self.reward_ally, self.done, {}

    # def render(self, mode='False'):
    #     df.set_renderless_mode(mode)


    def _get_reward(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        oppo_state = df.get_plane_state(self.Plane_ID_oppo)
        self.reward_ally = 0
        self.reward_oppo = 0
        ###############################################
        ##速度奖励
        ###############################################
        # 速度矢量
        Plane_v_dir = plane_state['move_vector']
        oppo_v_dir = oppo_state['move_vector']
        Plane_v = plane_state['linear_speed']
        oppo_v = oppo_state['linear_speed']
        v_diff = Plane_v - oppo_v  # 速度差

        ###############################################
        ##高度差奖励
        ###############################################
        Height_Difference = self.Plane_Irtifa - self.Oppo_Irtifa
        if Height_Difference > 500 and Height_Difference < 3000:
            self.reward_ally += 0.3
            #print("高度差奖励1")
        elif Height_Difference< -500:
            self.reward_ally -= 0.3
            #print("高度差奖励-1")

        ###############################################
        ##角度奖励
        ###############################################
        AA, ATA = self._get_AA_ATA()
        self.reward_ally+= (4-4*(AA+ATA)/math.pi)
        #print("角度奖励",4-4*(AA+ATA)/math.pi)
        ###############################################
            ##距离奖励
        ###############################################
        self._get_loc_diff()
         # 获取距离信息self.loc_diff
        if self.loc_diff > self.Dmin and self.loc_diff < self.Dmax:
            if AA < math.pi / 3:
                    self.reward_ally += (1.5-0.001*self.loc_diff)
                    #print("距离奖励",1.5-0.001*self.loc_diff)
        else:
            self.reward_ally -= 2
            #print("距离奖励-2")


        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.reward_ally -= 50

        if self.Ally_target_locked:
            self.reward_ally += 100

        if self.Oppo_target_locked:
            self.reward_ally -= 50



    def _apply_action(self, Plane_ID_ally, action_ally):
        """
        输入：飞机ID 我方动作
        完成了欧拉角计算，设置推力，更新场景
        """
        df.set_plane_pitch(Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(Plane_ID_ally, float(action_ally[2]))
        # 飞行器控制中的欧拉角(pitch俯仰角,roll翻滚角,yaw偏航角《摇头》)
        df.set_plane_thrust(Plane_ID_ally, 1)  # 设置Plane_ID_ally推力为等级1
        df.activate_post_combustion(Plane_ID_ally)
        df.update_scene()  # 更新场景
        # Reward func3
        # WEZ icinde olmasinida ekle target_locked
    def _apply_action2(self, Plane_ID_oppo, action_oppo):

         df.set_plane_pitch(Plane_ID_oppo, float(action_oppo[0]))
         df.set_plane_roll(Plane_ID_oppo, float(action_oppo[1]))
         df.set_plane_yaw(Plane_ID_oppo, float(action_oppo[2]))
         df.set_plane_thrust(Plane_ID_oppo,0.8)  # 设置Plane_ID_ally推力为等级1
         df.update_scene()  # 更新场景

    def _get_AA_ATA(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        oppo_state = df.get_plane_state(self.Plane_ID_oppo)
        Plane_v_dir = plane_state['move_vector']
        oppo_v_dir = oppo_state['move_vector']
        self._get_loc_diff()
        # 位置矢量
        Pos_Dir = (oppo_state["position"][0] - plane_state["position"][0],
                   oppo_state["position"][1] - plane_state["position"][1],
                   oppo_state["position"][2] - plane_state["position"][2])
        # 脱离角 有bug，取值正负
        AA = math.acos(max(-1, min(1, np.dot(oppo_v_dir, Pos_Dir) / (np.linalg.norm(oppo_v_dir) * self.loc_diff))))
        # 偏离角
        ATA = math.acos(max(-1, min(1, np.dot(Plane_v_dir, Pos_Dir) / (np.linalg.norm(Plane_v_dir) *self.loc_diff))))
        return AA, ATA
    def _get_termination(self):  # 终止条件

        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.done = True
            self.value = 1  # 超出安全范围
        if self.Oppo_target_locked :
            self.done = True
            self.value = 2  # 我方被击败
        if self.Ally_target_locked :
            self.done = True
            self.value = 3  # 敌方被击败


    def _reset_machine(self, Plane_ID_ally, Plane_ID_oppo):
        """
        恢复飞机状态，
        """
        df.set_health("ennemy_2", 1)
        df.set_health("ennemy_1", 1)     # 将ennemy_1 生命等级设为1
        df.rearm_machine(Plane_ID_ally)  # 重新武装
        df.rearm_machine(Plane_ID_oppo)  # 重新武装
        # df.set_target_id(Plane_ID_ally,Plane_ID_oppo)
        # df.set_target_id( Plane_ID_ally)
        df.reset_machine_matrix(Plane_ID_ally, 2000, 3000, -900, 0, 0,-1)  # random.randint(-3000, 4500) 重设机器模型（记录飞机ID，XYZ坐标，rx,ry,rz旋转信息）
        df.reset_machine_matrix(Plane_ID_oppo, 2500, 3000, -1000, 0, 0, 0)
        #df.reset_machine_matrix(Plane_ID_ally, 200, 3000, 0, 0, 0 , 0)  # random.randint(-3000, 4500) 重设机器模型（记录飞机ID，XY高度Z坐标，rx,ry,rz旋转信息）
        #df.reset_machine_matrix(Plane_ID_oppo, 200, 3000, 1000, 0, 0, 0)  # 俯仰 偏航 翻滚 偏航math.pi反转
        df.set_target_id(Plane_ID_ally, Plane_ID_oppo)
        df.set_target_id(Plane_ID_oppo, Plane_ID_ally)

        df.set_plane_thrust(Plane_ID_ally, 1)  # 记录飞机ID，以及对应的推力等级
        df.set_plane_thrust(Plane_ID_oppo, 0.8)

        # df.set_plane_thrust("ennemy_2", 0.8)
        df.set_plane_linear_speed(Plane_ID_ally, 300)  # 记录飞机ID，以及线性速度
        df.set_plane_linear_speed(Plane_ID_oppo, 200)
        df.retract_gear(Plane_ID_ally)  # 收回起落架
        df.retract_gear(Plane_ID_oppo)


    def _get_loc_diff(self):

        self.Dmax = 2000
        self.Dmin = 100
        self.loc_diff = (((self.Aircraft_Loc[0]-self.Oppo_Loc[0]) ** 2) + ((self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) + ((self.Aircraft_Loc[2]-self.Oppo_Loc[2]) ** 2)) ** (1 / 2)

    def _get_observation(self, Plane_ID_ally, Plane_ID_oppo):  # 观察环境
        """
        输入我方飞机ID，敌方飞机ID
        返回1是飞机位置，飞机欧拉角，飞机推力，敌机推力，敌机俯仰角，敌机翻滚角，目标角度 组成的矩阵
        返回2是我方飞机和敌方飞机是否被摧毁，残骸，坠毁信息和全部位置信息，以及敌机TargetID，
        """
        # Plane States # 飞机状态
        plane_state = df.get_plane_state(Plane_ID_ally)  # 传入飞机ID，获得飞机状态
        #print(plane_state)
        # Plane_Pos =[round(plane_state["position"][0] / NormStates["Plane_position"],3),  # 飞机位置
        #              round(plane_state["position"][1] / NormStates["Plane_position"],3),
        #              round(plane_state["position"][2] / NormStates["Plane_position"],3)]
        Plane_Pos = [plane_state["position"][0] / NormStates["Plane_position"],  # 飞机位置
                     plane_state["position"][1] / NormStates["Plane_position"],
                     plane_state["position"][2] / NormStates["Plane_position"]]

        #print(Plane_Pos)
        Plane_Euler = [plane_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"],  # 飞机欧拉角
                       plane_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                       plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]

        self.Plane_Health = plane_state["health_level"]
        # Plane_Heading = round(plane_state["heading"] / NormStates["Plane_heading"],3)  # 飞机推力
        # Plane_Pitch_Att = round(plane_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"],3)
        # Plane_Roll_Att = round(plane_state["roll_attitude"] / NormStates["Plane_roll_attitude"],3)
        Plane_Heading = plane_state["heading"] / NormStates["Plane_heading"]  # 飞机推力
        Plane_Pitch_Att = plane_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        Plane_Roll_Att = plane_state["roll_attitude"] / NormStates["Plane_roll_attitude"]
        Plane_Hor_Speed = plane_state["horizontal_speed"] / NormStates["Plane_horizontal_speed"]  # 横向速度
        Plane_Ver_Speed = plane_state["vertical_speed"] / NormStates["Plane_vertical_speed"]      # 纵向速度
        # Plane_Acc = plane_state["linear_acceleration"] / NormStates["Plane_linear_acceleration"]

        # Missile States
        Oppo_state = df.get_plane_state(Plane_ID_oppo)

        # Oppo_Pos = [round(Oppo_state["position"][0] / NormStates["Plane_position"],3),
        #             round(Oppo_state["position"][1] / NormStates["Plane_position"],3),
        #             round(Oppo_state["position"][2] / NormStates["Plane_position"],3)]
        Oppo_Pos = [Oppo_state["position"][0] / NormStates["Plane_position"],
                    Oppo_state["position"][1] / NormStates["Plane_position"],
                    Oppo_state["position"][2] / NormStates["Plane_position"]]
        Oppo_Euler = [Oppo_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                      Oppo_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                      Oppo_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
        self.Oppo_Health = Oppo_state["health_level"]
        # Oppo_Heading = round(Oppo_state["heading"] / NormStates["Plane_heading"],3)
        # Oppo_Pitch_Att = round(Oppo_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"],3)
        # Oppo_Roll_Att = round(Oppo_state["roll_attitude"] / NormStates["Plane_roll_attitude"],3)
        Oppo_Heading = Oppo_state["heading"] / NormStates["Plane_heading"]
        Oppo_Pitch_Att = Oppo_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        Oppo_Roll_Att = Oppo_state["roll_attitude"] / NormStates["Plane_roll_attitude"]
        Oppo_Hor_Speed = Oppo_state["horizontal_speed"] / NormStates["Plane_horizontal_speed"]
        Oppo_Ver_Speed = Oppo_state["vertical_speed"] / NormStates["Plane_vertical_speed"]
        # Oppo_Acc = Oppo_state["linear_acceleration"] / NormStates["Plane_linear_acceleration"]
        # Some other informations
        # print(plane_state["position"][1],Oppo_state["position"][1])
        self.plane_heading_2 = Oppo_state["heading"]     # 敌机俯仰角
        self.Plane_Destroyed = plane_state["destroyed"]  # 被摧毁
        self.Plane_Wreck = plane_state["wreck"]          # 残骸
        self.Plane_Crashed = plane_state["crashed"]      # 坠毁
        self.Plane_Irtifa = plane_state["position"][1]   # 海拔高度
        self.Oppo_Irtifa = Oppo_state["position"][1]

        self.Ally_target_out_of_range = plane_state["target_out_of_range"]  # 目标值超出范围
        self.Oppo_target_out_of_range = Oppo_state["target_out_of_range"]
        self.Oppo_Destroyed = Oppo_state["destroyed"]                       # 被摧毁
        self.Oppo_Wreck = Oppo_state["wreck"]                   # 残骸
        self.Oppo_Crashed = Oppo_state["crashed"]                   # 坠毁
        self.Oppo_TargetID = Oppo_state["target_id"]
        self.Aircraft_Loc = plane_state["position"]
        self.Oppo_Loc = Oppo_state["position"]

        self.Ally_target_locked = plane_state["target_locked"]
        self.Oppo_target_locked = Oppo_state["target_locked"]
        # for reward func
        self.Plane_Linear_Speed = plane_state['linear_speed']       # 线性速度
        self.Plane_Altitude = plane_state['altitude']               # 海拔高度
        #target_angle = round(plane_state['target_angle'] / 360,3)
        target_angle =plane_state['target_angle'] # 目标角度
        aa,ata=self._get_AA_ATA()
        AA=round(aa/math.pi,3)
        ATA=round(ata/math.pi,3)
        # print(Plane_Pos[0]*NormStates["Plane_position"],Oppo_Pos[0]*NormStates["Plane_position"],Plane_Pos[1]*NormStates["Plane_position"],Oppo_Pos[1]*NormStates["Plane_position"],Plane_Pos[2]*NormStates["Plane_position"],Oppo_Pos[2]*NormStates["Plane_position"])
        Pos_Diff = [Plane_Pos[0] - Oppo_Pos[0], Plane_Pos[1] - Oppo_Pos[1], Plane_Pos[2] - Oppo_Pos[2]]
        States = np.concatenate((Plane_Pos, Pos_Diff, Plane_Heading,Plane_Pitch_Att, Plane_Roll_Att,AA,ATA), axis=None)  # 拼接矩阵
        #States = np.concatenate((Plane_Pos, Pos_Diff, Plane_Heading, Oppo_Heading, Plane_Pitch_Att, Plane_Roll_Att, target_angle), axis=None)
        Oppo_States = np.concatenate((Oppo_Pos, Pos_Diff,
                                 Plane_Heading, Oppo_Pitch_Att, Oppo_Roll_Att, AA,ATA), axis=None)  # 拼接矩阵
        # 返回的是飞机位置，飞机欧拉角，飞机推力
        # 敌机推力，敌机俯仰角，敌机翻滚角，目标角度
        Others = np.concatenate((self.Plane_Destroyed, self.Plane_Wreck, self.Plane_Crashed,
                                 self.Oppo_Destroyed,self.Oppo_Wreck, self.Oppo_Crashed, self.Oppo_TargetID,
                                 self.Aircraft_Loc, self.Oppo_Loc), axis=None)
        # 返回的是飞机是否被摧毁，已经成为残骸，坠毁
        # 敌机是否被摧毁，已经成为残骸，坠毁，敌机的ID
        # 我方飞机全部位置信息，敌方飞机全部位置信息

        self.states = States
        return States, Others, Oppo_States

    def get_action(self,Plane_ID_ally):
        plane_state = df.get_plane_state(Plane_ID_ally)
        pitch=plane_state["user_pitch_level"]
        roll=plane_state["user_roll_level"]
        yaw=plane_state["user_yaw_level"]
        action=np.array([pitch,roll,yaw])
        return action