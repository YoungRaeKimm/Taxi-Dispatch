# environment tmp
from haversine import haversine
import numpy as np
# import cupy as np
import pickle as pkl
from collections import deque
import scipy as sp
import seaborn as sns
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, save_model
import tqdm
import pylab


class Env:
    def __init__(self, supply, demand):
        self.action_size = 9
        self.demand = demand  # 수요
        self.supply = supply  # 공급

        self.PD_distance1 = None  # Passenger들과 Driver들 사이의 거리 강남에서 강남
        self.PD_distance2 = None  # 강남에서 강북

        #         self.prefer_region = None #선호지역
        #         self.Dest = None #수요들의 도착지의 좌표
        #         self.Dest_idx = None #Dest의 index들
        #         self.Rev_Dest_idx = None #Dest_idx에서 key,vale 반대
        self.demand_hour = None
        self.demand_hour1 = None  # 시간대별의 수요
        self.demand_hour2 = None  # 시간대별의 수요
        self.demand_hour3 = None  # 시간대별의 수요
        self.demand_hour4 = None  # 시간대별의 수요

        self.supply_hour = None
        self.supply_hour1 = None  # 시간대별의 공급
        self.supply_hour2 = None  # 시간대별의 공급

        self.hour = 0

        self.load_data()

        self.divide_district()

        self.OD_distance1 = self.get_od(1)  ### 수요들의 OD거리 강남에서 강남
        self.OD_distance2 = self.get_od(2)  ### 강남에서 강북
        self.OD_distance3 = self.get_od(3)  ### 강북에서 강북
        self.OD_distance4 = self.get_od(4)  ### 강북에서 강남
        # test code
        #         self.avg_thre = []
        # test code

    def divide_district(self):
        self.demand_hour1 = self.demand_hour[self.demand_hour[:, 2] == '강남']
        self.demand_hour1 = self.demand_hour1[self.demand_hour1[:, 1] == '강남']

        self.demand_hour2 = self.demand_hour[self.demand_hour[:, 2] == '강남']
        self.demand_hour2 = self.demand_hour2[self.demand_hour2[:, 1] == '강북']

        self.demand_hour3 = self.demand_hour[self.demand_hour[:, 2] == '강북']
        self.demand_hour3 = self.demand_hour3[self.demand_hour3[:, 1] == '강북']

        self.demand_hour4 = self.demand_hour[self.demand_hour[:, 2] == '강북']
        self.demand_hour4 = self.demand_hour4[self.demand_hour4[:, 1] == '강남']

        self.supply_hour1 = self.supply_hour[self.supply_hour[:, 1] == '강남']
        self.supply_hour2 = self.supply_hour[self.supply_hour[:, 1] == '강북']

    # Data load
    def load_data(self):
        print('Loading Data')

        #         self.Dest = pkl.load(open('X:/Drive/dp/gus.pkl', 'rb')) #자치구들의 좌표
        #         self.Dest_idx = self.make_idx() #Dest의 index들
        #         self.Rev_Dest_idx = self.make_reverse_idx() #Dest의 index들

        #         self.demand = pkl.load(open('X:/Drive/dp/demand_final.pkl', 'rb')) #수요
        #         '''
        #         for i in range(len(self.demand)):
        #             self.demand[i,1] = self.Dest_idx[self.demand[i,1]]
        #             self.demand[i,2] = self.Dest_idx[self.demand[i,2]]
        #         self.demand = np.array(self.demand.astype(float))
        #         '''
        #         self.supply = pkl.load(open('X:/Drive/dp/supply_final.pkl', 'rb')) #공급
        #         '''
        #         for i in range(len(self.supply)):
        #             self.supply[i,1] = self.Dest_idx[self.supply[i,1]]
        #         self.supply = np.array(self.supply.astype(float))
        #         '''
        #
        # # main에 state들 넘겨주는 역할
        # def get_data(self, demand_idx, supply_idx):
        #     return self.demand_hour[demand_idx, :], self.supply_hour[supply_idx, :]
        self.supply_hour = self.supply[self.supply[:, 0] == str(self.hour)]
        self.demand_hour = self.demand[self.demand[:, 0] == str(self.hour)]

    def reset(self):
        self.hour = self.hour + 1 if self.hour < 59 else 0
        print('reset hour : ', self.hour)
        if self.hour == 0:
            self.supply_hour = self.supply[self.supply[:, 0] == str(self.hour)]
        self.demand_hour = self.demand[self.demand[:, 0] == str(self.hour)]

        self.divide_district()

        self.OD_distance1 = self.get_od(1)  ##
        self.OD_distance2 = self.get_od(2)  ##
        self.OD_distance3 = self.get_od(3)  ##
        self.OD_distance4 = self.get_od(4)  ##

        self.PD_distance1 = self.get_pd(self.supply_hour1, self.demand_hour[self.demand_hour[:, 2] == '강남'])
        self.PD_distance2 = self.get_pd(self.supply_hour2, self.demand_hour[self.demand_hour[:, 2] == '강북'])

    # 승객과 공차 거리, OD거리, 수요 공급 비율
    def get_state(self):
        s, r, d = self.step(1)
        return s

    def get_pd(self, supply, demand):

        tmp = np.zeros(shape=(len(demand), len(supply)))

        for i in tqdm.tqdm(range(len(demand)), desc='PD_distance'):
            for j in range(len(supply)):
                tmp[i, j] = self.get_distance(float(demand[i, 4]),
                                              float(demand[i, 3]),
                                              float(supply[j, 3]),
                                              float(supply[j, 2]))
        return tmp

    def get_od(self, idx):

        tmp_demand = np.zeros((1, 1))
        if idx == 1:
            tmp_demand = self.demand_hour[self.demand_hour[:, 2] == '강남']
            tmp_demand = tmp_demand[tmp_demand[:, 1] == '강남']
        elif idx == 2:
            tmp_demand = self.demand_hour[self.demand_hour[:, 2] == '강남']
            tmp_demand = tmp_demand[tmp_demand[:, 1] == '강북']
        elif idx == 3:
            tmp_demand = self.demand_hour[self.demand_hour[:, 2] == '강북']
            tmp_demand = tmp_demand[tmp_demand[:, 1] == '강북']
        elif idx == 4:
            tmp_demand = self.demand_hour[self.demand_hour[:, 2] == '강북']
            tmp_demand = tmp_demand[tmp_demand[:, 1] == '강남']

        tmp = np.zeros(shape=(len(tmp_demand)))
        for i in tqdm.tqdm(range(len(tmp_demand)), desc='OD_distance'):
            tmp[i] = self.get_distance(
                float(tmp_demand[i, 4]),
                float(tmp_demand[i, 3]),
                float(tmp_demand[i, 6]),
                float(tmp_demand[i, 5]))
        return tmp

    def get_distance(self, lat1, lon1, lat2, lon2):
        s = (lat1, lon1)  # (lat, lon)
        d = (lat2, lon2)
        return haversine(s, d, unit='m')

    def get_NofP(self):
        return len(self.demand_hour)

    def get_NofD(self):
        return len(self.supply_hour)

    ######################################################### most important
    # def get_reward(self, pass_idx, driver_idx, before, offer):

    #########################################################

    def step(self, action):

        len_supply = self.get_NofD()
        len_driver = self.get_NofP()
        results = []
        new_supply = np.zeros((1, 1))

        # action이 0이면 그에 맞는 확률들과 index를 mat함수에 던져주고 mat 함수는 index에 맞게 수요, 공급, pd, od 계산해서 매칭시키고 그 새로운 공급들 return
        # -0.1 -> 0, 0 -> 1, +0.1 -> 2
        # 남,남 -> 1, 남,북 -> 2, 북,북 -> 3, 북,남 -> 4
        if action == 0:  # 원래 비율에서 -0.1, +0.1, 0, 0
            new_supply, avg_PD, avg_OD, num_matched = self.mat(0, 1)
            results.append([avg_PD, avg_OD, num_matched])
            new_supply2, avg_PD, avg_OD, num_matched = self.mat(1, 3)
            new_supply = np.vstack((new_supply, new_supply2))
            results.append([avg_PD, avg_OD, num_matched])
        elif action == 1:  # 원래 비율에서 0, 0, 0, 0
            new_supply, avg_PD, avg_OD, num_matched = self.mat(1, 1)
            results.append([avg_PD, avg_OD, num_matched])
            new_supply2, avg_PD, avg_OD, num_matched = self.mat(1, 3)
            new_supply = np.vstack((new_supply, new_supply2))
            results.append([avg_PD, avg_OD, num_matched])
        elif action == 2:  # 원래 비율에서 +0.1, -0.1, 0, 0
            new_supply, avg_PD, avg_OD, num_matched = self.mat(2, 1)
            results.append([avg_PD, avg_OD, num_matched])
            new_supply2, avg_PD, avg_OD, num_matched = self.mat(1, 3)
            new_supply = np.vstack((new_supply, new_supply2))
            results.append([avg_PD, avg_OD, num_matched])
        elif action == 3:  # 원래 비율에서 -0.1, +0.1, -0.1, +0.1
            new_supply, avg_PD, avg_OD, num_matched = self.mat(0, 1)
            results.append([avg_PD, avg_OD, num_matched])
            new_supply2, avg_PD, avg_OD, num_matched = self.mat(0, 3)
            new_supply = np.vstack((new_supply, new_supply2))
            results.append([avg_PD, avg_OD, num_matched])
        elif action == 4:  # 원래 비율에서 0, 0, -0.1, +0.1
            new_supply, avg_PD, avg_OD, num_matched = self.mat(1, 1)
            results.append([avg_PD, avg_OD, num_matched])
            new_supply2, avg_PD, avg_OD, num_matched = self.mat(0, 3)
            new_supply = np.vstack((new_supply, new_supply2))
            results.append([avg_PD, avg_OD, num_matched])
        elif action == 5:  # 원래 비율에서 +0.1, -0.1, -0.1, +0.1
            new_supply, avg_PD, avg_OD, num_matched = self.mat(2, 1)
            results.append([avg_PD, avg_OD, num_matched])
            new_supply2, avg_PD, avg_OD, num_matched = self.mat(0, 3)
            new_supply = np.vstack((new_supply, new_supply2))
            results.append([avg_PD, avg_OD, num_matched])
        elif action == 6:  # 원래 비율에서 -0.1, +0.1, +0.1, -0.1
            new_supply, avg_PD, avg_OD, num_matched = self.mat(0, 1)
            results.append([avg_PD, avg_OD, num_matched])
            new_supply2, avg_PD, avg_OD, num_matched = self.mat(2, 3)
            new_supply = np.vstack((new_supply, new_supply2))
            results.append([avg_PD, avg_OD, num_matched])
        elif action == 7:  # 원래 비율에서 0, 0, +0.1, -0.1
            new_supply, avg_PD, avg_OD, num_matched = self.mat(1, 1)
            results.append([avg_PD, avg_OD, num_matched])
            new_supply2, avg_PD, avg_OD, num_matched = self.mat(2, 3)
            new_supply = np.vstack((new_supply, new_supply2))
            results.append([avg_PD, avg_OD, num_matched])
        elif action == 8:  # 원래 비율에서 +0.1, -0.1, +0.1, -0.1
            new_supply, avg_PD, avg_OD, num_matched = self.mat(2, 1)
            results.append([avg_PD, avg_OD, num_matched])
            new_supply2, avg_PD, avg_OD, num_matched = self.mat(2, 3)
            new_supply = np.vstack((new_supply, new_supply2))
            results.append([avg_PD, avg_OD, num_matched])

        self.supply_hour = new_supply
        self.reset()

        # make reward
        reward = (results[0][1] + results[1][1] - (results[0][0] + results[1][0]))*(100/132)  # matched * (OD - PD)
        states = [len(self.supply_hour1) / len(self.supply_hour), len(self.supply_hour2) / len(self.supply_hour),
                  len(self.demand_hour1) / len(self.demand_hour), len(self.demand_hour2) / len(self.demand_hour),
                  len(self.demand_hour3) / len(self.demand_hour), len(self.demand_hour4) / len(self.demand_hour),
                  np.mean(np.array(self.OD_distance1)) / 40000, np.mean(np.array(self.OD_distance2)) / 40000,
                  np.mean(np.array(self.OD_distance3)) / 40000, np.mean(np.array(self.OD_distance4)) / 40000,
                  np.mean(np.array(self.PD_distance1)) / 20000, np.mean(np.array(self.PD_distance2)) / 20000
                  ]
        done = 1 if self.hour == 59 else 0
        return states, reward, done

    ########################################################

    def mat(self, percent_idx, idx):
        """ Return : 새로운 supply, 평균 OD 거리, 평균 PD 거리, 매칭 수 """
        # param percent로 index 비율 맞춰서 sampling함.
        # sampling한 index로 pd 구함
        # '' -> get_od로 두개의 od 구함
        matrix = np.zeros((1, 1))
        p1, p2 = 0, 0
        if idx == 1:
            p1 = len(self.demand_hour1) / (len(self.demand_hour1) + len(self.demand_hour2))
            p2 = 1 - p1
        elif idx == 3:
            p1 = len(self.demand_hour3) / (len(self.demand_hour3) + len(self.demand_hour4))
            p2 = 1 - p1

        if percent_idx == 0:
            p1 -= 0.1
            p2 += 0.1
        elif percent_idx == 2:
            p1 += 0.1
            p2 -= 0.1

        # print('p1 {}  p2 {}'.format(p1, p2))

        if idx == 1:
            tmp_idx1 = random.sample(range(0, len(self.supply_hour1) - 1), int(p1 * len(self.supply_hour1)))
            supply1 = self.supply_hour1[tmp_idx1]
            supply2 = np.delete(self.supply_hour1, tmp_idx1, axis=0)
            PD1 = self.get_pd(supply1, self.demand_hour1)
            PD2 = self.get_pd(supply2, self.demand_hour2)
            OD1 = self.OD_distance1
            OD2 = self.OD_distance2
            next_supply1, sum_PD1, sum_OD1, num_matched = self.make_mat(supply1, self.demand_hour1, PD1, OD1)
            next_supply2, sum_PD2, sum_OD2, num_matched2 = self.make_mat(supply2, self.demand_hour2, PD2, OD2)
            return np.vstack((next_supply1, next_supply2)), sum_PD1 + sum_PD2, sum_OD1 + sum_OD2, num_matched + num_matched2
        elif idx == 3:
            tmp_idx1 = random.sample(range(0, len(self.supply_hour2) - 1), int(p1 * len(self.supply_hour2)))
            supply1 = self.supply_hour2[tmp_idx1]
            supply2 = np.delete(self.supply_hour2, tmp_idx1, axis=0)
            PD1 = self.get_pd(supply1, self.demand_hour3)
            PD2 = self.get_pd(supply2, self.demand_hour4)
            OD1 = self.OD_distance3
            OD2 = self.OD_distance4
            next_supply1, sum_PD1, sum_OD1, num_matched = self.make_mat(supply1, self.demand_hour3, PD1, OD1)
            next_supply2, sum_PD2, sum_OD2, num_matched2 = self.make_mat(supply2, self.demand_hour4, PD2, OD2)
            return np.vstack((next_supply1, next_supply2)), sum_PD1 + sum_PD2, sum_OD1 + sum_OD2, num_matched + num_matched2

    ########################################################

    def make_mat(self, supply, demand, PD, matrix):
        D_Matrix = PD  # n x m
        D_Matrix -= D_Matrix.min()  # 0 ~
        D_Matrix = D_Matrix / D_Matrix.max()  # 0 ~ 1
        D_Matrix = 1 - D_Matrix
        for i in range(len(D_Matrix)):
            D_Matrix[i, :] = D_Matrix[i, :] * matrix[i]
            # PD가 작을수록 유리하고 OD가 클수록 그 승객에 대한 배차 확률이 높게함

        for i in range(D_Matrix.shape[0]):
            for j in range(D_Matrix.shape[1]):
                if PD[i][j] > 10000:
                    D_Matrix[i][j] = 0
        matched_passengers = []
        matched_drivers = []
        total_PD = 0.
        total_OD = 0.
        while True:
            # print(len(matched_passengers))
            # print(D_Matrix.shape)
            if np.max(D_Matrix) == 0:
                print('supply {} matched_driv {}'.format(len(supply), len(matched_drivers)))
                newMatrix = np.delete(supply, matched_drivers, axis=0)
                # print('matched {}'.format(matched_drivers))
                # print('before {}'.format(len(newMatrix)))
                for k in range(len(matched_passengers)):
                    tmp = self.supply_hour[0]
                    tmp[1] = demand[k, 1]
                    tmp[2] = demand[k, 5]
                    tmp[3] = demand[k, 6]

                    newMatrix = np.vstack((newMatrix, tmp))
                    total_OD += matrix[k]
                    total_PD += PD[matched_passengers[k], matched_drivers[k]]
                newMatrix[:, 0] = str(self.hour + 1)
                # print('after {}'.format(len(newMatrix)))

                # print(len(matched_passengers))
                return newMatrix, total_PD, total_OD, len(matched_passengers)
            else:
                max_idx = np.argmax(D_Matrix)
                tmp_pass_idx = int(max_idx / D_Matrix.shape[1])
                tmp_dri_idx = max_idx % D_Matrix.shape[1]
                matched_passengers.append(tmp_pass_idx)
                matched_drivers.append(tmp_dri_idx)
                D_Matrix[tmp_pass_idx, :] = 0
                D_Matrix[:, tmp_dri_idx] = 0


# env=Env()
# #env.supply = supply
# #env.demand = demand
# #env.PD_distance = pd
# env.reset(0)
from tensorflow.keras.layers import Dense, BatchNormalization
import random


# agent tmp
class DQNAgent:
    def __init__(self):
        self.load_model = False
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 12

        # DQN HyperParameter
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.batch_size = 64

        ### test code ( 1000 -> 64 )
        self.train_start = 64
        ### test code

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=5000)

        # 모델과 타깃 모델 생성
        if self.load_model:
            self.model = load_model("./save_model/dqn_matching.h5")
            self.target_model = load_model("./save_model/dqn_matching.h5")

        else:
            self.model = self.build_model()
            self.target_model = self.build_model()

        # 타킷 모델 초기화
        self.update_target_model()

    # 상태가 입력 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            # print(q_value)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


import matplotlib.pyplot as plt

EPISODES = 1000
if __name__ == "__main__":

    demand = pkl.load(open('demand_gangnam_gangbook.pkl', 'rb'))  # 수요
    supply = pkl.load(open('supply_gangnam_gangbook.pkl', 'rb'))

    plt.style.use('seaborn-whitegrid')

    agent = DQNAgent()
    env = Env(supply, demand)
    state_size = 12

    scores, episodes = [], []
    actions = []

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.get_state()
        state = np.reshape(state, [1, state_size])

        while not done:

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            actions.append(action)

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done = env.step(action)
            # print("=============")
            # print("state : {}, reward : {}, done: {} info : {}".format(next_state, reward,done,info))
            next_state = np.reshape(next_state, [1, state_size])

            print("state : {}, reward : {}, done: {} ".format(next_state, reward, done))

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/matching.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                agent.model.save("./save_model/dqn_matching.h5")

    data = actions
    with open('actions.pkl', 'wb') as f:
        pkl.dump(data, f)

