import numpy as np
import pickle as pkl
import copy
import queue
import sys
import tqdm
from collections import deque
from haversine import haversine, haversine_vector
from sklearn.linear_model import LinearRegression
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, save_model
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

gu_list = []
total_result_reward = []
state_size=0
numsection = 0
second = 0
supply_minus_demand = []
section_dict = {}   # 숫자 : 행정구
reverse_section_dict = {}
ORR_list = []

def find_section_number(section_string):
  for num, string in section_dict.items(): #mydict에 아이템을 하나씩 접근해서, key, value를 각각 name, age에 저장
    if section_string == string:
        return num

def get_pd(supply, demand):
    if len(demand)==0 or len(supply)==0 :
        return []
    pd = []
    demand =[(float(demand[i, 4]), float(demand[i, 3])) for i in range(len(demand))]
    for j in range(len(demand)):
        tmp_supply = [(float(supply[i,3]), float(supply[i,2])) for i in range(len(supply))]
        tmp_demand = [demand[j]]*len(supply)
        pd.append(haversine_vector(tmp_demand, tmp_supply))

    return np.array(pd)

def get_od(demand):
    if len(demand)==0 :
        return []
    origin = [(float(demand[i, 4]), float(demand[i, 3])) for i in range(len(demand))]
    dest = [(float(demand[i, 6]), float(demand[i, 5])) for i in range(len(demand))]

    return np.array(haversine_vector(origin, dest))

class Gu:
    def __init__(self, idx):
        global state_size

        self.idx = idx #gu의 list idx
        self.hour = 0
        self.demand = None        #demand 한번에 해서 하지 않고 구마다 저장.
        self.demands = []
        self.supply = None
        self.PD_distance = None  # Passenger들과 Driver들 사이의 거리
        self.OD_distance = None
        self.PD_distances = []  # Passenger들과 Driver들 사이의 거리
        self.OD_distances = [] # OD도 list로 관리하는게 OD 구 단위로 matching 시킬때 편할듯
        self.percents = [] #원래 demand의 percent들
        self.priority = {} # dictionary 0:'강남구'
        self.demand_history = queue.Queue()   # queue

        self.line_fitter = LinearRegression()

        self.load_model = False
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [i for i in range(numsection + 1)]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = state_size

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

    def set_hour(self, hour):
        self.hour = hour

    def set_demand(self):  # percent 정리, priority 정리, OD, PD 정리
        global section_dict, numsection
        self.demands = []
        self.percents = []
        demands_length = len(self.demand)  # demand 아예 없는 경우도 대비
        for i in range(numsection):
            if demands_length>0:
                tmp_demand = self.demand[self.demand[:,1]==section_dict[i]]
                if tmp_demand.size != 0:
                    self.demands.append(tmp_demand)
                else:
                    self.demands.append(np.array([]))
            else:
                self.demands.append(np.array([]))

        for i in range(numsection):
            if demands_length != 0:
                self.percents.append(len(self.demand[self.demand[:, 2] == section_dict[i]]) / demands_length)
            else:
                self.percents.append(0)
        self.percents = np.array(self.percents)

        self.OD_distance = get_od(self.demand)
        self.PD_distance = get_pd(self.supply, self.demand)

        self.OD_distances = []
        self.PD_distances = []
        for i in range(numsection):
            self.OD_distances.append(get_od(self.demands[i]))
            self.PD_distances.append(get_pd(self.supply, self.demands[i]))

    def build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        for i in range(numsection):
            state.append(np.mean(self.OD_distances[i]))
        total_demand = len(self.demand)
        for i in range(numsection):
            try:
                state.append(len(self.demands[i]) / total_demand)
            except:
                state.append(0)
        state = np.reshape(np.array(state), [1,state_size])
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.action_size), state]
        else:
            q_value = self.model.predict(state)
            # print(q_value)
            return [np.argmax(q_value[0]), state]

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

    def matching(self, action, is_sim=True):  # reward, new supply return

        global section_dict, numsection
        if is_sim == False:
            self.demand_history.put(len(self.demand))
        total_OD = total_PD = flag = 0
        newMatrix = []  # 몇 timestep뒤에 시뮬레이션에 추가될 택시
        percents = copy.deepcopy(self.percents)  # supply 비율 맞추기
        ORR = 0

        if action < numsection:  # 해당 인덱스 0.1 추가 마지막에는 그냥 퍼센트 그대로
            percents = percents * 0.9
            percents[action] += 0.1

        if len(self.supply) == 0:
            tmp_supply = copy.deepcopy(self.supply)
            tmp_supply[:, 0] = str(int(tmp_supply[0, 0]) + 1)
            return 0, tmp_supply

        ##################################### matching 시작 #####################################

        unmatched_demand = []
        matched_demand = []
        matched_supply = []  # matching 된 supply index들
        matched_supply_arr = [0] * numsection  # [0,0 ,,,,0]
        tmp_demand = copy.deepcopy(self.demand)
        tmp_od = copy.deepcopy(self.OD_distance)
        tmp_pd = copy.deepcopy(self.PD_distance)
        tmp_pd = np.where(tmp_pd > 8, 0, tmp_pd)  # PD distance 8km(10min) 넘으면 매칭 안되게
        tmp_pd[:, matched_supply] = np.max(tmp_pd)  # matching된 supply들 PD 0으로
        matching_matrix = 1 - tmp_pd / np.max(tmp_pd)
        section_supply_len = np.array(self.percents) * len(self.demand)

        if len(self.supply) == 1 and len(matching_matrix) == 1:
            matching_matrix = matching_matrix.reshape((matching_matrix.shape[1], 1))

        for j in range(len(matching_matrix)):
            matching_matrix[j] = matching_matrix[j] * tmp_od[j]
            matching_matrix[j] = matching_matrix[j] * self.percents[
                reverse_section_dict[tmp_demand[j, 1]]]  # consider out degree
        # PD가 작을수록 유리하고 OD가 클수록 그 승객에 대한 배차 확률이 높게함

        while True:
            if np.max(matching_matrix) == 0 or len(matching_matrix) == len(matched_demand):  # matching 끝
                break
            else:
                max_idx = np.argmax(
                    matching_matrix)  # argmax는 flatten해서 하니까 아래 두줄에서 행과 열로 바꿔줌                    ########### matching ###########
                tmp_pass_idx = int(max_idx / matching_matrix.shape[1])
                tmp_dri_idx = max_idx % matching_matrix.shape[1]
                if matched_supply_arr[reverse_section_dict[tmp_demand[tmp_pass_idx, 1]]] == section_supply_len[reverse_section_dict[tmp_demand[tmp_pass_idx, 1]]]:  # 할당해야할 supply들 다 매칭되면 unmatching시킴
                    unmatched_demand.append(tmp_demand[tmp_pass_idx])
                    matching_matrix[tmp_pass_idx, :] = 0
                    continue
                matched_demand.append(tmp_pass_idx)
                matched_supply.append(tmp_dri_idx)
                matched_supply_arr[reverse_section_dict[tmp_demand[tmp_pass_idx, 1]]] += 1
                matching_matrix[tmp_pass_idx, :] = 0
                matching_matrix[:, tmp_dri_idx] = 0
                total_OD += tmp_od[tmp_pass_idx]
                total_PD += tmp_pd[tmp_pass_idx, tmp_dri_idx]

        # matching 안된 demand들. matching 안된 supply는 밑에서 한번에 할 수 있음
        if len(unmatched_demand) == 0:
            unmatched_demand = np.delete(tmp_demand, matched_demand, 0)
        else:
            unmatched_demand = np.vstack((unmatched_demand, np.delete(tmp_demand, matched_demand, 0)))

        for k in range(len(matched_demand)):  # 매칭이된 택시들을 이동시키는 for 문
            if len(matched_supply) > 0 and len(matched_demand) > 0 and matched_demand[k] < len(tmp_demand):
                tmp = [0] * 4
                matched_supply_idx = len(matched_supply) - len(matched_demand) + k - 1
                # print('len ms {} len md {} mdk {} len tmppd {} msi {}'.format(len(matched_supply),len(matched_demand), matched_demand[k], tmp_pd.shape, matched_supply_idx))
                tmp[0] = int(self.supply[matched_supply[k], 0]) + int(
                    tmp_pd[matched_demand[k], matched_supply[matched_supply_idx]] / 0.4) + int(
                    tmp_od[matched_demand[k]] / 0.4) + random.randint(0, 10)  # 30초 400m
                if tmp[0] == self.hour:
                    tmp[0] = self.hour + 1
                tmp[1] = tmp_demand[matched_demand[k], 1]
                tmp[2] = tmp_demand[matched_demand[k], 5]
                tmp[3] = tmp_demand[matched_demand[k], 6]
                newMatrix.append(tmp)
            else:
                break
        if is_sim == False:
            ORR += float(len(matched_demand)) / len(self.demand)

        ################# supply, demand가 남으면 나머지 matching
        if len(matched_supply) < len(self.supply) and len(unmatched_demand) > 0:
            matched_demand = []
            tmp_demand = unmatched_demand
            tmp_od = get_od(unmatched_demand)
            tmp_pd = get_pd(self.supply, unmatched_demand)
            tmp_pd = np.where(tmp_pd > 8, 0, tmp_pd)  # PD distance 8km(10min) 넘으면 매칭 안되게
            tmp_pd[:, matched_supply] = np.max(tmp_pd)  # matching된 supply들 PD 0으로
            matching_matrix = 1 - tmp_pd / np.max(tmp_pd)

            for j in range(len(matching_matrix)):
                matching_matrix[j] = matching_matrix[j] * tmp_od[j]
                matching_matrix[j] = matching_matrix[j] * self.percents[
                    reverse_section_dict[tmp_demand[j, 1]]]  # consider out degree

            while True:
                if np.max(matching_matrix) == 0 or len(matching_matrix) == len(matched_demand):  # matching 끝
                    break
                else:
                    max_idx = np.argmax(matching_matrix)  # argmax는 flatten해서 하니까 아래 두줄에서 행과 열로 바꿔줌
                    tmp_pass_idx = int(max_idx / matching_matrix.shape[1])
                    tmp_dri_idx = max_idx % matching_matrix.shape[1]
                    matched_demand.append(tmp_pass_idx)
                    matched_supply.append(tmp_dri_idx)
                    matching_matrix[tmp_pass_idx, :] = 0
                    matching_matrix[:, tmp_dri_idx] = 0
                    total_OD += tmp_od[tmp_pass_idx]
                    total_PD += tmp_pd[tmp_pass_idx, tmp_dri_idx]

            for k in range(len(matched_demand)):  # 매칭이된 택시들을 이동시키는 for 문
                if len(matched_supply) > 0 and len(matched_demand) > 0 and matched_demand[k] < len(tmp_demand):
                    tmp = [0] * 4
                    matched_supply_idx = len(matched_supply) - len(matched_demand) + k - 1
                    # print('len ms {} len md {} mdk {} len tmppd {} msi {}'.format(len(matched_supply),len(matched_demand), matched_demand[k], tmp_pd.shape, matched_supply_idx))
                    tmp[0] = int(self.supply[matched_supply[k], 0]) + int(
                        tmp_pd[matched_demand[k], matched_supply[matched_supply_idx]] / 0.4) + int(
                        tmp_od[matched_demand[k]] / 0.4) + random.randint(0, 10)  # 30초 400m
                    if tmp[0] == self.hour:
                        tmp[0] = self.hour + 1
                    tmp[1] = tmp_demand[matched_demand[k], 1]
                    tmp[2] = tmp_demand[matched_demand[k], 5]
                    tmp[3] = tmp_demand[matched_demand[k], 6]
                    newMatrix.append(tmp)
                else:
                    break
            if is_sim == False:
                ORR += float(len(matched_demand)) / len(self.demand)
        if is_sim == False:
            ORR_list.append(ORR)

        ##################################### matching 끝 #####################################

        reward = abs((total_OD - total_PD) / 0.132 * 100)  # 132m당 100원

        if len(self.supply) == 0:
            return 0, np.array([])
        next_supply = np.delete(self.supply, matched_supply, axis=0)  # 매칭이 끝나고 남은 택시들 (바로 뒤 timestep에서 쓸거라 따로 뺴놓음)
        next_supply[:, 0] = str(self.hour + 1)

        if len(matched_supply) == 0:
            return 0, next_supply

        if len(matched_supply) != 0 and len(newMatrix) > 0:
            if len(next_supply) == 0:
                next_supply = np.array(newMatrix)
            else:
                next_supply = np.vstack((next_supply, np.array(newMatrix)))
        return reward, next_supply


class Platform:  # 역할: OD별, PD별로 demand, supply 정리해서 gu에 넘겨주기, simulation, matching
    def __init__(self, case):

        self.numsection = 0
        self.second = 0
        self.case = case
        self.episode_time = 720  # 3 hours 진짜 실험시 6시간

        if case == 0:  # full greedy
            self.numsection = 25
            self.second = 10
        elif case == 1:
            self.numsection = 25
            self.second = 30
        elif case == 2:
            self.numsection = 8
            self.second = 30
        elif case == 3:
            self.numsection = 4
            self.second = 30
        elif case == 4:
            self.numsection = 2
            self.second = 30
        elif case == 5:
            self.numsection = 1
            self.second = 30

        self.demand_hour = []
        self.supply_hour = []
        self.hour = 0
        self.total_reward = 0
        self.section_dict = {}
        self.gu_list = []

        if self.numsection == 25:
            self.section_dict[0] = '도봉구'
            self.section_dict[1] = '노원구'
            self.section_dict[2] = '강북구'
            self.section_dict[3] = '성북구'
            self.section_dict[4] = '중랑구'
            self.section_dict[5] = '은평구'
            self.section_dict[6] = '종로구'
            self.section_dict[7] = '서대문구'
            self.section_dict[8] = '동대문구'
            self.section_dict[9] = '중구'
            self.section_dict[10] = '성동구'
            self.section_dict[11] = '광진구'
            self.section_dict[12] = '마포구'
            self.section_dict[13] = '용산구'
            self.section_dict[14] = '강동구'
            self.section_dict[15] = '송파구'
            self.section_dict[16] = '강남구'
            self.section_dict[17] = '서초구'
            self.section_dict[18] = '동작구'
            self.section_dict[19] = '관악구'
            self.section_dict[20] = '영등포구'
            self.section_dict[21] = '강서구'
            self.section_dict[22] = '양천구'
            self.section_dict[23] = '구로구'
            self.section_dict[24] = '금천구'
        elif self.numsection == 8:
            self.section_dict[0] = '동북'
            self.section_dict[1] = '동'
            self.section_dict[2] = '강서'
            self.section_dict[3] = '중'
            self.section_dict[4] = '남'
            self.section_dict[5] = '동남'
            self.section_dict[6] = '서'
            self.section_dict[7] = '서남'
        elif self.numsection == 4:
            self.section_dict[0] = '동북'
            self.section_dict[1] = '서북'
            self.section_dict[2] = '동남'
            self.section_dict[3] = '서남'
        elif self.numsection == 2:
            self.section_dict[0] = '강북'
            self.section_dict[1] = '강남'
        elif self.numsection == 1:
            self.section_dict[0] = '서울'

        global state_size
        state_size = 4 * self.numsection

        # for i in range(self.numsection):
        #     self.gu_list.append(Gu(i))

        global numsection, section_dict, second, reverse_section_dict
        numsection = self.numsection
        section_dict = self.section_dict
        for i in range(numsection):
            reverse_section_dict[section_dict[i]] = i
        second = self.second

        self.load_data()

        self.divide_district()

    def divide_district(self):
        for i in range(self.episode_time):  # 행 시간 열 지역에 따른 수요, 공급
            tmp_demand = self.demand[self.demand[:, 0] == str(i)]
            tmp_supply = self.supply[self.supply[:, 0] == str(i)]
            tmp = []
            for j in range(self.numsection):
                tmp.append(tmp_demand[tmp_demand[:, 2] == self.section_dict[j]])
            self.demand_hour.append(tmp)

            tmp_len = 0
            tmp = []
            for j in range(self.numsection):
                if i < 15:
                    tmp.append(tmp_supply[tmp_supply[:, 1] == self.section_dict[j]])
                    # print('supply {} time {}idx {}'.format(len(tmp_supply[tmp_supply[:, 1] == self.section_dict[j]]), i, j))
                    tmp_len += len(tmp_supply[tmp_supply[:, 1] == self.section_dict[j]])
                else:
                    tmp.append(np.reshape(np.array([]),(0,4)))
            self.supply_hour.append(tmp)
            # print('total len {} time {}'.format(tmp_len, i))


        self.demand_hour = np.array(self.demand_hour,dtype=object)
        self.supply_hour = np.array(self.supply_hour,dtype=object)

        # for i in range(20):

        # print(self.demand_hour.shape, self.supply_hour.shape)

    def divide_supply(self, supply, sim=True):
        global section_dict, numsection
        # print('divide_supply!!! supply {}'.format(supply.shape))

        if sim == False:
            how_long = 0
            for i in range(len(supply)):
                how_long += (int(supply[i,0]) - self.hour)
                if int(supply[i,0]) < self.episode_time:
                    self.supply_hour[int(supply[i,0]), find_section_number(supply[i,1])] = \
                        np.vstack((self.supply_hour[int(supply[i, 0]), find_section_number(supply[i, 1])], supply[i]))
            # print('average how long does it take {}'.format(how_long/len(supply)))

        if sim == True:
            new_supply = copy.deepcopy(self.supply_hour)
            for i in range(len(supply)):
                if int(supply[i, 0]) < self.episode_time:
                    new_supply[int(supply[i, 0]), find_section_number(supply[i, 1])] = \
                        np.vstack((new_supply[int(supply[i, 0]), find_section_number(supply[i, 1])], supply[i]))
            # print('divide new supply {}'.format(new_supply.shape))
            return new_supply

    def load_data(self):
        print('Loading Data')
        #########################################수정해야함##########################################
        self.demand = pkl.load(
            open('demand_final_' + str(self.numsection) + 'section_' + str(self.second) + 'second_test.pkl',
                 'rb'))  # 수요
        self.supply = pkl.load(
            open('supply_final_' + str(self.numsection) + 'section_' + str(self.second) + 'second_test_18000.pkl',
                 'rb'))  # 공급

    def get_state(self):    #DQN 돌리기 위해서 state 만들어주기 (list 형태로 반환)
        total_demand = 0
        for i in range(self.numsection):
            total_demand += len(self.demand_hour[self.hour, i])
        state = []
        for i in range(self.numsection):
            state.append(len(self.demand_hour[self.hour, i]) / total_demand)

        total_supply = 0
        for i in range(self.numsection):
            total_supply += len(self.supply_hour[self.hour, i])
        for i in range(self.numsection):
            state.append(len(self.supply_hour[self.hour, i]) / total_supply)

        return state

    def step(self):  # gu의 is_sim false로 하고 matching 시키는 함수
        global supply_minus_demand, total_result_reward

        reward = 0
        old_state=[]
        old_actions = []
        for i in range(self.episode_time):
            # old_state=[]
            new_state = []
            best_actions = []
            now_supply = 0
            distrib = 0.
            for j in range(numsection):
                self.gu_list[j].supply = self.supply_hour[self.hour][j]
                self.gu_list[j].demand = self.demand_hour[self.hour][j]
                self.gu_list[j].set_demand()
                now_supply += len(self.supply_hour[self.hour, j])
                state = self.get_state()
                tmp_action_state = self.gu_list[j].get_action(state)
                best_actions.append(tmp_action_state[0])
                new_state.append(tmp_action_state[1])

            tmplen = 0
            for j in range(numsection):
                tmplen += len(self.demand_hour[self.hour, j])
                if new_state[j][0][j] > new_state[j][0][j+numsection]:  #demand의 비율의 크기가 supply보다 크다면
                    distrib += new_state[j][0][j + numsection] / new_state[j][0][j]
                else:
                    distrib += new_state[j][0][j] / new_state[j][0][j+numsection]
            supply_minus_demand.append(now_supply-tmplen)

            tmplen = 0
            # print('best actions {}'.format(best_actions))
            for j in range(numsection):
                # print('REAL!!!!!!!!!!!!')
                r, s = self.gu_list[j].matching(best_actions[j], False)
                tmplen += len(s)
                reward += r
                self.divide_supply(s, sim=False)
                if i > 0:
                    if i == self.episode_time - 1:
                        self.gu_list[j].append_sample(old_state[j], old_actions[j], r * distrib, new_state[j], True)
                        self.gu_list[j].update_target_model()
                    else:
                        self.gu_list[j].append_sample(old_state[j], old_actions[j], r * distrib, new_state[j], False)

                if len(self.gu_list[j].memory) >= self.gu_list[j].train_start:
                    self.gu_list[j].train_model()

            old_state = copy.deepcopy(new_state)
            old_actions = copy.deepcopy(best_actions)
            self.hour += 1
        total_result_reward.append(reward)
        print('result reward {}'.format(reward))
        print('mean ORR {}'.format(sum(ORR_list) / float(len(ORR_list))))

if __name__ == "__main__":

    plt.style.use('seaborn-whitegrid')

    case = int(input('case : '))
    episode = 300

    env = Platform(case)
    for i in range(numsection):
        gu_list.append(Gu(i))
    env.gu_list = gu_list
    for i in tqdm.tqdm(range(episode)):
        env.step()
        env = Platform(case)
        env.gu_list = gu_list

    x = np.arange(len(supply_minus_demand))
    plt.figure(1)
    plt.plot(x,supply_minus_demand)
    plt.title('Number of supply - Number of demand in every each time')
    plt.xlabel('time (1time = 30s)')
    plt.ylabel('count')
    plt.savefig('./graph/new_graph/matching_RL_' + str(numsection) + 'section_' + str(second) + 'second_supply_minus_demand.png')

    x = np.arange(len(total_result_reward))
    plt.figure(2)
    plt.plot(x, total_result_reward)
    plt.title('score per episode')
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.savefig('./graph/new_graph/matching_RL_' + str(numsection) + 'section_' + str(second) + 'second_score.png')

