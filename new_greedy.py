import numpy as np
import random
import pickle as pkl
import pylab
import copy
import queue
import sys
import tqdm
from haversine import haversine, haversine_vector
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
#
simulation_level = 3
mean_profit = 0.
mean_revenue = 0.
numsection = 0
second = 0
supply_minus_demand = []
section_dict = {}   # 숫자 : 행정구
reverse_section_dict = {}
select_action = [0,0,0]
ORR_list = []
sum_OD = 0.
sum_PD = 0.
num_matched = 0

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
        self.idx = idx #gu의 list idx
        self.hour = 0
        self.demand = None        #demand 한번에 해서 하지 않고 구마다 저장.
        self.supply = None
        self.PD_distance = None  # Passenger들과 Driver들 사이의 거리
        self.OD_distance = None
        self.percents = [] #원래 demand의 percent들
        self.demand_history = queue.Queue()   # queue

        self.line_fitter = LinearRegression()

    def set_hour(self, hour):
        self.hour = hour

    def set_demand(self):   #percent 정리, priority 정리, OD, PD 정리
        global section_dict, numsection
        self.percents = []
        demands_length = len(self.demand)   # demand 아예 없는 경우도 대비
        for i in range(numsection):
            if demands_length != 0 :
                self.percents.append(len(self.demand[self.demand[:,2] == section_dict[i]]) / demands_length)
            else:
                self.percents.append(0)
        self.percents = np.array(self.percents)

        self.OD_distance = get_od(self.demand)
        self.PD_distance = get_pd(self.supply, self.demand)

    def approximate_demand(self, level):
        q_list = list(self.demand_history.queue)
        if len(q_list) >= 20:
            q_list = q_list[len(q_list)-20:]
        q_leng = len(q_list)

        demand_predicted = 0
        if q_leng > 2:
            idx_list = [i for i in range(q_leng)]
            self.line_fitter.fit(np.array(idx_list).reshape(-1,1), q_list)
            demand_predicted = round(self.line_fitter.predict(np.array(q_leng+level).reshape(-1,1))[0])
        elif q_leng == 0:
            demand_predicted = len(self.demand)
        else:
            demand_predicted = q_list[0]

        len_demand = len(self.demand)

        if len_demand > 0 and demand_predicted>0:  #전의 demand가 있다면
            new_demand=None
            for i in range(int(demand_predicted)):
                if i == 0:
                    new_demand = np.array(self.demand[random.randint(0, len_demand-1)]).reshape((1,7))
                else:
                    new_demand=np.vstack((new_demand, self.demand[random.randint(0, len_demand-1)]))
                new_demand[:,0] = self.hour
            self.demand = new_demand
        else:               #전의 demand가 없다면
            self.demand=np.array([])

        self.set_demand()

    def matching(self, action, is_sim = True): # reward, new supply return

        global section_dict, numsection, sum_OD, num_matched, mean_revenue, sum_PD
        if is_sim == False:
            self.demand_history.put(len(self.demand))
        total_OD = total_PD = flag = 0
        newMatrix = []  # 몇 timestep뒤에 시뮬레이션에 추가될 택시
        percents = copy.deepcopy(self.percents)         #supply 비율 맞추기
        ORR = 0

        if action < numsection:    #해당 인덱스 0.1 추가 마지막에는 그냥 퍼센트 그대로
            percents = percents * 0.9
            percents[action] += 0.1

        if len(self.supply) == 0 or len(self.demand) == 0:
            tmp_supply = copy.deepcopy(self.supply)
            if len(self.supply) > 0 :
                tmp_supply[:,0] = str(int(tmp_supply[0,0])+1)
            return 0, tmp_supply

        ##################################### matching 시작 #####################################

        unmatched_demand = []
        matched_demand = []
        matched_supply = [] # matching 된 supply index들
        matched_supply_arr = [0] * numsection       #[0,0 ,,,,0]
        tmp_demand = copy.deepcopy(self.demand)
        tmp_od = copy.deepcopy(self.OD_distance)
        tmp_pd = copy.deepcopy(self.PD_distance)
        # try:
        tmp_pd = np.where(tmp_pd > 8, 0, tmp_pd)  # PD distance 8km(10min) 넘으면 매칭 안되게
        # except TypeError:
        #     print(tmp_pd, self.supply, self.demand)
        tmp_pd[:,matched_supply] = np.max(tmp_pd)    # matching된 supply들 PD 0으로
        matching_matrix = 1 - tmp_pd/np.max(tmp_pd)
        section_supply_len = np.array(self.percents) * len(self.demand)

        if len(self.supply) == 1 and len(matching_matrix)==1:
            matching_matrix = matching_matrix.reshape((matching_matrix.shape[1],1))

        for j in range(len(matching_matrix)):
            matching_matrix[j] = matching_matrix[j] * tmp_od[j]
            matching_matrix[j] = matching_matrix[j] * self.percents[reverse_section_dict[tmp_demand[j,1]]]              # consider out degree
        # PD가 작을수록 유리하고 OD가 클수록 그 승객에 대한 배차 확률이 높게함

        while True:
            if np.max(matching_matrix) == 0 or len(matching_matrix)==len(matched_demand):   #matching 끝
                break
            else:
                max_idx = np.argmax(matching_matrix)    # argmax는 flatten해서 하니까 아래 두줄에서 행과 열로 바꿔줌                    ########### matching ###########
                tmp_pass_idx = int(max_idx / matching_matrix.shape[1])
                tmp_dri_idx = max_idx % matching_matrix.shape[1]
                if matched_supply_arr[reverse_section_dict[tmp_demand[tmp_pass_idx,1]]] == section_supply_len[reverse_section_dict[tmp_demand[tmp_pass_idx,1]]]:    #할당해야할 supply들 다 매칭되면 unmatching시킴
                    unmatched_demand.append(tmp_demand[tmp_pass_idx])
                    matching_matrix[tmp_pass_idx,:] = 0
                    continue
                matched_demand.append(tmp_pass_idx)
                matched_supply.append(tmp_dri_idx)
                matched_supply_arr[reverse_section_dict[tmp_demand[tmp_pass_idx,1]]] += 1
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
                tmp[0] = int(self.supply[matched_supply[k], 0]) + int(tmp_pd[matched_demand[k], matched_supply[matched_supply_idx]] / 0.4) + int(tmp_od[matched_demand[k]] / 0.4) + random.randint(0,10)  # 30초 400m
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
            num_matched += len(matched_demand)

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
                matching_matrix[j] = matching_matrix[j] * self.percents[reverse_section_dict[tmp_demand[j, 1]]]  # consider out degree

            while True:
                if np.max(matching_matrix) == 0 or len(matching_matrix) == len(matched_demand):   #matching 끝
                    break
                else:
                    max_idx = np.argmax(matching_matrix)    # argmax는 flatten해서 하니까 아래 두줄에서 행과 열로 바꿔줌
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
                    tmp[0] = int(self.supply[matched_supply[k], 0]) + int(tmp_pd[matched_demand[k], matched_supply[matched_supply_idx]] / 0.4) + int(tmp_od[matched_demand[k]] / 0.4) + random.randint(0,10)  # 30초 400m
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
            sum_OD += total_OD
            sum_PD += total_PD
            num_matched += len(matched_demand)

        ##################################### matching 끝 #####################################

        reward = abs((total_OD - total_PD) / 0.132 * 100)     # 132m당 100원

        mean_revenue += total_OD / 0.132 * 100

        if len(self.supply)==0:
            return 0, np.array([])
        next_supply = np.delete(self.supply, matched_supply, axis=0)  # 매칭이 끝나고 남은 택시들 (바로 뒤 timestep에서 쓸거라 따로 뺴놓음)
        next_supply[:,0] = str(self.hour + 1)

        if len(matched_supply) == 0:
            return 0, next_supply

        if len(matched_supply) != 0 and len(newMatrix)>0:
            if len(next_supply)==0 :
                next_supply = np.array(newMatrix)
            else:
                next_supply = np.vstack((next_supply, np.array(newMatrix)))
        return reward, next_supply

    def sim(self, level, demand):  # numsection 만큼 supply 0.1 씩 해보고 top 3의 supply, reward 반환. platform class에서 호출
        global section_dict, numsection
        results = []
        if level != 0:
            self.approximate_demand(level)
        else :
            self.demand = demand
            self.set_demand()

        for i in range(numsection+1):   # i -> action
            r, s = self.matching(i, True)
            results.append([r, s, i])   # reward, next supply, action
        results.sort(reverse=True, key=lambda x: x[0])    # reward가 큰 순서대로
        return results[:3]


class Platform:  # 역할: OD별, PD별로 demand, supply 정리해서 gu에 넘겨주기, simulation, matching
    def __init__(self, case):

        self.numsection = 0
        self.second = 0
        self.case = case
        self.episode_time = 720  # 6시간

        if case == 1:
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

        for i in range(self.numsection):
            self.gu_list.append(Gu(i))

        global numsection, section_dict, second, reverse_section_dict
        numsection = self.numsection
        section_dict = self.section_dict
        for i in range(numsection):
            reverse_section_dict[section_dict[i]] = i
        second = self.second

        self.load_data()

        self.divide_district()

    def divide_district(self):
        tmp_len = 0
        for i in range(self.episode_time):  # 행 시간 열 지역에 따른 수요, 공급
            tmp_demand = self.demand[self.demand[:, 0] == str(i)]
            tmp_supply = self.supply[self.supply[:, 0] == str(i)]
            tmp = []
            for j in range(self.numsection):
                tmp.append(tmp_demand[tmp_demand[:, 2] == self.section_dict[j]])
            self.demand_hour.append(tmp)

            tmp = []
            for j in range(self.numsection):
                if i < 15:
                    tmp.append(tmp_supply[tmp_supply[:, 1] == self.section_dict[j]])
                    # print('supply {} time {}idx {}'.format(len(tmp_supply[tmp_supply[:, 1] == self.section_dict[j]]), i, j))
                    tmp_len += len(tmp_supply[tmp_supply[:, 1] == self.section_dict[j]])
                else:
                    tmp.append(np.reshape(np.array([]),(0,4)))
            self.supply_hour.append(tmp)
        print('total len {}'.format(tmp_len))


        self.demand_hour = np.array(self.demand_hour,dtype=object)
        self.supply_hour = np.array(self.supply_hour,dtype=object)

    def divide_supply(self, supply, sim=True):
        global section_dict, numsection

        if sim == False:
            how_long = 0
            for i in range(len(supply)):
                how_long += (int(supply[i,0]) - self.hour)
                if int(supply[i,0]) < self.episode_time:
                    self.supply_hour[int(supply[i,0]), find_section_number(supply[i,1])] = \
                        np.vstack((self.supply_hour[int(supply[i, 0]), find_section_number(supply[i, 1])], supply[i]))

        if sim == True:
            new_supply = copy.deepcopy(self.supply_hour)
            for i in range(len(supply)):
                if int(supply[i, 0]) < self.episode_time:
                    new_supply[int(supply[i, 0]), find_section_number(supply[i, 1])] = \
                        np.vstack((new_supply[int(supply[i, 0]), find_section_number(supply[i, 1])], supply[i]))
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

    def sim(self, next_gu_list, next_supply, level, hour):
        global section_dict, numsection, simulation_level, select_action

        if level > 0 and hour >= self.episode_time:
            return 0

        new_gu_list = next_gu_list[:]
        new_supply = copy.deepcopy(next_supply)
        for i in range(len(new_gu_list)):
            new_gu_list[i].hour = hour
        gap = []  # 각 구의 top1이랑 top2 차이

        reward_supply_action_list = []  # shape -> [numsection, 2]로 구성

        for i in range(self.numsection):
            new_gu_list[i].supply = new_supply[hour, i]  # 새로운 supply 넣어줌.
            reward_supply_action_list.append(
                new_gu_list[i].sim(level, self.demand_hour[hour][i]))  # reward, supply, action 반환
            gap.append(reward_supply_action_list[i][0][0] - reward_supply_action_list[i][1][0])  # reward 갭
            # 첫번쨰 idx : 구역, 두번째 idx : 몇번째로 높은지, 세번째 idx : 순서대로 reward,supply,action중 무엇일지

        reward_supply_action_list = np.array(reward_supply_action_list,dtype=object)
        actions = reward_supply_action_list[:,:,2]

        minidx = np.argmin(gap)
        gap[minidx] = sys.maxsize
        secidx = np.argmin(gap)

        if level == simulation_level-1:     # 마지막 시뮬레이션때 바로 최대 reward 반환
            return np.sum(reward_supply_action_list[:,0,0])

        top1_supply = None
        top2_supply = None
        top3_supply = None

        start_idx = 0   # shape 관계상
        for i in range(numsection):
            if len(reward_supply_action_list[i,0,1])>0:
                if i == minidx:
                    if len(reward_supply_action_list[i, 1, 1]) != 0:
                        top2_supply = reward_supply_action_list[i, 1, 1]
                    if len(reward_supply_action_list[i, 0, 1]) != 0:
                        top1_supply = reward_supply_action_list[i, 0, 1]
                        top3_supply = reward_supply_action_list[i, 0, 1]
                elif i == secidx:
                    if len(reward_supply_action_list[i, 1, 1]) != 0:
                        top3_supply = reward_supply_action_list[i, 1, 1]
                    if len(reward_supply_action_list[i, 0, 1]) != 0:
                        top1_supply = reward_supply_action_list[i, 0, 1]
                        top2_supply = reward_supply_action_list[i, 0, 1]
                else:
                    if len(reward_supply_action_list[i, 0, 1]) != 0:
                        top1_supply = reward_supply_action_list[i, 0, 1]
                        top2_supply = reward_supply_action_list[i, 0, 1]
                        top3_supply = reward_supply_action_list[i, 0, 1]
                start_idx = i+1
                break
        for i in range(start_idx,numsection):
            if i==minidx:
                if len(reward_supply_action_list[i,1,1])!=0:
                    top2_supply = np.vstack((top2_supply, reward_supply_action_list[i, 1, 1]))
                if len(reward_supply_action_list[i,0,1])!=0:
                    top1_supply = np.vstack((top1_supply, reward_supply_action_list[i, 0, 1]))
                    top3_supply = np.vstack((top3_supply, reward_supply_action_list[i, 0, 1]))
            elif i==secidx:
                if len(reward_supply_action_list[i, 1, 1]) != 0:
                    top3_supply = np.vstack((top3_supply, reward_supply_action_list[i, 1, 1]))
                if len(reward_supply_action_list[i, 0, 1]) != 0:
                    top1_supply = np.vstack((top1_supply, reward_supply_action_list[i, 0, 1]))
                    top2_supply = np.vstack((top2_supply, reward_supply_action_list[i, 0, 1]))
            else:
                if len(reward_supply_action_list[i, 0, 1]) != 0:
                    top1_supply = np.vstack((top1_supply, reward_supply_action_list[i, 0, 1]))
                    top2_supply = np.vstack((top2_supply, reward_supply_action_list[i, 0, 1]))
                    top3_supply = np.vstack((top3_supply, reward_supply_action_list[i, 0, 1]))


        top1_supply = self.divide_supply(top1_supply, sim=True)
        reward = []
        if level < simulation_level-1:      # simulation_level이 3이면 시뮬레이션 2번
            reward.append(self.sim(new_gu_list, top1_supply, level + 1, hour + 1))

        # top2
        top2_supply = self.divide_supply(top2_supply, sim=True)
        top2_actions = copy.deepcopy(actions[:,0])     #return 전용
        top2_actions[minidx] = actions[minidx, 1]
        if level < simulation_level-1:
            reward.append(self.sim(new_gu_list, top2_supply, level + 1, hour + 1))

        # top3
        top3_supply = self.divide_supply(top3_supply, sim=True)
        top3_actions = copy.deepcopy(actions[:,0])
        top3_actions[secidx] = actions[secidx, 1]
        if level < simulation_level-1:
            reward.append(self.sim(new_gu_list, top3_supply, level + 1, hour + 1))

        if level > 0:
            return max(reward)
        else:
            top2_actions[minidx] = actions[minidx, 1]
            # print('actions {} top2 {} top3 {} minidx{} secidx {}'.format(actions, top2_actions.shape, top3_actions, minidx, secidx))
            f = lambda i: reward[i]  # numpy말고 list 에서의 argmax
            best_action = max(range(len(reward)), key=f)
            if best_action == 0:
                select_action[0] += 1
                best_actions = actions[:,0]
                return best_actions
            elif best_action == 1:
                select_action[1] += 1
                return top2_actions
            else:
                select_action[2] += 1
                return top3_actions

    def step(self):  # simulation 돌리고 gu의 is_sim false로 하고 matching 시키는 함수
        global supply_minus_demand, mean_profit, mean_revenue, num_matched, sum_OD, sum_PD

        reward = 0
        for i in tqdm.tqdm(range(self.episode_time)):
            now_supply = 0
            for j in range(numsection):
                now_supply += len(self.supply_hour[self.hour, j])

            # print('platform total supply length {}'.format(total_supply))

            if self.case != 5:
                best_actions = self.sim(self.gu_list, self.supply_hour, 0, self.hour)
            else:
                best_actions = [0]

            tmplen = 0
            for j in range(numsection):
                tmplen += len(self.demand_hour[self.hour, j])
            print('platform supply - demand length {}'.format(now_supply-tmplen))
            supply_minus_demand.append(now_supply-tmplen)

            tmplen = 0
            for j in range(numsection):
                self.gu_list[j].supply = self.supply_hour[self.hour][j]
                self.gu_list[j].demand = self.demand_hour[self.hour][j]
                self.gu_list[j].set_demand()
                # print('REAL!!!!!!!!!!!!')
                r, s = self.gu_list[j].matching(best_actions[j], False)
                tmplen += len(s)

                reward += r
                self.divide_supply(s, sim=False)
            self.hour += 1

        print('mean profit {}'.format(reward))
        print('mean revenue {}'.format(sum_OD/ 0.132 * 100))
        print('sum OD {}'.format(sum_OD))
        print('sum PD {}'.format(sum_PD))

        print('result reward {}'.format(reward))
        print('mean ORR {}'.format(sum(ORR_list) / float(len(ORR_list))))
        print('mean OD {}'.format(sum_OD / float(num_matched)))

if __name__ == "__main__":

    plt.style.use('seaborn-whitegrid')

    case = int(input('case : '))
    # case = 1
    env = Platform(case)

    env.step()

    x = ['1', '2', '3']
    plt.figure(1)
    plt.bar(x, select_action)
    plt.title('Best action count in first simulation')
    plt.xlabel('action index')
    plt.ylabel('count')
    plt.savefig('./graph/real_new_graph/matching_greedy_' + str(numsection) + 'section_' + str(second) + 'second_action.png')

    print('supply minus demand : {}'.format(supply_minus_demand))
    x = np.arange(len(supply_minus_demand))
    plt.figure(2)
    plt.plot(x,supply_minus_demand)
    plt.title('Number of supply - Number of demand in every each time')
    plt.xlabel('time (1time = 30s)')
    plt.ylabel('count')
    plt.savefig('./graph/real_new_graph/matching_greedy_' + str(numsection) + 'section_' + str(second) + 'second_supply_minus_demand.png')