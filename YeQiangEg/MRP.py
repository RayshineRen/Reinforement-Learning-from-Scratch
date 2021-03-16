import numpy as np

# num_states = 7
# {"0": "C1", "1":"C2", "2":"C3", "3":"Pass", "4":"Pub", "5":"FB", "6":"Sleep"}
i_to_n = {}
i_to_n["0"] = "C1"
i_to_n["1"] = "C2"
i_to_n["2"] = "C3"
i_to_n["3"] = "Pass"
i_to_n["4"] = "Pub"
i_to_n["5"] = "FB"
i_to_n["6"] = "Sleep"

n_to_i = {}
for i, name in zip(i_to_n.keys(), i_to_n.values()):
    n_to_i[name] = int(i)

#   C1   C2   C3  Pass Pub   FB  Sleep
Pss = [
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
    [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]
Pss = np.array(Pss)
rewards = [-2, -2, -2, 10, 1, -1, 0]
gamma = 0.5

chains =[
    ["C1", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "Sleep"],
    ["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB",\
     "FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

def compute_return(start_index=0,
                   chain=None,
                   gamma=0.5)->float:
    '''计算一个马尔科夫奖励过程中某状态的收获值
    Args:
        start_index 要计算的状态在链中的位置
        chain 要计算的马尔科夫过程
        gamma 衰减系数
    Returns：
        Gt 收获值
    '''
    Gt, power, gamma = 0.0, 0, gamma
    for i in range(start_index, len(chain)):
        Gt += np.power(gamma, power) * rewards[n_to_i[chain[i]]]
        power += 1
    return Gt

def compute_value(Pss, rewards, gamma=0.5):
    '''通过求解矩阵方程的形式直接计算状态的价值
    Args：
        P 状态转移概率矩阵 shape(7, 7)
        rewards 即时奖励 list
        gamma 衰减系数
    Return
        values 各状态的价值
    '''
    rewards = np.array(rewards).reshape((-1, 1))
    values = np.dot(np.linalg.inv(np.eye(7,7)-gamma*Pss), rewards)
    return values

values = compute_value(Pss, rewards, gamma = 0.5)
print(values)



