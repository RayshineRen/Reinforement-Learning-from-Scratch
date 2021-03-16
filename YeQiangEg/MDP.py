# 辅助函数
def str_key(*args):
    '''将参数用"_"连接起来作为字典的键，需注意参数本身可能会是tuple或者list型，
    比如类似((a,b,c),d)的形式。
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)

def set_dict(target_dict, value, *args):
    target_dict[str_key(*args)] = value

def set_prob(P, s, a, s1, p=1.0):  # 设置概率字典
    set_dict(P, p, s, a, s1)

def get_prob(P, s, a, s1):  # 获取概率值
    return P.get(str_key(s, a, s1), 0)

def set_reward(R, s, a, r):  # 设置奖励字典
    set_dict(R, r, s, a)

def get_reward(R, s, a):  # 获取奖励值
    return R.get(str_key(s, a), 0)

def display_dict(target_dict):  # 显示字典内容
    for key in target_dict.keys():
        print("{}:　{:.2f}".format(key, target_dict[key]))
    print("")

# 辅助方法
def set_value(V, s, v):  # 设置价值字典
    set_dict(V, v, s)

def get_value(V, s):  # 获取价值值
    return V.get(str_key(s), 0)

def set_pi(Pi, s, a, p=0.5):  # 设置策略字典
    set_dict(Pi, p, s, a)

def get_pi(Pi, s, a):  # 获取策略（概率）值
    return Pi.get(str_key(s, a), 0)

# 构建学生马尔科夫决策过程
S = ['浏览手机中','第一节课','第二节课','第三节课','休息中']
A = ['浏览手机','学习','离开浏览','泡吧','退出学习']
R = {} # 奖励Rsa
P = {} # 状态转移概率Pss'a
gamma = 0.5 # 衰减因子

set_prob(P, S[0], A[0], S[0]) # 浏览手机中 - 浏览手机 -> 浏览手机中
set_prob(P, S[0], A[2], S[1]) # 浏览手机中 - 离开浏览 -> 第一节课
set_prob(P, S[1], A[0], S[0]) # 第一节课 - 浏览手机 -> 浏览手机中
set_prob(P, S[1], A[1], S[2]) # 第一节课 - 学习 -> 第二节课
set_prob(P, S[2], A[1], S[3]) # 第二节课 - 学习 -> 第三节课
set_prob(P, S[2], A[4], S[4]) # 第二节课 - 退出学习 -> 退出休息
set_prob(P, S[3], A[1], S[4]) # 第三节课 - 学习 -> 退出休息
set_prob(P, S[3], A[3], S[1], p = 0.2) # 第三节课 - 泡吧 -> 第一节课
set_prob(P, S[3], A[3], S[2], p = 0.4) # 第三节课 - 泡吧 -> 第一节课
set_prob(P, S[3], A[3], S[3], p = 0.4) # 第三节课 - 泡吧 -> 第一节课

set_reward(R, S[0], A[0], -1) # 浏览手机中 - 浏览手机 -> -1
set_reward(R, S[0], A[2],  0) # 浏览手机中 - 离开浏览 -> 0
set_reward(R, S[1], A[0], -1) # 第一节课 - 浏览手机 -> -1
set_reward(R, S[1], A[1], -2) # 第一节课 - 学习 -> -2
set_reward(R, S[2], A[1], -2) # 第二节课 - 学习 -> -2
set_reward(R, S[2], A[4],  0) # 第二节课 - 退出学习 -> 0
set_reward(R, S[3], A[1], 10) # 第三节课 - 学习 -> 10
set_reward(R, S[3], A[3], +1) # 第三节课 - 泡吧 -> -1

MDP = (S, A, R, P, gamma)
# 设置行为策略：pi(a|.) = 0.5
Pi = {}
set_pi(Pi, S[0], A[0], 0.5) # 浏览手机中 - 浏览手机
set_pi(Pi, S[0], A[2], 0.5) # 浏览手机中 - 离开浏览
set_pi(Pi, S[1], A[0], 0.5) # 第一节课 - 浏览手机
set_pi(Pi, S[1], A[1], 0.5) # 第一节课 - 学习
set_pi(Pi, S[2], A[1], 0.5) # 第二节课 - 学习
set_pi(Pi, S[2], A[4], 0.5) # 第二节课 - 退出学习
set_pi(Pi, S[3], A[1], 0.5) # 第三节课 - 学习
set_pi(Pi, S[3], A[3], 0.5) # 第三节课 - 泡吧

V = {}

def compute_q(MDP, V, s, a):
    S, A, R, P, gamma = MDP
    q_sa = 0.0
    for s_prime in S:
        q_sa += get_prob(P, s, a, s_prime) * get_value(V, s_prime)
    q_sa = get_reward(R, s, a) + gamma * q_sa
    return q_sa

def compute_v(MDP, V, Pi, s):
    '''给定MDP下依据某一策略Pi和当前状态价值函数V计算某状态s的价值
    '''
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s, a) * compute_q(MDP, V, s, a)
    return v_s

def update_v(MDP, V, Pi):
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[str_key(s)] = compute_v(MDP, V_prime, Pi, s)
    return V_prime

def policy_evaluate(MDP, V, Pi, n):
    '''使用n次迭代计算来评估一个MDP在给定策略Pi下的状态价值，初始时价值为V
    '''
    for i in range(n):
        V = update_v(MDP, V, Pi)
        #display_dict(V)
    return V

V = policy_evaluate(MDP, V, Pi, 100)
display_dict(V)
# 验证状态在某策略下的价值
v = compute_v(MDP, V, Pi, "第三节课")
print("第三节课在当前策略下的价值为:{:.2f}".format(v))

# 价值迭代得到最优状态价值过程
def compute_v_from_max_q(MDP, V, s):
    '''根据一个状态的下所有可能的行为价值中最大一个来确定当前状态价值
    '''
    S, A, R, P, gamma = MDP
    v_s = -float('inf')
    for a in A:
        qsa = compute_q(MDP, V, s, a)
        if qsa >= v_s:
            v_s = qsa
    return v_s

def update_V_without_pi(MDP, V):
    '''在不依赖策略的情况下直接通过后续状态的价值来更新状态价值
    '''
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        set_value(V_prime, s, compute_v_from_max_q(MDP, V_prime, s))
        # V_prime[str_key(s)] = compute_v_from_max_q(MDP, V_prime, s)
    return V_prime

def value_iterate(MDP, V, n):
    '''价值迭代
    '''
    for i in range(n):
        V = update_V_without_pi(MDP, V)
        display_dict(V)
    return V

V = {}
# 通过价值迭代得到最优状态价值及
V_star = value_iterate(MDP, V, 4)
display_dict(V_star)

# 验证最优行为价值
s, a = "第三节课", "泡吧"
q = compute_q(MDP, V_star, "第三节课", "泡吧")
print("在状态{}选择行为{}的最优价值为:{:.2f}".format(s,a,q))

# display q_star
def display_q_star(MDP, V_star):
    S, A, _, _, _ = MDP
    for s in S:
        for a in A:
            print("q*({},{}):{}".format(s,a, compute_q(MDP, V_star, s, a)))

display_q_star(MDP, V_star)