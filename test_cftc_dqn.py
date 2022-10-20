import base64
import datetime
import configparser
import io
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import tempfile
import tensorflow as tf
import zipfile
import pandas as pd

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from cftc_env import TradingEnv, TradingEnvVal, TradingEnvTest
from utils.make_chart import write_img, draw_polyline

symbol_dict = {
    # 'EURUSD':957000,
    # 'GBPUSD':853000,
    # 'AUDUSD':1832000,
    # 'GOLD':1921000,
    # 'COCOA':364000,
}
config_reader = configparser.ConfigParser()
config_reader.read('config/config.ini', encoding='utf-8')
ob_shape = config_reader.getint(section='CFTC', option='ob_shape')
hold_week = config_reader.getint(section='CFTC', option='hold_week')
review_week = config_reader.getint(section='CFTC', option='review_week')

start_date_str = '2010-01-0500:00'
end_date_str = '2022-04-0500:00'
tick_spacing = 200

def compute_avg_return(environment, policy, policy_num, symbol, start_date):
    episode_return = 0.0
    time_step = environment.reset()
    episode_return_list = []
    episode_date_list = []
    time_bias = 7

    trade_date = pd.to_datetime(start_date, format='%Y-%m-%d')

    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        print(environment.pyenv.envs[0].entry_date, action_step.action.numpy(), time_step.reward.numpy())
        episode_return += time_step.reward
        # 计算盈利百分比 货币对除以100000 黄金除以150000
        if symbol in ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD']:
            episode_return_list.append(int(episode_return.numpy())/100000)
        elif symbol in ['GOLD']:
            episode_return_list.append(int(episode_return.numpy()) / 150000)
        elif symbol in ['COCOA']:
            episode_return_list.append(int(episode_return.numpy()) / 2000)
        else:
            episode_return_list.append(int(episode_return.numpy()) / 200000)
        trade_date = trade_date + datetime.timedelta(days=time_bias)
        episode_date_list.append(trade_date.strftime('%Y-%m-%d'))

    print(f'{symbol} 第{policy_num}迭代交易获利：{episode_return}')
    if episode_return > 30000:
        df = pd.DataFrame()
        df['date'] = episode_date_list
        df['profit_curve'] = episode_return_list
        draw_polyline(df, f'{policy_num}', 'profit')
        # write_img([episode_return_list], [episode_date_list], [symbol], F'CFTC_DQN_{policy_num}', tick_spacing=tick_spacing)
    avg_return = episode_return
    return avg_return.numpy()[0], episode_return_list, episode_date_list


# 范围测试
def range_test(symbol, policy_num):
    policy_base_dir = os.path.join(os.getcwd(), 'dqn_policy', str(policy_num))
    eval_interval = 500
    train_py_env = TradingEnv(symbol, ob_shape=ob_shape, hold_week=hold_week, review_week=review_week)
    train_start_date = train_py_env.train_date[0:1].values[0]
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_py_env = TradingEnvVal(symbol=symbol, mode='dev', ob_shape=ob_shape, hold_week=hold_week, review_week=review_week)
    eval_start_date = eval_py_env.train_date[0:1].values[0]
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    test_py_env = TradingEnvTest(symbol, mode='dev', ob_shape=ob_shape, hold_week=hold_week, review_week=review_week, start_time=None,
                             end_time=None)
    test_start_date = test_py_env.train_date[0:1].values[0]
    test_env = tf_py_environment.TFPyEnvironment(test_py_env)
    for i in range(15000,50000):
        if i % eval_interval == 0:
            policy_dir = os.path.join(policy_base_dir, f'{i}')
            saved_policy = tf.compat.v2.saved_model.load(policy_dir)
            avg_return_train = compute_avg_return(train_env, saved_policy, i, symbol, train_start_date)
            avg_return_eval = compute_avg_return(eval_env, saved_policy, i, symbol, eval_start_date)
            avg_return_test = compute_avg_return(test_env, saved_policy, i, symbol, test_start_date)
            # print('step = {0}: Erain Average Return = {1}'.format(1, avg_return_train))
            # print('step = {0}: Eval Average Return = {1}'.format(1, avg_return_eval))
            # print('step = {0}: Test Average Return = {1}'.format(1, avg_return_test))
            
# 独立测试
def single_test(number, symbol):
    policy_base_dir = os.path.join(os.getcwd(), 'dqn_policy')
    policy_dir = os.path.join(policy_base_dir, f'{number}')
    saved_policy = tf.compat.v2.saved_model.load(policy_dir)
    eval_py_env = TradingEnvTest(symbol=symbol, mode='product', ob_shape=36, hold_week=2, review_week=3, start_time=None, end_time=None)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    avg_return = compute_avg_return(eval_env, saved_policy, f'{number}', symbol)
    print('step = {0}: Average Return = {1}'.format(1, avg_return))


# 生产测试
def product_test(number):
    policy_base_dir = os.path.join(os.getcwd(), f'goal\\CFTC_{symbol}_2week')
    policy_dir = os.path.join(policy_base_dir, f'{number}')
    saved_policy = tf.compat.v2.saved_model.load(policy_dir)

    product_env_py = TradingEnvProductTwoWeek(f'{symbol}')
    product_env = tf_py_environment.TFPyEnvironment(product_env_py)
    time_step = product_env.reset()
    action_step = saved_policy.action(time_step)
    print(action_step.action.numpy())

# 生产回溯测试
def product_single_test():
    sum_return_list = []
    sum_date_list = []
    for symbol in symbol_dict.keys():
        eval_py_env = TradingEnvValTwoWeek(symbol=symbol, mode='product', version='2.0', start_time=pd.to_datetime(start_date_str, format='%Y-%m-%d%H:%M'), end_time=pd.to_datetime(end_date_str, format='%Y-%m-%d%H:%M'))
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        number = symbol_dict[symbol]
        policy_base_dir = os.path.join(os.getcwd(), f'goal\\CFTC_{symbol}_2week')
        policy_dir = os.path.join(policy_base_dir, f'{number}')
        saved_policy = tf.compat.v2.saved_model.load(policy_dir)
        avg_return, return_list, date_list = compute_avg_return(eval_env, saved_policy, f'{number}', symbol)
        sum_return_list.append(return_list)
        sum_date_list.append(date_list)
        print('step = {0}: Average Return = {1}'.format(1, avg_return))
    # all_symbol_sum = []
    # for i in range(len(sum_return_list[0])):
    #     all_symbol_profit = 0
    #     for x in range(len(sum_return_list)):
    #         all_symbol_profit += sum_return_list[x][i]
    #     all_symbol_sum.append([all_symbol_profit])
    # sum_return_list.append(all_symbol_sum)
    # sum_date_list.append(sum_date_list[0])
    # symbol_dict['SUM'] = 0
    write_img(sum_return_list, sum_date_list, list(symbol_dict), F'CFTC_DQN模型综合表现',tick_spacing=tick_spacing)

range_test('EURUSD', 5)

# collection_list = [
#     150000,
#     # 163000,
#     # 166000,
#     # 167000,
#     # 168000,
#     # 173000,
# ]
# for i in collection_list:
#     single_test(i, 'USDCAD')

# product_test(1832000)
# product_single_test()

