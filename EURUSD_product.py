import os
import configparser
import pickle

import psycopg2
import tensorflow as tf
import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

'''
ob_shape = 8
hold_week = 1
review_week = 4
data: Change_in_Asset_Mgr_Long_All Change_in_Asset_Mgr_Short_All
symbol: EURUSD
'''

config_reader = configparser.ConfigParser()
config_reader.read('config/db.ini', encoding='utf-8')
DATABASE = config_reader.get(section='postgres', option='database')
USER = config_reader.get(section='postgres', option='user')
PASSWORD = config_reader.get(section='postgres', option='password')
HOST = config_reader.get(section='postgres', option='host')
PORT = config_reader.get(section='postgres', option='port')


class TradingEnvProduct(py_environment.PyEnvironment):

    def __init__(self, symbol, ob_shape, review_week):
        super().__init__()
        self.symbol = symbol
        self.ob_shape = ob_shape
        self.review_week = review_week
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.ob_shape,), dtype=np.float64, name='observation'
        )
        self.train_data = self.gen_state_data()
        self._state = self.train_data[0:self.review_week].reshape((self.ob_shape,))
        self._episode_ended = False


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self.train_data[0:self.review_week].reshape((self.ob_shape,))
        self._state = np.array(self._state, dtype=np.float64)
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        '''
        返回 新的state 奖励 是否done
        '''
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        reward = 0
        new_state = self._state
        self._state_count += 1
        self._episode_ended = True

        if self._episode_ended:
            return ts.termination(new_state, reward=reward)
        else:
            return ts.transition(
                new_state, reward=reward, discount=0
            )

    def gen_state_data(self):
        ret = fetch_recent_cftc_data('financial.euro_fx', 4).reverse()
        train_data = pd.DataFrame(ret)
        train_data = train_data.reset_index(drop=True)

        # 数据标准化
        with open(f'CFTC_{self.symbol}_2week_mean.pickle', 'rb') as f:
            train_mean = pickle.load(f)
        with open(f'CFTC_{self.symbol}_2week_std.pickle', 'rb') as f:
            train_std = pickle.load(f)
        train_data = (train_data - train_mean) / train_std

        train_data_array = train_data.values
        return train_data_array


def fetch_recent_cftc_data(table_name, num_of_week):
    conn = psycopg2.connect(database=DATABASE, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT institutional_long_change,institutional_short_change FROM {table_name}
        ORDER BY report_date DESC
        LIMIT {num_of_week}
    """)
    ret = cur.fetchall()

    cur.close()
    conn.close()
    return ret


def product(symbol, policy_num, iter_num):
    policy_dir = os.path.join(os.getcwd(), 'product_cftc_data', str(policy_num), f'{iter_num}')
    saved_policy = tf.saved_model.load(policy_dir)

    product_env_py = TradingEnvProduct(symbol, ob_shape=8, review_week=4)
    product_env = tf_py_environment.TFPyEnvironment(product_env_py)
    time_step = product_env.reset()
    action_step = saved_policy.action(time_step)
    print(action_step.action.numpy())


if __name__ == '__main__':
    product('EURUSD', 17, 45000)