import datetime
import pickle
import copy

import numpy as np
import pandas as pd
import talib

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

TRAIN_CUT_PERCENT = 0.7
VAL_CUT_PERCENT = 0.15
TEST_CUT_PERCENT = 0.15

REVIEW_LEN = 3

class TradingEnv(py_environment.PyEnvironment):
    '''
    交易规则：
        1.输入六十天开高低收价，Mom8，Stoch833，RSI3/13
        2.持仓七天后平仓
    '''

    def __init__(self, symbol, ob_shape):
        super().__init__()
        self.symbol = symbol
        self.ob_shape = ob_shape

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.ob_shape,), dtype=np.float64, name='observation'
        )
        self.gen_state_data()
        self._state = self.train_data[0:REVIEW_LEN].reshape((self.ob_shape,))
        self._state_count = 0
        self._episode_ended = False
        self.mul, self.single_prpfit, self.spread = self.get_base_point(symbol)

    @staticmethod
    def get_base_point(symbol):
        if 'USD' in symbol:
            if 'JPY' in symbol:
                multiplier = 1000
                USDprofit = 0.0102
            else:
                multiplier = 100000
                USDprofit = 0.0103
        elif symbol in ['CADJPY']:
            multiplier = 1000
            USDprofit = 0.0102
        elif symbol in ['EURNZD', 'CADCHF', 'AUDNZD', 'AUDCAD', 'EURGBP', 'GBPCHF']:
            multiplier = 100000
            USDprofit = 0.0103
        elif symbol in ['BRENT_OIL', 'Brent']:
            multiplier = 100
            USDprofit = 0.1
        elif symbol in ['GOLD', 'USD_500', 'WHEAT']:
            multiplier = 100
            USDprofit = 0.01
        elif symbol in ['SUGAR#11']:
            multiplier = 100
            USDprofit = 1
        elif symbol in ['COCOA']:
            multiplier = 1
            USDprofit = 1
        elif symbol == 'SP500m':
            multiplier = 10
            USDprofit = 0.1
        elif symbol in ['SILVER']:
            multiplier = 1000
            USDprofit = 0.1
        elif symbol == 'COPPER':
            multiplier = 10000
            USDprofit = 0.01
        elif symbol == 'NASDAQ100':
            multiplier = 100
            USDprofit = 0.01
        elif symbol == 'SH':
            multiplier = 1000
            USDprofit = 1
        elif symbol == 'CHINA_A50':
            multiplier = 10
            USDprofit = 0.01
        else:
            multiplier = 1
            USDprofit = 1

        if symbol in ['EURUSD']:
            trade_point = 13
        elif symbol == 'USDJPY':
            trade_point = 15
        elif symbol == 'AUDUSD':
            trade_point = 18
        elif symbol == 'GBPUSD':
            trade_point = 20
        elif symbol == 'USDCAD':
            trade_point = 25
        elif symbol == 'USDCHF':
            trade_point = 20
        elif symbol == 'GOLD':
            trade_point = 35
        elif symbol == 'SILVER':
            trade_point = 35
        elif symbol == 'BRENT_OIL':
            trade_point = 6
        elif symbol == 'S&P500':
            trade_point = 50
        elif symbol == 'SP500m':
            trade_point = 5
        elif symbol == 'COPPER':
            trade_point = 60
        elif symbol == 'WHEAT':
            trade_point = 50
        elif symbol == 'CHINA_A50':
            trade_point = 125
        else:
            trade_point = 30
        return multiplier, USDprofit, trade_point

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state_count = 0
        self._state = self.train_data[0:REVIEW_LEN].reshape((self.ob_shape,))
        self._state = np.array(self._state, dtype=np.float64)
        self._episode_ended = False
        return ts.restart(self._state)

    def get_original_data(self, value, column_name):
        return value * self.train_std[column_name] + self.train_mean[column_name]


    def get_reward(self, action):
        entry_price = self.train_data_backup[self._state_count+REVIEW_LEN:self._state_count+REVIEW_LEN+1]['open'].values[0]
        leave_price = self.train_data_backup[self._state_count+REVIEW_LEN+3:self._state_count+REVIEW_LEN+4]['close'].values[0]

        # 进空头
        if action == 0:
            reward = (entry_price - (leave_price))*self.mul
        # 进多头
        elif action == 1:
            reward = ((leave_price) - entry_price) * self.mul
        # 不交易
        else:
            reward = 0
        return reward

    def _step(self, action):
        '''
        返回 新的state 奖励 是否done
        '''
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        reward = self.get_reward(action)

        self._state_count += 1
        new_state = np.array(
            self.train_data[self._state_count:self._state_count+REVIEW_LEN].reshape((self.ob_shape,)),
            dtype=np.float64)

        if self._state_count >= len(self.train_data)-(REVIEW_LEN+5):
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(new_state, reward=reward)
        else:
            return ts.transition(
                new_state, reward=reward, discount=0
            )

    def gen_state_data(self):
        train_data = pd.read_csv(f'history_data/{self.symbol}1440.csv')
        data_len = len(train_data)
        train_end = int(data_len * TRAIN_CUT_PERCENT)
        train_data = train_data[0:train_end].reset_index(drop=True)
        print(f'训练集数据序列：0:{train_end}')

        # 插入指标
        train_data['ATR6'] = talib.ATR(train_data['high'], train_data['low'], train_data['close'], timeperiod=6)
        train_data['MOM8'] = talib.MOM(train_data['close'], timeperiod=8)
        train_data['K8'], train_data['D8'] = talib.STOCH(
            train_data['high'], train_data['low'], train_data['close'],
            fastk_period=8, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
        )
        train_data['RSI3'] = talib.RSI(train_data['close'], timeperiod=3)
        train_data['RSI8'] = talib.RSI(train_data['close'], timeperiod=8)
        train_data = train_data[11:].reset_index(drop=True)
        self.train_data_backup = copy.deepcopy(train_data)

        #删除不要的列
        del (train_data['volume'])
        del (train_data['date_time'])
        del (train_data['minute'])
        del (train_data['ATR6'])
        del (train_data['open'])
        del (train_data['high'])
        del (train_data['low'])


        # 数据标准化
        train_data.fillna(0, inplace=True)
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()

        # with open(f'CFTC_{self.symbol}_2week_mean.pickle', 'rb') as f:
        #     train_mean = pickle.load(f)
        # with open(f'CFTC_{self.symbol}_2week_std.pickle', 'rb') as f:
        #     train_std = pickle.load(f)

        with open(f'CFTC_{self.symbol}_tech_mean.pickle', 'wb') as f:
            pickle.dump(self.train_mean, f, True)
        with open(f'CFTC_{self.symbol}_tech_std.pickle', 'wb') as f:
            pickle.dump(self.train_std, f, True)

        self.train_data = (train_data - self.train_mean) / self.train_std
        self.train_data = train_data.values

class TradingEnvValAndTest(py_environment.PyEnvironment):
    '''
    交易规则：
        1.输入六十天开高低收价，Mom8，Stoch833，RSI3/13
        2.持仓七天后平仓
    '''

    def __init__(self, symbol, ob_shape, mode='dev', val_or_test='val'):
        super().__init__()
        self.symbol = symbol
        self.ob_shape = ob_shape
        self.mode = mode
        self.val_or_test = val_or_test

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.ob_shape,), dtype=np.float64, name='observation'
        )
        self.gen_state_data()
        self._state = self.train_data[0:REVIEW_LEN].reshape((self.ob_shape,))
        self._state_count = 0
        self._episode_ended = False
        self.mul, self.single_prpfit, self.spread = self.get_base_point(symbol)

    @staticmethod
    def get_base_point(symbol):
        if 'USD' in symbol:
            if 'JPY' in symbol:
                multiplier = 1000
                USDprofit = 0.0102
            else:
                multiplier = 100000
                USDprofit = 0.0103
        elif symbol in ['CADJPY']:
            multiplier = 1000
            USDprofit = 0.0102
        elif symbol in ['EURNZD', 'CADCHF', 'AUDNZD', 'AUDCAD', 'EURGBP', 'GBPCHF']:
            multiplier = 100000
            USDprofit = 0.0103
        elif symbol in ['BRENT_OIL', 'Brent']:
            multiplier = 100
            USDprofit = 0.1
        elif symbol in ['GOLD', 'USD_500', 'WHEAT']:
            multiplier = 100
            USDprofit = 0.01
        elif symbol in ['SUGAR#11']:
            multiplier = 100
            USDprofit = 1
        elif symbol in ['COCOA']:
            multiplier = 1
            USDprofit = 1
        elif symbol == 'SP500m':
            multiplier = 10
            USDprofit = 0.1
        elif symbol in ['SILVER']:
            multiplier = 1000
            USDprofit = 0.1
        elif symbol == 'COPPER':
            multiplier = 10000
            USDprofit = 0.01
        elif symbol == 'NASDAQ100':
            multiplier = 100
            USDprofit = 0.01
        elif symbol == 'SH':
            multiplier = 1000
            USDprofit = 1
        elif symbol == 'CHINA_A50':
            multiplier = 10
            USDprofit = 0.01
        else:
            multiplier = 1
            USDprofit = 1

        if symbol in ['EURUSD']:
            trade_point = 13
        elif symbol == 'USDJPY':
            trade_point = 15
        elif symbol == 'AUDUSD':
            trade_point = 18
        elif symbol == 'GBPUSD':
            trade_point = 20
        elif symbol == 'USDCAD':
            trade_point = 25
        elif symbol == 'USDCHF':
            trade_point = 20
        elif symbol == 'GOLD':
            trade_point = 35
        elif symbol == 'SILVER':
            trade_point = 35
        elif symbol == 'BRENT_OIL':
            trade_point = 6
        elif symbol == 'S&P500':
            trade_point = 50
        elif symbol == 'SP500m':
            trade_point = 5
        elif symbol == 'COPPER':
            trade_point = 60
        elif symbol == 'WHEAT':
            trade_point = 50
        elif symbol == 'CHINA_A50':
            trade_point = 125
        else:
            trade_point = 30
        return multiplier, USDprofit, trade_point

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state_count = 0
        self._state = self.train_data[0:REVIEW_LEN].reshape((self.ob_shape,))
        self._state = np.array(self._state, dtype=np.float64)
        self._episode_ended = False
        return ts.restart(self._state)

    def get_original_data(self, value, column_name):
        return value * self.train_std[column_name] + self.train_mean[column_name]

    def get_reward(self, action):
        entry_price = self.train_data_backup[self._state_count+REVIEW_LEN:self._state_count+REVIEW_LEN+1]['open'].values[0]
        leave_price = self.train_data_backup[self._state_count+REVIEW_LEN+3:self._state_count+REVIEW_LEN+4]['close'].values[0]

        # 进空头
        if action == 0:
            reward = (entry_price - (leave_price))*self.mul
        # 进多头
        elif action == 1:
            reward = ((leave_price) - entry_price) * self.mul
        # 不交易
        else:
            reward = 0
        return reward

    def _step(self, action):
        '''
        返回 新的state 奖励 是否done
        '''
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        reward = self.get_reward(action)

        self._state_count += 1
        new_state = np.array(
            self.train_data[self._state_count:self._state_count+REVIEW_LEN].reshape((self.ob_shape,)),
            dtype=np.float64)

        if self._state_count >= len(self.train_data)-(REVIEW_LEN+5):
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(new_state, reward=reward)
        else:
            return ts.transition(
                new_state, reward=reward, discount=0
            )

    def gen_state_data(self):
        train_data = pd.read_csv(f'history_data/{self.symbol}1440.csv')
        # 开发回测阶段
        if self.mode == 'dev':
            if self.val_or_test == 'val':
                train_len = len(train_data)
                val_cut_start = int(train_len * TRAIN_CUT_PERCENT)
                val_cut_end = int(train_len * (TRAIN_CUT_PERCENT+VAL_CUT_PERCENT))
                train_data = train_data[val_cut_start:val_cut_end].reset_index(drop=True)
                print(f'验证集数据序列：{val_cut_start}:{val_cut_end}')
            else:
                train_len = len(train_data)
                test_cut_start = int(train_len * (TRAIN_CUT_PERCENT+VAL_CUT_PERCENT))
                train_data = train_data[test_cut_start:].reset_index(drop=True)
                print(f'验证集数据序列：{test_cut_start}:')
        else:
            pass

        # 插入指标
        train_data['ATR6'] = talib.ATR(train_data['high'], train_data['low'], train_data['close'], timeperiod=6)
        train_data['MOM8'] = talib.MOM(train_data['close'], timeperiod=8)
        train_data['K8'], train_data['D8'] = talib.STOCH(
            train_data['high'], train_data['low'], train_data['close'],
            fastk_period=8, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
        )
        train_data['RSI3'] = talib.RSI(train_data['close'], timeperiod=3)
        train_data['RSI8'] = talib.RSI(train_data['close'], timeperiod=8)
        train_data = train_data[11:].reset_index(drop=True)
        self.train_data_backup = copy.deepcopy(train_data)

        #删除不要的列
        del (train_data['volume'])
        del (train_data['date_time'])
        del (train_data['minute'])
        del (train_data['ATR6'])
        del (train_data['open'])
        del (train_data['high'])
        del (train_data['low'])

        # 数据标准化
        train_data.fillna(0, inplace=True)

        with open(f'CFTC_{self.symbol}_tech_mean.pickle', 'rb') as f:
            self.train_mean = pickle.load(f)
        with open(f'CFTC_{self.symbol}_tech_std.pickle', 'rb') as f:
            self.train_std = pickle.load(f)

        self.train_data = (train_data - self.train_mean) / self.train_std
        self.train_data = train_data.values



# env = TradingEnv(symbol='EURUSD', ob_shape=18)
# tf_env = tf_py_environment.TFPyEnvironment(env)
# print("TimeStep Specs:", tf_env.time_step_spec())
# print("Action Specs:", tf_env.action_spec())
# tf_env.reset()
# for i in range(10):
#     tf_env.step(0)
#     tf_env.step(1)
# pass