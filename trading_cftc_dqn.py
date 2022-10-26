import os
import configparser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from cftc_env import TradingEnv, TradingEnvVal, TradingEnvTest

print(tf.version.VERSION)

symbol = 'EURUSD'

config_reader = configparser.ConfigParser()
config_reader.read('config/config.ini', encoding='utf-8')


num_iterations = config_reader.getint(section='CFTC', option='num_iterations')
initial_collect_steps = config_reader.getint(section='CFTC', option='initial_collect_steps')
collect_steps_per_iteration = config_reader.getint(section='CFTC', option='collect_steps_per_iteration')
replay_buffer_max_length = config_reader.getint(section='CFTC', option='replay_buffer_max_length')
batch_size = config_reader.getint(section='CFTC', option='batch_size')
learning_rate = config_reader.getfloat(section='CFTC', option='learning_rate')
log_interval = config_reader.getint(section='CFTC', option='log_interval')
num_eval_episodes = config_reader.getint(section='CFTC', option='num_eval_episodes')
eval_interval = config_reader.getint(section='CFTC', option='eval_interval')

ob_shape = config_reader.getint(section='CFTC', option='ob_shape')
hold_week = config_reader.getint(section='CFTC', option='hold_week')
review_week = config_reader.getint(section='CFTC', option='review_week')

# for test
env = TradingEnv(symbol=symbol, ob_shape=ob_shape, hold_week=hold_week, review_week=review_week)
print('Observation Spec:')
print(env.time_step_spec().observation)
print('Reward Spec:')
print(env.time_step_spec().reward)
print('Action Spec:')
print(env.action_spec())
time_step = env.reset()
print('Time step:')
print(time_step)
action = np.array(1, dtype=np.int32)
next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)
#
train_py_env = TradingEnv(symbol, ob_shape=ob_shape, hold_week=hold_week, review_week=review_week)
eval_py_env = TradingEnvVal(symbol, ob_shape=ob_shape, hold_week=hold_week, review_week=review_week, start_time=None, end_time=None)
test_py_env = TradingEnvTest(symbol, ob_shape=ob_shape, hold_week=hold_week, review_week=review_week, start_time=None, end_time=None)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
test_env = tf_py_environment.TFPyEnvironment(test_py_env)

fc_layer_params = (108, 14)
action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
	return tf.keras.layers.Dense(
		num_units,
		activation=tf.keras.activations.relu,
		kernel_initializer=tf.keras.initializers.VarianceScaling(
			scale=2.0, mode='fan_in', distribution='truncated_normal'))


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
	num_actions,
	activation=None,
	kernel_initializer=tf.keras.initializers.RandomUniform(
		minval=-0.03, maxval=0.03),
	bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
	train_env.time_step_spec(),
	train_env.action_spec(),
	q_network=q_net,
	optimizer=optimizer,
	td_errors_loss_fn=common.element_wise_squared_loss,
	train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


def compute_avg_return(environment, policy, num_episodes=10):
	total_return = 0.0
	for _ in range(num_episodes):

		time_step = environment.reset()
		episode_return = 0.0

		while not time_step.is_last():
			action_step = policy.action(time_step)
			time_step = environment.step(action_step.action)
			episode_return += time_step.reward
		total_return += episode_return

	avg_return = total_return / num_episodes
	return avg_return.numpy()[0]


random_result = compute_avg_return(eval_env, random_policy, num_eval_episodes)
print('random result:', random_result)

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
	agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
	replay_buffer_signature)

table = reverb.Table(
	table_name,
	max_size=replay_buffer_max_length,
	sampler=reverb.selectors.Uniform(),
	remover=reverb.selectors.Fifo(),
	rate_limiter=reverb.rate_limiters.MinSize(1),
	signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
	agent.collect_data_spec,
	table_name=table_name,
	sequence_length=2,
	local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
	replay_buffer.py_client,
	table_name,
	sequence_length=2)

print(agent.collect_data_spec)
print(agent.collect_data_spec._fields)

py_driver.PyDriver(
	env,
	py_tf_eager_policy.PyTFEagerPolicy(
		random_policy, use_tf_function=True),
	[rb_observer],
	max_steps=initial_collect_steps).run(train_py_env.reset())

dataset = replay_buffer.as_dataset(
	num_parallel_calls=3,
	sample_batch_size=batch_size,
	num_steps=2).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
eval_avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
eval_returns = [eval_avg_return]

test_avg_return = compute_avg_return(test_env, agent.policy, num_eval_episodes)
test_returns = [test_avg_return]

# Reset the environment.
time_step = train_py_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
	env,
	py_tf_eager_policy.PyTFEagerPolicy(
		agent.collect_policy, use_tf_function=True),
	[rb_observer],
	max_steps=collect_steps_per_iteration)

base_policy_dir = 'dqn_policy'
for i in range(1,100000):
	full_policy_dir = os.path.join(base_policy_dir, str(i))
	if not os.path.exists(full_policy_dir):
		os.makedirs(full_policy_dir)
		break
	else:
		continue
else:
	raise ValueError('too many train result')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

for _ in range(num_iterations):
	# Collect a few steps and save to the replay buffer.
	time_step, _ = collect_driver.run(time_step)

	# Sample a batch of data from the buffer and update the agent's network.
	experience, unused_info = next(iterator)
	train_loss = agent.train(experience).loss

	step = agent.train_step_counter.numpy()

	if step % log_interval == 0:
		print('step = {0}: loss = {1}'.format(step, train_loss))

	if step % eval_interval == 0:
		avg_return_eval = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
		print('step = {0}: Eval Average Return = {1}'.format(step, avg_return_eval))
		eval_returns.append(avg_return_eval)
		avg_return_test = compute_avg_return(test_env, agent.policy, num_eval_episodes)
		print('step = {0}: Test Average Return = {1}'.format(step, avg_return_test))
		tf_policy_saver.save(os.path.join(full_policy_dir, str(step)))
		test_returns.append(avg_return_test)

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, eval_returns, color='red')
plt.plot(iterations, test_returns, color='#054E9F')
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=55000)
plt.savefig(os.path.join(full_policy_dir, 'result.jpg'))
# plt.show()
