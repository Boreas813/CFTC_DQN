# CFTC_DQN
基于CFTC报告的增强学习交易模型

## 训练你自己的模型

什么是增强学习？ https://en.wikipedia.org/wiki/Reinforcement_learning

什么是DQN？ https://openai.com/blog/openai-baselines-dqn/

什么是CFTC报告？ https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm

如何使用tensorflow训练DQN？ https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

## 数据源

CFTC报告和周线历史数据放在history_data文件夹

CFTC历史数据来源:
https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm

价格历史数据源自我的交易商：
https://www.avatrade.com/


## 使用警告

1. 增强学习模型是黑盒科技，我们无法知道市场环境发生巨大变化后它是否还正常工作。
2. 市场一直在变化，当欧洲央行决定将利率降至负值后，流动性变高波动性变低，收益也随之下降。当08年金融危机发生后，全球央行降低利率迅速向市场注入流动性，08年之前的货币市场波动水平和08年之后变化巨大，市场发生巨变后，模型必须被重新训练。
3. 确保模型可以撑过各种危机，例如2008.7、2020.3、2021.12、2022.2。
4. 反复确认输入数据是否正确，确保每个训练参数转成tensor后不出现错位等bug，垃圾进垃圾出。
