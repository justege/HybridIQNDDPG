### HybridIQNDDPG
This is a PyTorch implementation of the Hybrid IQN-DDPG algorithm

## Preprocessing
- CSVs: This folder contains the raw data for the stocks. The data is from Yahoo Finance.
- FirstProcessing.py: This file is used to preprocess the raw data and generate the training data.
- SecondProcessing.py: This file is used to preprocess the training data and generate the training data for the DDPG model.
- DataCheckForErrors.py: This file is used to check the data for errors and remove them.
- Correlations.py: This file is used to calculate the correlation for Volume and Price data generated from first and secondprocessing 
- Plotting_all_together.py: This file is used to plot the data generated from first and secondprocessing, including technical indicators, prices and volumes.
- Plotting_each_TA.py. Use this file if you would like to plot each Technical Indicator separately.

Stock Trading Environment (Version 3)
This folder contains the StockEnvTrainVersion3 class, which is a custom OpenAI gym environment for stock trading based on historical market data. This environment allows you to train reinforcement learning agents to optimize trading strategies using technical indicators.

#Features
- Supports training with multiple stocks in the portfolio
- Handles transaction fees
- Utilizes common technical indicators like MACD, RSI, CCI, and ADX
- Calculates daily portfolio returns and Sharpe ratio

To use this environment, you will need the following Python libraries:

- numpy
- pandas
- gym
- matplotlib

#Usage
1. Import the StockEnvTrainVersion3 class from the provided Python file.
2. Initialize the environment with a pandas dataframe containing the historical market data and the starting day of the simulation.
3. Use the environment's step and reset methods in your reinforcement learning training loop.

```
import pandas as pd
from stock_env_v3 import StockEnvTrainVersion3

# Load historical market data into a pandas dataframe
data = pd.read_csv('your_market_data.csv')

# Initialize the environment
env = StockEnvTrainVersion3(df=data, day=0)

# Use the environment in a reinforcement learning training loop
state = env.reset()
done = False

while not done:
    action = agent.choose_action(state)  # Replace with your agent's action selection method
    next_state, reward, done, info = env.step(action)
    state = next_state
```

#Observation Space, Reward function, and action space
The observation space is a vector of 22 elements, including:

- Current account balance
- Current stock prices for each stock
- Owned shares for each stock
- MACD, RSI, CCI, and ADX indicators for each stock

!!! You can change this observation space, action space and the reward function as you wish. Currently The action space is a continuous vector of 4 elements, with values ranging between 0 and 1. 

# Test and Validation. 
- TestEnvironment.py: test the RL agent with this environment 
- ValidationEnvironment.py: validate the agent according to the Sharpe ratio and the portfolio value.

additional feature: It tracks and visualizes agent's performance, including account value, daily returns, and Sharpe ratio

# Training
- run.py: run the code with the following command: python run.py, specify informations about training the agent by running: python run.py --help
after running the code, you will get the following results:
- The agent's model saved, the performance, including account value, daily returns, and Sharpe ratio as csv and png.
- Change the Tau value in the run.py and networks.py file to get the results for different Tau values.

#Testing
- run test_run.py with the same command: python test_run.py without. Specify the model path, Tau value you are interested of and the test timeframe in test_py.py


#Analysis
Analyze the results with 
- AnalyzeAverageWeights.py: This file is used to analyze the average weights of the agent's model.
- AnalyzePortfolioValueInOnePlot.py : This file is used to analyze the portfolio value of different DistRL agents models.
- AnalyzeReturnDistribution.py: This file is used to analyze the return distribution of different DistRL agents models.
- AnalyzeRechnicalIndicatorVsBuyActivity.py: This file is used to analyze the technical indicator vs buy activity of the trained DistRL agents models.

