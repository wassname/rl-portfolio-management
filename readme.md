Attempting to replicate "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" by [Jiang et. al. 2017](https://arxiv.org/abs/1706.10059) [1].

tl;dr I managed to get 8% growth on training data, but it disapeared on test data. However, RL papers can be very difficult to replicate due to bugs, framework differences, and hyperparameter sensistivity.

# About

This paper trains an agent to choose a good portfolio of cryptocurrencies. It's reported that it can give 4-fold returns in 50 days and the paper seems to do all the right things so I wanted to see if I could acheive the same results.

This repo includes an environment for portfolio management (with unit tests). Hopefully others will find this usefull as I am not aware of any other implementations (as of 2017-07-17).

Author: wassname

License: AGPLv3

[[1](https://arxiv.org/abs/1706.10059)] Jiang, Zhengyao, Dixing Xu, and Jinjun Liang. "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem." *arXiv preprint arXiv:1706.10059* (2017).


# Results

I have managed to overfit to the training data with no trading costs but it could not generalise to the test data. So far there have been poor results. I have not yet tried hyperparameter optimisation so it could be that parameter tweaking will allow the model to fit, or I may have subtle bugs.

- VPG model,
  - training: 190% portfolio growth in 50 days
  - testing: 100% portfolio growth in 50 days

![](https://raw.githubusercontent.com/wassname/rl-portfolio-management/8c74f136765f621eb45d484553b9f778e9243a84/docs/tensorforce-VPG-test.png)

This test period is directly after the training period and it looks like the usefullness of the models learned knowledge may decay as it moves away from its training interval.

# Installing

- `git clone https://github.com/wassname/rl-portfolio-management.git`
- `cd rl-portfolio-management`
- `pip install -r requirements/requirements.txt`
- `jupyter-notebook`
    - Then open tensorflow-VPG.ipynb in jupyter
    - Or try an alternative agent  with tensorforce-PPO.ipynb and train


# Using the environment

There are three environments defined here to use them:
```py
import gym
import rl_portfolio_management.environments  # this registers them

env = gym.envs.spec('CryptoPortfolioEIIE-v0').make()
print("CryptoPortfolioEIIE has an state shape suitable for an EIIE model (see https://arxiv.org/abs/1706.10059)")
print("shape =", env.reset().shape)
# shape = (5, 50, 3)

env = gym.envs.spec('CryptoPortfolioMLP-v0').make()
print("CryptoPortfolioMLP has an state shape for a dense model")
print("shape =", env.reset().shape)
# shape = (750,)

env = gym.envs.spec('CryptoPortfolioAtari-v0').make()
print("CryptoPortfolioAtari has an state shape for models that expect an image, such as ones tuned on Atari games")
print("shape =", env.reset().shape)
# shape = (50, 50, 3)
```

Or define your own:
```py
import rl_portfolio_management.environments import PortfolioEnv
df_train = pd.read_hdf('./data/poloniex_30m.hf', key='train')
env = PortfolioEnv(
  df=df_train,
  steps=256,
  scale=True,
  augment=0.00,
  trading_cost=0.0025,
  time_cost=0.00,
  window_length=50,
  output_mode='mlp'
)
```

Let try it with a random agent and plot the results:


```py
for _ in tqdm(range(50)):

    # get random weights and normalize
    action = env.action_space.sample()
    action /= action.sum()

    state, reward, done, info = env.step(action)
    if done:
        break

env.render('notebook', True)
```

Unsuprisingly, the random agent doesn't perform well in portfolio management.

![](docs/img/price_performance.png)
![](docs/img/weights.png)


# Tests

We have partial test coverage of the environment, just run:

- `python -m pytest`


# Files

- enviroments/portfolio.py - contains an openai environment for porfolio trading
- tensorforce-VPG.ipynb - notebook to try a policy gradient agent
- tensorforce-PPO - notebook to try a PPO agent
- data/poloniex_30m.hdf - hdf file with cryptocurrency 30 minutes prices

# Differences in implementation


The main differences from Jiang et. al. 2017 are:

- The first step in a deep learning project should be to make sure the model can overfit, this provides a sanity check. So I am first trying to acheive good results with no trading costs.
- I have not used portfolio vector memory. For ease of implementation I made the information available by replacing the oldest timestep. Your model can slice it, or a Dense or CNN models can just be given the information.
- Instead of DPG ([deterministic policy gradient](http://jmlr.org/proceedings/papers/v32/silver14.pdf)) I tried and DDPG ([deep deterministic policy gradient]( http://arxiv.org/pdf/1509.02971v2.pdf)) and VPG (vanilla policy gradient) with generalized advantage estimation and PPO.
- I tried to replicate the best performing CNN model from the paper and haven't attempted the LSTM or RNN models.
- instead of selecting 12 assets for each window I chose 5 assets that have existed for the longest time
- My topology had an extra layer [see issue 3](https://github.com/wassname/rl-portfolio-management/issues/3)

# TODO

See issue [#4](https://github.com/wassname/rl-portfolio-management/issues/4) and [#2](https://github.com/wassname/rl-portfolio-management/issues/2) for ideas on where to go from here
