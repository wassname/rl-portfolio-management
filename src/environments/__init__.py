import pandas as pd
from gym.envs.registration import register

df_train = pd.read_hdf('./data/poloniex_30m.hf', key='train')

register(
    id='PortfolioMLP-v0',
    entry_point='src.environments.portfolio:PortfolioEnv',
    kwargs=dict(
        output_mode='mlp',
        df=df_train
    )
)

register(
    id='PortfolioEIIE-v0',
    entry_point='src.environments.portfolio:PortfolioEnv',
    kwargs=dict(
        output_mode='EIIE',
        df=df_train
    )
)

register(
    id='PortfolioAtari-v0',
    entry_point='src.environments.portfolio:PortfolioEnv',
    kwargs=dict(
        output_mode='atari',
        df=df_train
    )
)
