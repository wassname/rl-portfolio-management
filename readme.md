Replicating [Jiang 2017](https://arxiv.org/abs/1706.10059)

The main differences are:
- I have not used portfolio vector memory, which could lead to it incurring large trading costs, but it still should produce good results (with large and unrealistic rebalancings).

Author: wassname
Liscense: AGPLv3

# Installing

- git clone <repo>
- cd <repo>
- `pip install -r requirements/requirements.txt`
- `jupyter-notebook`

# Testing

Partial test coverage is included

- python -m pytest
