Replicating [Jiang 2017](https://arxiv.org/abs/1706.10059)

The main differences are:

- I have not used portfolio vector memory, which could lead to it incurring large trading costs, but it still should produce good results (with large and unrealistic rebalancings).
- I added some random shifts as data augumentation to prevent overfitting
- The first step is to make sure the model can overfit or peform on an easier problem. So I am first trying to acheive good results with no trading costs. However there are no good results so far.

Author: wassname

License: AGPLv3

# Installing

- git clone $REPO
- cd $NAME
- `pip install -r requirements/requirements.txt`
- `jupyter-notebook`

# Testing

Partial test coverage is included

- python -m pytest
