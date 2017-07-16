Replicating [Jiang 2017](https://arxiv.org/abs/1706.10059)

The main difference is that where Jian et al. takes into account the previous protfolio weights by using "Portfolio-Vector Memory" I use conversion weights. In my approach the model tells us the idea portfolio and how far to go towards it, this way it controls it's trading costs based on it's confidence. In this approach model outputs [cash_bian, w0..., c0, c1...], w0 is the first portfolio weight, and c0 is the conversion_weight for the first asset.
