# Bayesian Probabilistic Matrix Factorization on MovieLens 1M

A movie-rating recommender built as a probabilistic graphical model in [Pyro](https://pyro.ai/). It predicts how a user will rate a movie from the latent factors learned for both, and it reports how uncertain each prediction is.

Group project for UC Irvine's CS 179 (Probabilistic Graphical Models), June 2025, by Kevin Chen, Michael Cherian Thomas and Suyash Jain.

## Data

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/): 1,000,209 ratings from 6,040 users on 3,706 movies, split 80/20 into 800,167 training and 200,042 test ratings (random seed 42). The raw data ships as `ml-1m.zip`.

## Models

All three are trained with stochastic variational inference (SVI, `Trace_ELBO`, Adam).

1. **Bias baseline.** A rating is a global mean plus a per-user and a per-movie bias, with Gaussian noise.
2. **PMF.** Each user and each movie gets a K-dimensional latent vector with a Gaussian prior, and a rating is their dot product plus Gaussian noise. The posterior is approximated with an `AutoDiagonalNormal` guide.
3. **PMF + bias.** The PMF model with the baseline's mean and bias terms added back in.

## Results

Test-set RMSE (lower is better):

| Model | RMSE |
|---|---|
| Bias baseline | 0.9120 |
| PMF, K = 5 | 0.9312 |
| PMF, K = 10 | 0.9386 |
| PMF, K = 20 | 0.9414 |
| PMF, K = 50 | 0.9263 |
| **PMF + bias, K = 50** | **0.8677** |

Adding the bias terms to the best PMF model lowered RMSE by 0.070. Plain PMF never beat the bias baseline on its own; most of the gain comes from combining the two.

Training PMF (K = 50) on a fraction of the training set:

| Training data | RMSE |
|---|---|
| 20% | 1.1685 |
| 50% | 0.9810 |
| 80% | 0.9412 |

The notebook also plots training and validation RMSE every 10 SVI steps, and compares each test prediction's posterior standard deviation against its squared error to check whether the model's uncertainty tracks its mistakes.

## Files

- `FinalProjectOneMillionTrial.ipynb`: data loading, all three models, the sweeps and the plots
- `FinalProjectOneMillionTrial.pdf`: the notebook rendered with its outputs
- `ml-1m.zip`: the MovieLens 1M dataset

## Running it

```bash
pip install torch pyro-ppl numpy pandas scikit-learn matplotlib
jupyter notebook FinalProjectOneMillionTrial.ipynb
```

The first cells unzip `ml-1m.zip` into `ml-1m/`.
