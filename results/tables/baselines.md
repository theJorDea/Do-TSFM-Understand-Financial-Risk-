# Baseline results

Generated from `results/baselines_full.parquet` — 6,632,892 evaluated forecasts, 21 series, 2003-11-03 to 2026-06-28.


## Volatility forecasting (QLIKE, h=1, lower is better)

| model         |   QLIKE |           n |
|:--------------|--------:|------------:|
| gjr-garch11-t | -7.9712 | 105445.0000 |
| garch11-n     | -7.9551 | 105445.0000 |
| garch11-t     | -7.9549 | 105445.0000 |
| fhs-ewma      | -7.9457 | 105445.0000 |
| ewma-rm       | -7.9407 | 105445.0000 |
| har-proxy     | -7.8222 | 105445.0000 |
| historical    | -7.6213 | 105445.0000 |


### Ranking with formal tests (QLIKE, h=1)

| model         |   mean_loss |   vs_best_p | in_MCS   |
|:--------------|------------:|------------:|:---------|
| gjr-garch11-t |     -7.6957 |    nan      | True     |
| garch11-n     |     -7.6886 |      0.3208 | True     |
| garch11-t     |     -7.6832 |      0.0000 | False    |
| fhs-ewma      |     -7.6776 |      0.0022 | False    |
| ewma-rm       |     -7.6735 |      0.0001 | False    |
| har-proxy     |     -7.5706 |      0.0001 | False    |
| historical    |     -7.3691 |      0.0000 | False    |


**Model Confidence Set (90%):** garch11-n, gjr-garch11-t


## Tail risk, h=1, alpha=1% (FZ0 lower is better; pass_* = share of series passing at 5%)

| model         |     FZ0 |   breach_rate |   pass_kupiec |   pass_christoffersen |   pass_dq |
|:--------------|--------:|--------------:|--------------:|----------------------:|----------:|
| gjr-garch11-t | -3.2181 |        0.0118 |        0.6667 |                1.0000 |    0.5714 |
| garch11-t     | -3.2105 |        0.0118 |        0.6667 |                0.9048 |    0.4762 |
| fhs-ewma      | -3.1903 |        0.0113 |        0.9048 |                0.7143 |    0.2857 |
| garch11-n     | -3.1366 |        0.0166 |        0.0476 |                0.8095 |    0.1429 |
| ewma-rm       | -3.0549 |        0.0200 |        0.0000 |                0.8095 |    0.0000 |
| har-proxy     | -3.0433 |        0.0158 |        0.0952 |                0.6190 |    0.0000 |
| historical    | -2.8983 |        0.0125 |        0.5238 |                0.1429 |    0.0952 |



## Tail risk, h=5, alpha=1% (FZ0 lower is better; pass_* = share of series passing at 5%)

| model         |     FZ0 |   breach_rate |   pass_kupiec |   pass_christoffersen |   pass_dq |
|:--------------|--------:|--------------:|--------------:|----------------------:|----------:|
| gjr-garch11-t | -2.3743 |        0.0124 |        0.5714 |                0.0000 |    0.0000 |
| garch11-t     | -2.3612 |        0.0121 |        0.6190 |                0.0000 |    0.0000 |
| fhs-ewma      | -2.3186 |        0.0139 |        0.3333 |                0.0000 |    0.0000 |
| garch11-n     | -2.2826 |        0.0166 |        0.1429 |                0.0000 |    0.0000 |
| har-proxy     | -2.2313 |        0.0151 |        0.2381 |                0.0000 |    0.0000 |
| ewma-rm       | -2.1740 |        0.0207 |        0.0476 |                0.0000 |    0.0000 |
| historical    | -2.0658 |        0.0131 |        0.4762 |                0.0000 |    0.0000 |



## Tail risk, h=20, alpha=1% (FZ0 lower is better; pass_* = share of series passing at 5%)

| model         |     FZ0 |   breach_rate |   pass_kupiec |   pass_christoffersen |   pass_dq |
|:--------------|--------:|--------------:|--------------:|----------------------:|----------:|
| gjr-garch11-t | -1.5821 |        0.0106 |        0.4762 |                0.0476 |    0.0000 |
| garch11-t     | -1.5799 |        0.0107 |        0.5714 |                0.0000 |    0.0000 |
| fhs-ewma      | -1.4945 |        0.0140 |        0.2857 |                0.0000 |    0.0000 |
| garch11-n     | -1.4929 |        0.0143 |        0.2857 |                0.0000 |    0.0000 |
| har-proxy     | -1.4689 |        0.0133 |        0.3810 |                0.0000 |    0.0000 |
| ewma-rm       | -1.3693 |        0.0205 |        0.0952 |                0.0000 |    0.0000 |
| historical    | -1.0462 |        0.0166 |        0.1905 |                0.0000 |    0.0000 |



### Ranking with formal tests (FZ0, h=1, alpha=1%)

| model         |   mean_loss |   vs_best_p | in_MCS   |
|:--------------|------------:|------------:|:---------|
| gjr-garch11-t |     -3.0713 |    nan      | True     |
| garch11-t     |     -3.0676 |      0.6028 | True     |
| fhs-ewma      |     -3.0540 |      0.2222 | True     |
| garch11-n     |     -3.0023 |      0.0000 | False    |
| har-proxy     |     -2.9265 |      0.0000 | False    |
| ewma-rm       |     -2.9231 |      0.0000 | False    |
| historical    |     -2.7867 |      0.0000 | False    |


**Model Confidence Set (90%):** fhs-ewma, garch11-t, gjr-garch11-t


## Regime split (FZ0, h=1, alpha=1%)

| model         |   ('FZ0', 'calm') |   ('FZ0', 'crisis') | ('in_MCS', 'calm')   | ('in_MCS', 'crisis')   |
|:--------------|------------------:|--------------------:|:---------------------|:-----------------------|
| ewma-rm       |           -3.0915 |             -2.0272 | False                | False                  |
| fhs-ewma      |           -3.1680 |             -2.4469 | True                 | True                   |
| garch11-n     |           -3.1546 |             -2.1911 | False                | False                  |
| garch11-t     |           -3.1886 |             -2.4233 | True                 | True                   |
| gjr-garch11-t |           -3.1864 |             -2.4586 | True                 | True                   |
| har-proxy     |           -3.1248 |             -1.8712 | False                | False                  |
| historical    |           -2.9907 |             -1.7006 | False                | False                  |


## Per asset class (FZ0, h=1, alpha=1%; * = in MCS)

| model         | bond_etf   | commodity   | crypto   | equity_index   | equity_single   | fx       |
|:--------------|:-----------|:------------|:---------|:---------------|:----------------|:---------|
| ewma-rm       | -3.7972*   | -2.7127     | -1.5977  | -3.0831        | -2.7191         | -3.8035* |
| fhs-ewma      | -3.8848*   | -2.8758*    | -1.8637* | -3.2551        | -2.8365         | -3.8902* |
| garch11-n     | -3.8676*   | -2.7737     | -1.7952* | -3.1937        | -2.7928         | -3.8459* |
| garch11-t     | -3.9095*   | -2.8756*    | -1.8488* | -3.2904*       | -2.8741*        | -3.8763* |
| gjr-garch11-t | -3.9220*   | -2.8806*    | -1.8262* | -3.3117*       | -2.8834*        | -3.8674* |
| har-proxy     | -3.5990*   | -2.5703     | -1.8882* | -3.1559        | -2.7473         | -3.6925* |
| historical    | -3.5063    | -2.5754     | -1.9170* | -2.8791        | -2.5530         | -3.6557  |