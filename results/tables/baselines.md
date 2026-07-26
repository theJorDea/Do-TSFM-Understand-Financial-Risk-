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
| gjr-garch11-t |     -7.9712 |    nan      | True     |
| garch11-n     |     -7.9551 |      0.0000 | False    |
| garch11-t     |     -7.9549 |      0.0000 | False    |
| fhs-ewma      |     -7.9457 |      0.0000 | False    |
| ewma-rm       |     -7.9407 |      0.0000 | False    |
| har-proxy     |     -7.8222 |      0.0000 | False    |
| historical    |     -7.6213 |      0.0000 | False    |


**Model Confidence Set (90%):** gjr-garch11-t


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
| gjr-garch11-t |     -3.2181 |    nan      | True     |
| garch11-t     |     -3.2105 |      0.0932 | False    |
| fhs-ewma      |     -3.1903 |      0.0001 | False    |
| garch11-n     |     -3.1366 |      0.0000 | False    |
| ewma-rm       |     -3.0549 |      0.0000 | False    |
| har-proxy     |     -3.0433 |      0.0000 | False    |
| historical    |     -2.8983 |      0.0000 | False    |


**Model Confidence Set (90%):** gjr-garch11-t


## Regime split (FZ0, h=1, alpha=1%)

| model         |   ('FZ0', 'calm') |   ('FZ0', 'crisis') | ('in_MCS', 'calm')   | ('in_MCS', 'crisis')   |
|:--------------|------------------:|--------------------:|:---------------------|:-----------------------|
| ewma-rm       |           -3.2035 |             -2.2937 | False                | False                  |
| fhs-ewma      |           -3.3014 |             -2.6213 | False                | True                   |
| garch11-n     |           -3.2836 |             -2.3841 | False                | False                  |
| garch11-t     |           -3.3308 |             -2.5943 | True                 | False                  |
| gjr-garch11-t |           -3.3299 |             -2.6454 | True                 | True                   |
| har-proxy     |           -3.2474 |             -1.9976 | False                | False                  |
| historical    |           -3.1241 |             -1.7417 | False                | False                  |


## Per asset class (FZ0, h=1, alpha=1%; * = in MCS)

| model         | bond_etf   | commodity   | crypto   | equity_index   | equity_single   | fx       |
|:--------------|:-----------|:------------|:---------|:---------------|:----------------|:---------|
| ewma-rm       | -3.8073*   | -2.7122     | -1.6971  | -3.0705        | -2.7191         | -3.8364* |
| fhs-ewma      | -3.9064*   | -2.8758*    | -1.9368* | -3.2503        | -2.8365         | -3.9157* |
| garch11-n     | -3.8903*   | -2.7734     | -1.8998* | -3.1856        | -2.7928         | -3.8652* |
| garch11-t     | -3.9365*   | -2.8755*    | -1.9457* | -3.2856*       | -2.8741*        | -3.8958* |
| gjr-garch11-t | -3.9499*   | -2.8805*    | -1.9369* | -3.3066*       | -2.8834*        | -3.8862* |
| har-proxy     | -3.5961*   | -2.5698     | -1.9647* | -3.1460        | -2.7473         | -3.7252* |
| historical    | -3.5277    | -2.5751     | -1.9602* | -2.8867        | -2.5530         | -3.6609  |