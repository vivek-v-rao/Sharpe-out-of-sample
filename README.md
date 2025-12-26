# Sharpe-out-of-sample
Formulas for out-of-sample Sharpe ratio under estimation error from [In-Sample and Out-of-Sample Sharpe Ratios of Multi-Factor Asset Pricing Models (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3454628) by Kan, Wang, and Zheng.

`python xsharpe.py` gives

```
wrote: fig1_replication.png
M: 200000

theta: 0.1
T: 120
N: 6

Sharpe ratios:
           tilde    hat    diff
exact     0.0391 0.2429 -0.2038
simulated 0.0391 0.2432 -0.2041

theta: 0.1
T: 1000
N: 6

Sharpe ratios:
           tilde    hat    diff
exact     0.0799 0.1237 -0.0438
simulated 0.0798 0.1238 -0.0439

theta: 0.1
T: 120
N: 10

Sharpe ratios:
           tilde    hat    diff
exact     0.0310 0.3134 -0.2825
simulated 0.0309 0.3137 -0.2828

theta: 0.1
T: 1000
N: 10

Sharpe ratios:
           tilde    hat    diff
exact     0.0708 0.1396 -0.0688
simulated 0.0707 0.1396 -0.0689

theta: 0.5
T: 120
N: 6

Sharpe ratios:
           tilde    hat    diff
exact     0.4516 0.5584 -0.1068
simulated 0.4515 0.5586 -0.1070

theta: 0.5
T: 1000
N: 6

Sharpe ratios:
           tilde    hat    diff
exact     0.4938 0.5069 -0.0131
simulated 0.4938 0.5069 -0.0131

theta: 0.5
T: 120
N: 10

Sharpe ratios:
           tilde    hat    diff
exact     0.4197 0.6005 -0.1808
simulated 0.4196 0.6007 -0.1811

theta: 0.5
T: 1000
N: 10

Sharpe ratios:
           tilde    hat    diff
exact     0.4890 0.5119 -0.0229
simulated 0.4890 0.5119 -0.0230
```
