import pandas as pd
import numpy as np

print("FinScope - Risk Engine Running...")

data = pd.read_csv("clean_prices.csv", index_col="Date", parse_dates=True)

if data.isnull().all().all():
    print(" Data is empty. Run data_fetch.py again.")
    exit()

returns = np.log(data / data.shift(1)).dropna()

if returns.empty:
    print("  Returns calculation failed.")
    exit()

returns.to_csv("returns.csv")

rolling_vol = returns.rolling(30).std() * np.sqrt(252)
rolling_vol.to_csv("rolling_volatility.csv")

corr_matrix = returns.corr()
corr_matrix.to_csv("correlation_matrix.csv")

n_assets = len(returns.columns)
weights = np.ones(n_assets) / n_assets

num_simulations = 10000
days = 252
initial_value = 100000

mean_returns = returns.mean()
cov_matrix = returns.cov()

simulated_values = []

for _ in range(num_simulations):
    simulated = np.random.multivariate_normal(
        mean_returns,
        cov_matrix,
        days
    )

    portfolio = np.dot(simulated, weights)
    path = initial_value * np.cumprod(1 + portfolio)

    simulated_values.append(path[-1])

simulated_values = np.array(simulated_values)

VaR_95 = initial_value - np.percentile(simulated_values, 5)

portfolio_variance = np.dot(weights.T, np.dot(cov_matrix * 252, weights))
portfolio_volatility = np.sqrt(portfolio_variance)

risk_free_rate = 0.02
sharpe = (portfolio_volatility - risk_free_rate) / portfolio_volatility

pd.DataFrame(simulated_values, columns=["Final Portfolio Value"]).to_csv("mc_results.csv")

summary = pd.DataFrame({
    "Metric": ["VaR (95%)", "Volatility", "Sharpe Ratio"],
    "Value": [VaR_95, portfolio_volatility, sharpe]
})

summary.to_csv("portfolio_summary.csv", index=False)

print(" Risk analysis complete!")
print(summary)
