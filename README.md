# IPA
IPA (Investment Portfolio Analyser). This dashboard provides optimisation of weights within a portfolio dependent on different optimisation criteria.

Given a portfolio (see example for format) this program will calculate the max Sharpe Ratio portfolio and minimal conditional Value at Risk portfolio, these can be benchmarked against your own index fund, or the same portfolio but optimised in different ways (inverse volatility, equally weighted etc).

This program is just to demonstrate the uses of Python for portfolio optimisation, and should not be taken as financial advice.

The program could be improved through the use of a preselection transformer to eliminate coindependence of assets before optimisation, as this will yield a better model under Mean-Variance Optimisation.

Portfolio optimality is not a well defined problem, and therefore can yield unexpected results when given correlated assets.
