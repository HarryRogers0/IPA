import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt 
import numpy as np 
import streamlit as st 
import datetime
import time
import altair as alt
from plotly.io import show
from skfolio import Portfolio, RiskMeasure, Population, PerfMeasure,RatioMeasure, measures
from skfolio.preprocessing import prices_to_returns
from sklearn.model_selection import train_test_split
from skfolio.optimization import InverseVolatility, MeanRisk, ObjectiveFunction, EqualWeighted, Random
from skfolio.prior import EmpiricalPrior
Failed_Downloads = []
def downloadData(Tickers, start):

    for ticker in Tickers:
        ticker_data = yf.download(ticker, start=start)
        data[ticker] = prices_to_returns(pd.DataFrame(ticker_data['Adj Close']))
        if ticker_data.empty:
            Failed_Downloads.append(ticker)
            
    if Failed_Downloads != []:
        st.error(f"The following stocks failed to download {Failed_Downloads}")
    return data

def downloadBench(benchmarkTicker):
    benchmark_data = prices_to_returns(pd.DataFrame(yf.download(benchmarkTicker, start = start)['Adj Close']))
    return benchmark_data



def cov_calc(Portfolio1, Portfolio2):
    # input Portfolio Objects defined by skfolio
    portfolio_returns = Portfolio1.returns
    benchmark_returns = Portfolio2.returns
    
    portfolio_returns = portfolio_returns[-len(benchmark_returns):]
    benchmark_returns = benchmark_returns[-len(portfolio_returns):]
    covariance = np.cov(portfolio_returns, benchmark_returns, rowvar=False)
    
    beta = covariance[0, 1] / covariance[1, 1] 
    var_portfolio1 = covariance[0,0]
    var_portfolio2 = covariance[1,1]
    return beta, var_portfolio1, var_portfolio2


def Metric_Calc(Portfolio1, Benchmark_portfolio):
    # Calculate metrics for Portfolio 1
    sharpe_ratio = Portfolio1.annualized_sharpe_ratio
    cumulative_return = Portfolio1.cumulative_returns
    cvar = measures.cvar(Portfolio1.returns, beta = 0.95) * 100
    beta, var_portfolio1, var_portfolio2 = cov_calc(Portfolio1, Benchmark_portfolio)
    volatility = np.sqrt(var_portfolio1 * 252)

    return sharpe_ratio, cumulative_return, cvar, volatility

def determine_comparison_text(delta):
    return "Above" if delta > 0 else "Below"

def Max_Sharpe_Model(X_train, X_test, investment_horizon, rfr):
        model = MeanRisk(
        risk_measure=RiskMeasure.ANNUALIZED_VARIANCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        portfolio_params=dict(name="Max Sharpe"),
        prior_estimator=EmpiricalPrior(
            is_log_normal=True, investment_horizon=investment_horizon),
        risk_free_rate=rfr
        )
        model.fit(X_train)
        analysis_Portfolio = model.predict(X_test) 
        return analysis_Portfolio
    
    
def Max_Sharpe_Model2(X_train, X_test, investment_horizon, rfr):
    
        eff_front_model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        efficient_frontier_size=300,
        portfolio_params=dict(name="Max Sharpe"),
        risk_free_rate = rfr,
        prior_estimator=EmpiricalPrior(
            is_log_normal=True, investment_horizon=investment_horizon),
        )
        eff_front_model.fit(X_train)
        eff_front_test = eff_front_model.predict(X_test)
        idx = np.argmax(eff_front_test.measures(measure = RatioMeasure.ANNUALIZED_SHARPE_RATIO))
        analysis_Portfolio = eff_front_test[idx]
        return analysis_Portfolio
        
        
        
    
def Minimise_CVar(X_train, X_test, investment_horizon, rfr):
        model = MeanRisk(
        risk_measure=RiskMeasure.CVAR,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        portfolio_params=dict(name="Min CVaR"),
        prior_estimator=EmpiricalPrior(
            is_log_normal=True, investment_horizon=investment_horizon),
        risk_free_rate=rfr
        )
        model.fit(X_train)
        analysis_Portfolio = model.predict(X_test)
        return analysis_Portfolio
    

st.title("Investment Portfolio Analysis")

data = {
    'Ticker': ['NVDA', 'AAPL', 'MSFT', 'GME'],
    'Weight': [0.15, 0.2, 0.5, 0.15],
}
data = pd.DataFrame(data)


with st.expander("Instructions for use"):
    st.write("This program assumes all stocks were bought at date of portfolio inception.")
    st.write("This program will analyse a stock portfolio for a given CSV. The CSV must be in the following convention (without the 0,1,2,3 indexing)")
    st.write(data)
    st.write("Important to note that only stocks available on Yahoo Finance can be used.")
    st.write("This dashboard will take your portfolio and weights, downloads market data and optimises portfolio weights based on different criteria.")
    st.write("To get started, you can compare your imported portfolio against any individual asset (consider choosing index funds to compare performance metrics against market)")
    st.write("Alternatively, choose to compare your portfolio performance against the same portfolio but under different criteria. For example, what are the differences between a portfolio optimised on Sharpe Ratio versus equally weighted?")
    
    st.warning("This tool should not be used for financial decision-making. This is a project focussed on providing insight on how portfolio optimisation can be implemented within a Python environment. In reality, Mean Variance Optimisation does not perform well and input assumptions can make large differences in optimal portfolio output. This is because portfolio optimality is not well defined, however more sophisticated models such as Resampled Efficient Frontier can take into account these shortfalls. \n Another consideration is whether the tickers given are highly correlated, if they are highly correlated this can lead to suboptimal (and sometimes completely wrong) optimisation, ways to avoid this would be to run a preselection transformer on the data and drop correlated assets.")
    

data.set_index('Ticker', inplace=True)


with st.sidebar:
    Portfolio_Weights = st.file_uploader("Upload CSV with Stock ticker and portfolio weight", type ={"csv"})        
    portfolio_choice = st.radio("Portfolio optimisation method", ["Imported Portfolio", "Maximise Sharpe Ratio", "Minimise CVar"] )
    benchmark_choice = st.radio("Benchmarking portfolio", ["Custom Ticker", "Equally Weighted Portfolio", "Inverse Volatility Portfolio", "Maximise Sharpe Ratio", "Minimise CVar"])
    if benchmark_choice == "Custom Ticker": benchmarkTicker = st.text_input("Select a ticker to benchmark your portfolio against", value = "^GSPC")
    start = st.date_input("Date of Portfolio Start", value = pd.to_datetime('2015-01-01'))
    investment_horizon = st.radio("Select investment horizon:", ['3M', '6M', '1Y', '5Y', '10Y'])
    rfr = st.slider("Select a risk free rate", 0.0, 10.0, 3.0, 0.5)/100
    dictionary = {
                  '3M' : 252 / 4,
                  '6M' : 252 / 2,
                  '1Y' : 252,
                  '5Y' : 252 * 2,
                  '10Y' : 252 * 10
                  } 

    investment_horizon = dictionary.get(investment_horizon, None)
    
if Portfolio_Weights is None:
    Portfolio_Weights = "Example Portfolio.csv"

if Portfolio_Weights is not None:
    Portfolio_Weights = pd.read_csv(Portfolio_Weights)
    Portfolio_df = pd.DataFrame(Portfolio_Weights)
    Tickers = Portfolio_Weights['Ticker']
    Tickers = pd.DataFrame(Tickers)
    Tickers = Tickers['Ticker'].tolist()
    data = pd.DataFrame()
    W = Portfolio_Weights['Weight']

    
        
    with st.spinner('Fetching data...'):
        
        data = downloadData(Tickers, start)
        if 'benchmarkTicker' in locals():
            benchmark_data = downloadBench(benchmarkTicker)
        data = data.dropna()
    X_train, X_test = train_test_split(data, test_size = 0.33, shuffle = False)
    
    if Failed_Downloads != []:
        st.error(f"The following stocks failed to download {Failed_Downloads}")


    match portfolio_choice:
        case "Imported Portfolio":
            analysis_Portfolio = Portfolio(X = X_test, weights = Portfolio_Weights['Weight'], name = "Imported Portfolio", risk_free_rate = rfr)
            # error here
        case "Maximise Sharpe Ratio":
            analysis_Portfolio = Max_Sharpe_Model2(X_train, X_test, investment_horizon, rfr)
            
        case "Minimise CVar":
            analysis_Portfolio = Minimise_CVar(X_train, X_test, investment_horizon, rfr)




    match benchmark_choice:
        case "Custom Ticker": # 'Equally Weighted' model applied to one asset. This ensures consistency for outputs across models.
            benchmark_data =  benchmark_data.rename(columns={'Adj Close' : benchmarkTicker})
            X_bench_train, X_bench_test = train_test_split(benchmark_data, test_size = 0.33, shuffle = False)        
            benchmark = EqualWeighted(portfolio_params=dict(name="Custom Benchmark", risk_free_rate = rfr))
            benchmark.fit(X_bench_train)
            Benchmark_portfolio = benchmark.predict(X_bench_test)
                 
            
        case "Inverse Volatility Portfolio":
            benchmark = InverseVolatility(portfolio_params=dict(name="Inverse Vol", risk_free_rate = rfr))
            benchmark.fit(X_train)
            Benchmark_portfolio = benchmark.predict(X_test)
            
            
        case "Equally Weighted Portfolio":
            benchmark = EqualWeighted(portfolio_params=dict(name="Equal Weighted", risk_free_rate = rfr))
            benchmark.fit(X_train)
            Benchmark_portfolio = benchmark.predict(X_test)
            
            
        case "Maximise Sharpe Ratio":
            Benchmark_portfolio = Max_Sharpe_Model2(X_train, X_test, investment_horizon, rfr)
            
        case "Minimise CVar":
            Benchmark_portfolio = Minimise_CVar(X_train, X_test, investment_horizon, rfr)

    sharpe_ratio_analysed, cumulative_return_analysed, cVar_analysed, volatility_analysed = Metric_Calc(analysis_Portfolio, Benchmark_portfolio)

    sharpe_ratio_benchmark, cumulative_return_benchmark, cVar_benchmark, volatility_benchmark = Metric_Calc(Benchmark_portfolio, Benchmark_portfolio)
    
    
    annual_cumulative_return = (1 + cumulative_return_analysed[-1])**(252 / len(X_test) ) - 1
    annual_cumulative_return_benchmark = (1 + cumulative_return_benchmark[-1])**(252 / len(X_test) ) - 1
    
    
    sharpe_delta = round(sharpe_ratio_analysed - sharpe_ratio_benchmark,3)
    cVar_delta = round(cVar_analysed - cVar_benchmark, 2)
    cum_ret_delta = round(100*(annual_cumulative_return - annual_cumulative_return_benchmark),2)
    vol_delta = round(volatility_analysed - volatility_benchmark,2)

    wor = determine_comparison_text(sharpe_delta)
    wor2 = determine_comparison_text(cVar_delta)
    wor3 = determine_comparison_text(cum_ret_delta)
    wor4 = determine_comparison_text(vol_delta)

    # Display main portfolio metrics

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualised Sharpe Ratio", round(sharpe_ratio_analysed,3), f"{round(sharpe_delta,3)} {wor} \n Benchmark")
    col2.metric("Expected Shortfall (CVar)", f"{round(cVar_analysed,2)}%", f"{round(cVar_delta,2)}% {wor2} \n Benchmark", delta_color = "inverse")
    col3.metric("Annual Return", f"{round(100*annual_cumulative_return,2)}%", f"{round(cum_ret_delta,2)}% {wor3} \n Benchmark")
    col4.metric("Annual Volatility", f"{np.round(100*volatility_analysed,2)}%", f"{round(100*vol_delta,2)}% {wor4} \n Benchmark", delta_color="inverse")




    population = Population([analysis_Portfolio, Benchmark_portfolio])

    fig = population.plot_cumulative_returns()
    st.plotly_chart(fig)

    composition = population.plot_composition()
    st.plotly_chart(composition)
    
    col1,col2,col3 = st.columns(3)
    gen_eff_front = col1.button("Generate Efficient Frontier")

    summaries = population.summary()
    summaries = summaries.to_csv().encode("utf-8")
    
    weights = pd.DataFrame({'Ticker' : Tickers})
    weights[portfolio_choice] = population[0].weights
    
    if benchmark_choice != "Custom Ticker":
        weights[benchmark_choice] = population[1].weights

    weights = weights.to_csv().encode("utf-8")
    
    
    col2.download_button("Download portfolio summaries", summaries, file_name = "portfolio summaries.csv")   
    
    col3.download_button("Download portfolio weights", weights, file_name = "Portfolio weights.csv")
    
    




    if gen_eff_front:
        eff_front_model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        efficient_frontier_size=50,
        portfolio_params=dict(name="Variance"),
        risk_free_rate = rfr
        )
        eff_front_model.fit(X_train)
        eff_front_train = eff_front_model.predict(X_train)
        eff_front_test = eff_front_model.predict(X_test)
        
        eff_front_train.set_portfolio_params(tag="Train")
        eff_front_test.set_portfolio_params(tag="Test")

        eff_front = eff_front_train + eff_front_test
    
        distributions = eff_front.plot_measures(x = RiskMeasure.ANNUALIZED_VARIANCE,
                                                y = PerfMeasure.ANNUALIZED_MEAN,
                                                color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
                                                hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO])
        st.plotly_chart(distributions)
        

    