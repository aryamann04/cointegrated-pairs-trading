import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

from BuySellSignals import generate_dataframes, generate_signals, profit, leveraged_profit
from Indicator import generate_signals_indicator, calculate_indicator
from PerformanceMetrics import performance

#----------------------------------------------------#
#                   Data set-up                      #
#----------------------------------------------------#

# Results from equity pair screening
data = {
    'Ticker 1': ['MLM', 'IBM', 'KLAC', 'ETN', 'ADI', 'IEX', 'LH', 'AMGN', 'CTAS', 'IBM', 'STZ', 'NSC', 'DOV', 'ANET', 'AME', 'NDSN', 'DHR', 'LIN', 'CTAS', 'DRI'],
    'Ticker 2': ['PH', 'VRSN', 'NDSN', 'JBL', 'ODFL', 'MSCI', 'QCOM', 'RSG', 'MOH', 'SYK', 'HUBB', 'ZBRA', 'MSFT', 'GWW', 'TSCO', 'TMUS', 'SHW', 'WMT', 'PGR', 'HON'],
    'Ratio Variance': [0.005718, 0.005133, 0.004253, 0.004090, 0.003839, 0.003498, 0.003237, 0.002821, 0.002727, 0.002554, 0.002547, 0.002400, 0.002009, 0.001832, 0.001695, 0.001329, 0.001023, 0.000782, 0.000750, 0.000740],
    'Cointegration Test P-Value': [0.026953, 0.003023, 0.000741, 0.011188, 0.001841, 0.018063, 0.004475, 0.024894, 0.015033, 0.018365, 0.016497, 0.005069, 0.017830, 0.013301, 0.015155, 0.007882, 0.027492, 0.026370, 0.014683, 0.024686],
    'Ranking': [191, 372, 288, 345, 136, 318, 204, 314, 146, 116, 390, 59, 85, 69, 322, 138, 22, 222, 364, 379]
}
ranked_df = pd.DataFrame(data)

# Results from commodities screening (optional)
selected_commodity = {
    'Ticker 1': ['PPLT', 'WEAT', 'CORN', 'CORN', 'CPER', 'WEAT', 'BCIM'],
    'Ticker 2': ['PALL', 'UNL', 'DBB', 'USL', 'SLV', 'DBB', 'DBB'],
    'Ratio Variance': [0.004637, 0.003130, 0.002529, 0.002384, 0.001787, 0.001783, 0.000063],
    'Cointegration Test P-Value': [0.120510, 0.007510, 0.153944, 0.111754, 0.212381, 0.069488, 0.054788],
    'Ranking': [34, 18, 8, 81, 1, 3, 64]
}
ranked_df_commodity = pd.DataFrame(selected_commodity)

#----------------------------------------------------#
#               Plotting Functions                   #
#----------------------------------------------------#
def plot_indicator_strategy(price_df, signals_df, indicator_lower_bound, indicator_upper_bound, show_one_year=True):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price Ratio', color='black')
    ax1.plot(price_df.index, price_df['Raw Price Data 1'] / price_df['Raw Price Data 2'], color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([
        price_df['Raw Price Data 1'].min() / price_df['Raw Price Data 2'].max(),
        price_df['Raw Price Data 1'].max() / price_df['Raw Price Data 2'].min()
    ])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Indicator', color='tab:blue')
    ax2.plot(signals_df.index, signals_df['Indicator'], color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.axhline(y=indicator_lower_bound, color='gray', linestyle='--')
    ax2.axhline(y=indicator_upper_bound, color='gray', linestyle='--')

    for index, row in signals_df.iterrows():
        if row['orders_ticker1'] == 1:
            ax1.scatter(index, price_df.loc[index, 'Raw Price Data 1'] / price_df.loc[index, 'Raw Price Data 2'], color='green', marker='^')
        elif row['orders_ticker2'] == 1:
            ax1.scatter(index, price_df.loc[index, 'Raw Price Data 1'] / price_df.loc[index, 'Raw Price Data 2'], color='red', marker='v')

    if show_one_year:
        ax1.set_xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01'))

    fig.tight_layout()
    plt.title(f"{price_df['Ticker 1'].iloc[0]}/{price_df['Ticker 2'].iloc[0]} Indicator Strategy")
    plt.show()


def plot_portfolio_value(portfolio_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(portfolio_df['Date'], portfolio_df['Cumulative_Return'], label='Portfolio', color='blue')

    sp500_data = yf.download('^GSPC', start='2017-01-01', end='2024-01-01', progress=False)
    sp500_returns = sp500_data['Close'].pct_change().dropna()
    sp500_cumulative = (1 + sp500_returns).cumprod()

    ax.plot(sp500_cumulative.index, sp500_cumulative, label='S&P 500', color='black', linestyle='--')
    ax.set_title('Portfolio vs S&P 500 Cumulative Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    plt.show()

#----------------------------------------------------#
# Run strategy function:                             #
#                                                    #
#  ma: moving average/look-back window               #
#  use_leverage: boolean, if leveraged or not        #
#  leverage_ratio: leverage multiplier               #
#  use_initial_z_score: boolean, use z-score         #
#  z_value: critical z-score for buy/sell trigger    #
#  use_indicator_strategy: boolean, use indicator    #
#  lower_bound: indicator cutoff for long signal     #
#  upper_bound: indicator cutoff for short signal    #
#----------------------------------------------------#
def run_strategy(ma, df_list, use_leverage=False, leverage_ratio=False, use_initial_z_score=False, z_value=None,
                 use_indicator_strategy=False, lower_bound=None, upper_bound=None):

    all_dates = pd.date_range(start='2017-01-01', end='2024-01-01', freq='B')
    pair_returns = []

    for df in df_list:
        if use_initial_z_score:
            signals_df, paired_trades = generate_signals(df, z_value, ma)
        elif use_indicator_strategy:
            signals_df, paired_trades = generate_signals_indicator(df, lower_bound, upper_bound, ma)
        else:
            raise ValueError("Please choose either initial z-score strategy or indicator strategy.")

        if paired_trades.empty:
            continue

        profit_df = leveraged_profit(profit(paired_trades, df), leverage_ratio) if use_leverage else profit(paired_trades, df)

        if profit_df.empty:
            continue

        profit_df['Date2'] = pd.to_datetime(profit_df['Date2'], format='%m-%d-%Y')
        daily = profit_df.groupby('Date2')['Pair_Profit'].sum()
        pair_returns.append(daily)

    if pair_returns:
        combined = pd.concat(pair_returns, axis=1).reindex(all_dates).fillna(0)
        daily_returns = combined.sum(axis=1) / len(df_list)
    else:
        daily_returns = pd.Series(0.0, index=all_dates)

    portfolio_data = pd.DataFrame({
        'Date': all_dates,
        'Daily_Return': daily_returns.values
    })
    portfolio_data['Cumulative_Return'] = (1 + portfolio_data['Daily_Return']).cumprod()
    return portfolio_data

#----------------------------------------------------#
#               Parameter Optimization              #
#----------------------------------------------------#
def optimize_params(df_list, optimize_for_sharpe=False, use_indicator_strategy=False):
    best_lower_bound = None
    best_upper_bound = None
    best_metric_value = float('-inf')
    best_z_score = None
    best_ma = None

    if use_indicator_strategy:
        for lower_bound in range(5, 35, 5):
            for upper_bound in range(30, 55, 5):
                print(f"Testing Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
                portfolio_df = run_strategy(50, df_list, use_indicator_strategy=True,
                                            lower_bound=lower_bound, upper_bound=upper_bound)
                performance_dict = performance(portfolio_df)
                metric_value = float(performance_dict["Sharpe Ratio"]) if optimize_for_sharpe else float(performance_dict["CAGR"])
                if metric_value > best_metric_value:
                    best_lower_bound = lower_bound
                    best_upper_bound = upper_bound
                    best_metric_value = metric_value
    else:
        for z_score in [x * 0.2 + 0.3 for x in range(1, 12)]:
            for ma in range(25, 200, 25):
                print(f"Testing Z-Score: {z_score}, Moving Average: {ma}")
                portfolio_df = run_strategy(ma, df_list, use_initial_z_score=True, z_value=z_score)
                performance_dict = performance(portfolio_df)
                metric_value = float(performance_dict["Sharpe Ratio"]) if optimize_for_sharpe else float(performance_dict["CAGR"])
                if metric_value > best_metric_value:
                    best_z_score = z_score
                    best_ma = ma
                    best_metric_value = metric_value

    print("\nOptimization Result:")
    if use_indicator_strategy:
        print(f"Best Lower Bound: {best_lower_bound}")
        print(f"Best Upper Bound: {best_upper_bound}")
    else:
        print(f"Best Z-Score: {best_z_score}")
        print(f"Best Moving Average: {best_ma}")
    print(f"Best Metric Value: {best_metric_value}")

    return (best_lower_bound, best_upper_bound) if use_indicator_strategy else (best_z_score, best_ma)

#----------------------------------------------------#
#                  Entry Point                       #
#----------------------------------------------------#
if __name__ == '__main__':
    df_list = generate_dataframes(ranked_df)

    indicator_portfolio = run_strategy(50, df_list, use_indicator_strategy=True, lower_bound=45, upper_bound=55)
    zscore_portfolio = run_strategy(50, df_list, use_initial_z_score=True, z_value=1.5)

    print("\n--- Indicator Strategy ---")
    performance(indicator_portfolio)

    print("\n--- Z-Score Strategy ---")
    performance(zscore_portfolio)

    plot_portfolio_value(indicator_portfolio)
    plot_portfolio_value(zscore_portfolio)

    # Uncomment to run parameter optimization:
    # optimize_params(df_list, optimize_for_sharpe=True, use_indicator_strategy=True)
    # optimize_params(df_list, optimize_for_sharpe=True, use_indicator_strategy=False)

    # Uncomment to run commodity pairs:
    # df_list_commodity = generate_dataframes(ranked_df_commodity)
    # commodity_portfolio = run_strategy(50, df_list_commodity, use_indicator_strategy=True, lower_bound=45, upper_bound=55)
    # performance(commodity_portfolio)
