import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import csd

def clean_price_data(ticker1, ticker2):
    data_i = yf.download(ticker1, start='2017-01-01', end='2024-01-01')['Close']
    data_j = yf.download(ticker2, start='2017-01-01', end='2024-01-01')['Close']

    data_i = data_i.dropna()
    data_j = data_j.dropna()

    data_i = pd.Series(data_i).apply(lambda x: 0 if x <= 0 else (x if x == 0 else np.log(x)))
    data_j = pd.Series(data_j).apply(lambda x: 0 if x <= 0 else (x if x == 0 else np.log(x)))

    common_dates = data_i.index.intersection(data_j.index)
    data_i = data_i[common_dates]
    data_j = data_j[common_dates]

    return data_i, data_j

def ratio_var(ticker1, ticker2):
    data1, data2 = clean_price_data(ticker1, ticker2)
    return np.var(data1 / data2)
def ssd(log_prices_1, log_prices_2):
    normalized_prices_1 = (log_prices_1 - np.mean(log_prices_1)) / np.std(log_prices_1)
    normalized_prices_2 = (log_prices_2 - np.mean(log_prices_2)) / np.std(log_prices_2)
    return np.sum((normalized_prices_1 - normalized_prices_2) ** 2)

# how close the price ratio is to 1
def price_ratio(log_prices_1, log_prices_2):
    return np.abs((log_prices_1 / log_prices_2).mean() - 1)


def corr_log_prices(log_prices_1, log_prices_2):
    return np.abs(np.corrcoef(log_prices_1, log_prices_2)[0, 1])


def corr_returns(log_prices_1, log_prices_2):
    returns_1 = np.diff(log_prices_1)
    returns_2 = np.diff(log_prices_2)
    return np.abs(np.corrcoef(returns_1, returns_2)[0, 1])


def cov_log_prices(log_prices_1, log_prices_2):
    return np.cov(log_prices_1, log_prices_2)[0, 1]


def cov_returns(log_prices_1, log_prices_2):
    returns_1 = np.diff(log_prices_1)
    returns_2 = np.diff(log_prices_2)
    return np.cov(returns_1, returns_2)[0, 1]

def magnitude_squared_coherence(x, y):
    max_len = max(len(x), len(y))
    x_padded = np.pad(x, (0, max_len - len(x)))
    y_padded = np.pad(y, (0, max_len - len(y)))

    fs = 1 / np.mean(np.diff(np.arange(max_len)))

    _, Pxy = csd(x_padded, y_padded, fs=fs, nperseg=max_len)
    CSD_at_zero = Pxy[0]
    _, Pxx = csd(x_padded, x_padded, fs=fs, nperseg=max_len)
    _, Pyy = csd(y_padded, y_padded, fs=fs, nperseg=max_len)

    MSC_at_zero = np.abs(CSD_at_zero) ** 2 / (Pxx[0] * Pyy[0])

    return MSC_at_zero

def pairs_measures(selected_tickers):
    securities_pairs = []

    # Generate pairs of securities
    for i in range(len(selected_tickers)):
        for j in range(i + 1, len(selected_tickers)):
            securities_pairs.append([selected_tickers[i], selected_tickers[j]])

    pairs_measures = {}

    for pair in securities_pairs:
        try:
            pair_tuple = tuple(pair)
            data_i, data_j = clean_price_data(pair[0], pair[1])

            measures = {
                'SSD': ssd(data_i, data_j),
                'PR': price_ratio(data_i, data_j),
                'Corr_log_prices': corr_log_prices(data_i, data_j),
                'Corr_returns': corr_returns(data_i, data_j),
                'Cov_log_prices': cov_log_prices(data_i, data_j),
                'Cov_returns': cov_returns(data_i, data_j),
                'MSC': magnitude_squared_coherence(data_i, data_j)
            }

            pairs_measures[pair_tuple] = measures
        except Exception as e:
            print(f"Error processing pair {pair}: {e}")

    return pairs_measures


if __name__ == '__main__':
    test_stocks = ['MSFT', 'META', 'GOOG', 'NFLX']
    measures = pairs_measures(test_stocks)

def rank_pairs(pair_dict):
    pairs = list(pair_dict.keys())
    measures = list(pair_dict[pairs[0]].keys())

    ranks = {param: np.argsort([pair_dict[pair][param] for pair in pairs]) + 1 for param in measures}
    total_ranks = {pair: 0 for pair in pairs}

    ranking_criteria = {
        'SSD': 'ascending',  # Lowest (best) to highest (worst)
        'PR': 'ascending',  # Closest to 1 (best) to farthest from 1 (worst)
        'Corr_log_prices': 'descending',  # Highest (best) to lowest (worst)
        'Corr_returns': 'descending',  # Highest (best) to lowest (worst)
        'Cov_log_prices': 'descending',  # Highest (best) to lowest (worst)
        'Cov_returns': 'descending',  # Highest (best) to lowest (worst)
        'MSC': 'descending'  # Highest (best) to lowest (worst)
    }

    for param in measures:
        if ranking_criteria[param] == 'ascending':
            total_ranks = {pair: total_ranks[pair] + ranks[param][i] for i, pair in enumerate(pairs)}
        elif ranking_criteria[param] == 'descending':
            total_ranks = {pair: total_ranks[pair] + len(pairs) - ranks[param][i] + 1 for i, pair in enumerate(pairs)}

    ranked_pairs = sorted(total_ranks.items(), key=lambda x: x[1])

    final_ranking = {i + 1: pair for i, (pair, rank) in enumerate(ranked_pairs)}

    return final_ranking
