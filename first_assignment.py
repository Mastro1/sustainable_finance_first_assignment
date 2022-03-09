import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
from statistics import mean
import pypfopt as opt
from pandas_datareader import data as wb

warnings.simplefilter(action='ignore', category=FutureWarning)

DIR = os.getcwd()
DATA = os.path.join(DIR, "Data_Excel", "MSCI_ESGscores")
FF = os.path.join(DATA, "FF")
FUNDAMENTALS = os.path.join(DATA, "Fundamentals")
RETURNS = os.path.join(DATA, "Returns")
SCORES = os.path.join(DATA, "Scores")
firm_names = os.path.join(DATA, "firm_names.xlsx")

if __name__ == '__main__':
    names = pd.read_excel(firm_names)
    emerging_countries = ['AE', "AR", "BR", "CL", "CN", "CO", "CZ", "EG", "GR", "HU", "ID", "IN", "KR", "KV", "MX",
                          "MY", "PE", "PH", "PK", "PL", "QA", "RU", "SA", "TH", "TR", "TW", "ZA"]
    emerging_firms = list(names[names['Country'].isin(emerging_countries)]["ISIN"])
    env_scores = pd.read_excel(os.path.join(SCORES, "Env.xlsx"))
    env_scores.dropna(how='all', axis=1, inplace=True)
    env_scores_firms = env_scores.columns
    filtered = [emerging_firm for emerging_firm in emerging_firms if emerging_firm in env_scores_firms]

    returns = pd.read_excel(os.path.join(RETURNS, "monthlyreturns.xlsx"))
    returns = returns.rename(columns={"Unnamed: 0": "timestamp"})
    returns = returns[["timestamp"] + filtered]

    annualized_return = pd.DataFrame(returns[filtered].mean() * 12, columns=["annualized_return"]).reset_index()
    annualized_return = annualized_return.rename(columns={"index": "firm"})

    annualized_volatility = pd.DataFrame(returns[filtered].apply(np.nanstd) * math.sqrt(12),
                                         columns=["annualized_volatility"]).reset_index()
    annualized_volatility = annualized_volatility.rename(columns={"index": "firm"})

    solution = pd.merge(annualized_return, annualized_volatility)

    correlation = solution['annualized_return'].corr(solution["annualized_volatility"])
    print("correlation: ", correlation)

    # ex 2:
    print("\nEqually weighted portfolio:")
    monthly_portfolio_returns_eq = returns[filtered].mean(axis=1)
    annualized_return_eq = monthly_portfolio_returns_eq.mean() * 12
    min_return = min(monthly_portfolio_returns_eq)
    max_return = max(monthly_portfolio_returns_eq)
    annualized_volatility_eq = np.std(monthly_portfolio_returns_eq) * math.sqrt(12)
    rf_mean = pd.read_excel(os.path.join(FF, "devrf.xlsx"), header=None)[1].mean() * 12
    sharp_ratio = (annualized_return_eq - rf_mean) / annualized_volatility_eq
    print("annualized return: ", annualized_return_eq, ", annualized_volatility:", annualized_volatility_eq)
    print("min return: ", min_return, ", max return: ", max_return, ", sharp_ratio: ", sharp_ratio)

    # value-weighted
    print("\nMarket value weighted portfolio:")
    size = pd.read_excel(os.path.join(FUNDAMENTALS, "size.xlsx"))[filtered]
    avg_returns = []
    for i in range(len(size)):
        weights = list(size.iloc[i].fillna(0))
        returns_ = returns.drop("timestamp", 1)
        returns_ = list(returns_.iloc[i].fillna(0))
        weighted_avg = np.average(returns_, weights=weights)
        avg_returns.append(weighted_avg)

    # fare una funzione per le statistiche e i plot
    annualized_return_mc = mean(avg_returns) * 12
    annualized_volatility_mc = np.std(avg_returns) * math.sqrt(12)
    min_return = min(avg_returns)
    max_return = max(avg_returns)
    sharp_ratio = (annualized_return_mc - rf_mean) / annualized_volatility_mc
    print("annualized return: ", annualized_return_mc, ", annualized_volatility:", annualized_volatility_mc)
    print("min return: ", min_return, ", max return: ", max_return, ", sharp_ratio: ", sharp_ratio)

    # ex 3:
    best_asset = solution.iloc[solution["annualized_return"].idxmax()]
    print("best asset: ")
    print(best_asset)
    best_asset_monthly_returns = returns[["timestamp", best_asset['firm']]].dropna()
    plt.plot(list(best_asset_monthly_returns["timestamp"]), list(best_asset_monthly_returns["MXP904131325"]))
    plt.xlabel("date")
    plt.ylabel("return")
    plt.title(str(best_asset["firm"] + "returns"))
    plt.show()

    plt.plot(list(returns["timestamp"]), list(monthly_portfolio_returns_eq))
    plt.xlabel("date")
    plt.ylabel("return")
    plt.title("equally weighted portfolio")
    plt.show()

    # TODO: What explains the differences between a one-asset portfolio and a portfolio composed of many stocks?

    first_two_years = best_asset_monthly_returns.loc[142:142 + 24]
    avg_revenue = first_two_years[best_asset["firm"]].mean()  # c'è qualcosa che non va
    start = 1
    for i in list(first_two_years[best_asset['firm']]):
        start = start * (1 + (i / 100))
    print(start)

    # ex 4:
    print("\n min variance portfolio")
    returns_min_var = []
    fifty_random = returns.sample(50, axis=1, random_state=42)
    for i in range(len(fifty_random)-1):
        running_data = fifty_random.iloc[:2+i].copy()
        running_data.dropna(how='all', axis=1, inplace=True)
        running_data = running_data.fillna(0)
        cov = running_data.cov()
        ef = opt.EfficientFrontier(running_data.mean(), cov)
        weights = ef.min_volatility()
        returns_min_var.append(np.average(running_data.mean(), weights=list(weights.values())))

    # funzione che fose dario farà un giorno
    annualized_return_mc = mean(returns_min_var) * 12
    annualized_volatility_mc = np.std(returns_min_var) * math.sqrt(12)
    min_return = min(returns_min_var)
    max_return = max(returns_min_var)
    sharp_ratio = (annualized_return_mc - rf_mean) / annualized_volatility_mc
    print("annualized return: ", annualized_return_mc, ", annualized_volatility:", annualized_volatility_mc)
    print("min return: ", min_return, ", max return: ", max_return, ", sharp_ratio: ", sharp_ratio)





    # test = returns[filtered].iloc[:2].fillna(0)
    # #test.cov() * 12
    # portfolio_returns = []
    # portfolio_volatilities = []
    #
    # for x in range(100):
    #     weights = np.random.random(test.shape[1])
    #     weights /= np.sum(weights)
    #
    #     portfolio_returns.append(np.sum(weights * test.mean()) * 12)
    #     portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(test.cov() * 12, weights))))
    #
    # portfolio_returns = np.array(portfolio_returns)
    # portfolio_volatilities = np.array(portfolio_volatilities)
    #
    # print(portfolio_returns, portfolio_volatilities)
    # #np.sum(weights * test.mean()) * 12
    # #np.sqrt(np.dot(weights.T, np.dot(test.cov() * 12, weights)))
    # portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatilities})
    # portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(15, 10));
    # plt.xlabel('Expected Volatility')
    # plt.ylabel('Expected Return')
