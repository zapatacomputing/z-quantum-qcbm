import datetime
import bs4 as bs
import pandas as pd
import requests
import yfinance as yf
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage, risk_matrix
from scipy.special import comb
from math import comb 
from itertools import combinations
import cvxpy as cp
import pandas as pd
from scipy import spatial
from zquantum.core.bitstring_distribution import BitstringDistribution


def download_sp500(start_date, end_date):
    
    '''Gets S&P 500 stock asset data from wikipedia 
    Args: 
        start_date (string): string in the following representation "2017-12-01" for "year-month-day" 
        end_date (string): string in the following representation "2018-12-01" for "year-month-day" 
    Returns: 
        Data: pandas.core.frame.DataFrame object with the S&P 500 stock data from the start-date to the end-date '''

    try:
        data = pd.read_csv(f"data/sp500_{start_date}_{end_date}.csv", index_col="Date")
        print("Reading data from file")
    except:
        resp = requests.get("http://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        soup = bs.BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", {"class": "wikitable sortable"})
        tickers = []

        for row in table.findAll("tr")[1:]:
            ticker = row.findAll("td")[0].text
            tickers.append(ticker)

        tickers = [s.replace("\n", "") for s in tickers]
        data = yf.download(tickers, start=start_date, end=end_date)

        data = data["Adj Close"]

        try:
             data.to_csv(f"sp500_{start_date}_{end_date}.csv")
        except:
            print("didn't save to a .csv file :( ")

    return data

def generate_initial_dataset(nbits: int, N: int, M: int):

    '''Take nbits random S&P 500 assets and generate N random portfolios with the ability to choose M assets
        in each portfolio. 
    Args: 
        nbits (int): number of assets sampled from the S&P 500 that could go in the portfolio 
        N (int): number of possible portfolios with different asset combinations 
        M (int): number of assets one can have in a portfolio 
    Returns: 
        unique_dataset(np.array): nbits x N size np.array of all of the portfolio options 
        string_list(np.array of strings): array of strings of nbit binary digits up to 2**nbits 
        unique_string(np.array of strings): array of strings of each array in unique_dataset '''
    string_list = np.array([])
    dataset = np.zeros((2**nbits, nbits), dtype=int)
    unique_string =  np.array([])
    for i in range(2 ** nbits): 
        s = format(i, '0' + str(nbits) + 'b' ) 
        vector_string = np.asarray(s) 
        string_list = np.append(string_list, vector_string)
        flipped_bits = np.random.choice(nbits, M, replace=False)
        dataset[i, flipped_bits] = 1
        
    unique_dataset = np.unique(dataset, axis=0) 
    unique_dataset = unique_dataset[::-1]
    for i in range(len(unique_dataset)): 
        string = ''
        for j in range(nbits): 
            string += str(unique_dataset[i][j])
        unique_string = np.append(unique_string, string)
    return unique_dataset, string_list, unique_string

def objective_function(x, covs, parameters):

    '''Returns the risk for a given portfolio, covariance matrix, and parameters that include the fixed return 
        and weight boundaries. 
    Args: 
        x (np.array): array of the assets in the given portfoloio 
        covs (np.array): covariance matrix for the given portfolio 
        parameters(dictionary): dictionary containing the target return, lower and uppoer bound, and temperature 
    Returns: 
        risk (int): returns the risk of the given portfolio '''
  
    w = cp.Variable(len(x))
    objective = None
    constraints = []
    q = np.zeros_like(x)

    objective = cp.Minimize(0.5 * cp.quad_form(w, covs) - q.T @ w)

    constraints = [
        sum(w) == 1,
        w >= parameters["lower_bound"],
        w <= parameters["upper_bound"],
        x @ w == parameters["target_return"],
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    weights = w.value
    if weights is None: 
        weights = np.ones(len(x)) / len(x)
    
    variance = np.dot(weights, np.dot(covs, weights))
    return np.sqrt(variance)

def cost(bitstr, returns, covs, parameters):
    
    '''Returns the cost for a given portfolio, covariance matrix, and parameters that include the fixed return 
        and weight boundaries. 
    Args: 
        bitstr (np.array): array of the chosen assets in the given portfoloio 
        returns (np.array): array of all of the assets sampled from the S&P 500 
        covs (np.array): covariance matrix for the given portfolio 
        parameters(dictionary): dictionary containing the target return, lower and uppoer bound, and temperature 
    Returns: 
        cost (int): returns the cost of the given portfolio '''

    where = np.where(bitstr)[0]
    try:
        cost = objective_function(
            returns[where],
            covs[where, :][:, where],
            parameters,
        )
    except:
        cost = 1.0
    return cost

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def probability(cost, T):

    '''Returns the probability for a given portfolio given the cost and the temperature T. 
    Args: 
        cost (int): the value of the cost function for our optimization 
        T (int): temperature for the distribution 
    Returns: 
        cost (int): returns the cost of the given portfolio '''

    return np.exp(-cost / T)

def get_probabilities(start_date, end_date, total_assets, target_return, num_assets_to_choose): 

    '''Returns a BitstringDistribution object of the portfolio probability distribution.  
    Args: 
        start_date (string): string in the following representation "2017-12-01" for "year-month-day" 
        end_date (string): string in the following representation "2018-12-01" for "year-month-day" 
        total_assets (int): number of assets sampled from the S&P 500 that could go in the portfolio 
        target_return (float): target return in decimal form (2.5% = 0.025)
        num_assets_to_choose (int): number of assets one can have in a portfolio 
    Returns: 
        get_probabilities (BitstringObject): returns the portfolio probability distribution'''

    data = download_sp500(start_date, end_date)
    mu = mean_historical_return(data)
    S = CovarianceShrinkage(data).ledoit_wolf()
    assets = np.random.choice(len(mu), size=total_assets, replace=False)

    new_mu = mu.values[assets]
    new_S = S.values[assets][:, assets]
    parameters = {
        "target_return": target_return,
        "lower_bound": 0.05,
        "upper_bound": 0.9,
        "T": 1} 
    portfolio_num = nCr(total_assets, num_assets_to_choose)
    possibilities = generate_initial_dataset(total_assets, portfolio_num, num_assets_to_choose)
    string_possibilities = possibilities[1]
    portfolio_string_possibilities = possibilities[2]
    portfolio_possibilities = possibilities[0]
    prob_sum = 0 
    prob_list = [] 
    reverse = []
    count = 0 
    for i in string_possibilities: 
        if i not in portfolio_string_possibilities: 
            prob_list.append(int(0))
        else:  
            risk = cost(portfolio_possibilities[count],new_mu, new_S, parameters)
            where = np.where(portfolio_possibilities[count])[0]
            prob = probability(risk, 1)
            prob_list.append(prob)
            prob_sum += prob
            count += 1 
        reverse_string = i[len(i)::-1]
        reverse.append(reverse_string)
    prob_list = [i / prob_sum for i in prob_list]
    assert(round(sum(prob_list), 8) == 1)
    print(reverse)
    prob_dist = {reverse[i]: prob_list[i] for i in range(len(prob_list))} 
    return BitstringDistribution(prob_dist) 

