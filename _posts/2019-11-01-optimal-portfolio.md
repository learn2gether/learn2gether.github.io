---
title: "Optimal Portfolio Part I"
date: 2019-11-01
# author: Yifei Luo
tags: [finance analysis, data science, forecasting, stocks portfolio]
# header:
#     image: "images/posts/titanic.jpeg"
excerpt: "A case study on investment by selecting stocks from S&P 500 for optimal portfolio."
---

![alt text](https://learn2gether.github.io/images/posts/portfolio/optimal_portfolio.jpg "OPTIMAL PORTFOLIO")

<br />

# Introduction

<div style="text-align: justify"> Every investor should know the acronym “TINSTAAFL”, which represents “There is no such thing as a free lunch”. This phrase emphasizes the concept of the trade-off between the risk and the return. The relatively high return also brings the high risk, and a lower risk is also likely to result in the lower return. Markowitz published a research about Portfolio Selection in 1952, and it introduced a Modern Portfolio Theory (MPT). This theory demonstrated several main points (Markowitz 1952). First, the return of a portfolio is a weighted average of the components of each asset, and its volatility is also affected by the correlation among assets. In addition, investors should not assess asset benefits independently, and they should understand how each asset is going to affect the effectiveness of the portfolio. Furthermore, investors should allocate their capital and resources into multiple assets rather than concentrating on one or a small number of assets, which is able to greatly reduce the volatility of the portfolio. </div>

<br />

<div style="text-align: justify"> The MPT attempts to explain the trade-off relationship between obtaining the maximum return and avoiding the excessive risk. Investors combine distinct assets in a certain proportion to achieve the expected return. Under the premise, the risk is minimized or the yield is maximized. The MPT assumes that investors prefer risk-neutral, which means that if there are two portfolios with the same return, investors are more likely to select the portfolio with lower risk. Thus, if there exists a portfolio with high return, investors are more likely to choose a high-risk portfolio. In order to reduce the volatility, the MPT introduced the diversity. The benefit of the diversification is that investors can optimize the asset allocation to obtain the maximum return under their risk preference. </div>

<br />

<div style="text-align: justify"> Markowitz proposed a mean-variance analysis to find an optimal asset allocation, which is able to balance the expected return and the variances in earnings (Markowitz 1952). There is a key concept associated with mean-variance analysis, which is the Efficient Frontier, and it is a list of optimal portfolios which generate the maximum return for each distinct risk level or the lowest variance for the distinct rate of return. </div>

<br />

<div style="text-align: justify"> The successor of Markowitz is dedicated to simplifying the portfolio model. Under a series of assumptions, scholars such as William Sharpe derived the Capital Asset Pricing Model (CAPM) (Sharpe 1977). Even though the CAPM simplified the MPT, it does not reduce the effectiveness. Also, some scholars believe that the MPT has certain defects to define the risk, and they proposed some new portfolio optimization models such as the VaR model. </div>

<br />

# Stocks Selecting

<div style="text-align: justify"> In order to achieve portfolio optimization, we need to determine the number of stocks in the portfolio and what these stocks are. The client plan to invest $10 million dollars in companies listing on S&P 500. In order to select valuable stocks, I am going to build a portfolio including all companies on the list of S&P 500, and the list is extracted from the WIKIPEDIA “List of S&P 500 Companies”. Then, extracting and analyzing daily adjusted closing price for each stock in the past three years. All these data are extracting from the Yahoo Finance. The following figure shows the daily closing price from 2016 to 2019. There are 8 stocks containing missing values, which means that these companies went public after year 2016. Thus, we need to remove them from the list. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/stocks.png "Stocks")

<br />

<div style="text-align: justify"> PyPortfolioOpt is a Python library, which can help find the portfolio that maximize the Sharpe ratio. The Sharpe ratio is a measure of risk adjusted returns. The following figure presents a set of weights of the optimal portfolio. We can notice that many stocks are assigned the weight of zero representing that they are not recommended stocks. </div>

<br />

```python
# expected returns for all 500 stocks
mu=expected_returns.mean_historical_return(stockPrices)
# covariance for all stocks
S=risk_models.sample_cov(stockPrices)
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
```

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/weights_all_stocks.png "Weights of All Stocks")

<br />

<div style="text-align: justify"> There are only 25 stocks left with non-zero weights. Therefore, we are going to implement portfolio optimization based on these selected stocks. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/selected_stocks.png "Selected Stocks")

<br />

<div style="text-align: justify"> All selected companies and their stock codes are displayed on the following table. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/stocks_code.png "Stocks Code")

<br />

<div style="text-align: justify"> Then, we can calculate the daily rate of return by closing prices of each stock. </div>

<br />

```python
StockReturns = stockPrices_selected.pct_change().dropna()
```

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/rate_of_return.png "Rate Of Return")

<br />

# Portfolio Return

<div style="text-align: justify"> According to the analysis above, we have chosen 25 stocks, but how can we allocate the capital into these stocks? This requires us setting the corresponding weights for each asset. There are three common weight allocation methods to calculate the return under different combinations. </div>

## Portfolio with the given weights

<div style="text-align: justify"> In this method, we assign weights to each stock by the previous process of selecting stocks. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/assigned_weights.png "Assigned Weights")

<br />

<div style="text-align: justify"> We multiply the rate of return of each stock by its corresponding weights to obtain the weighted stock returns, then summing the weighted returns of all stocks to get the return of the portfolio. The following figure shows the daily rate of return over-time. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/rate_of_return_assigned.png "Rate Of Return Assigned Weights")

<br />

<div style="text-align: justify"> Then, we can calculate the cumulative return to present the yield curve shown as the following figure. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/cumulative_return_assigned.png "Cumulative Return Assigned Weights")

<br />

## Portfolio With The Equal Weight

<div style="text-align: justify"> The second option is to evenly distribute the weight of each stock so that they all have an equal weight. This is the easiest way to invest and can be used as a benchmark for other portfolios. The following figure shows the comparison of the cumulated return. The blue line represents the portfolio of the equal weight, and the orange line represents the portfolio of the given weight. We can conclude the benchmark outperforms the portfolio of the given weight. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/return_comparision1.png "Cumulative Return Assigned Vs. Equal")

<br />

## Portfolio Based On The Market Capitalization

<div style="text-align: justify"> This method considers the market value of the company and allocates weights based on the proportion of market capitalization. Therefore, companies with high market capitalization have more weights. When the stocks of these big firms have a better performance, the portfolio also performs better. As we all known, the S&P 500 index is weighted by the market value of each company. The following table presents the market capitalization of each selected companies in 2019. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/market_capitalization.png "Market Capitalization")

<br />

<div style="text-align: justify"> The following figure presents the cumulative return of three portfolios. The green line represents the
portfolio of the cumulative return. The benchmark seems to outperform the other two portfolios. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/return_comparision2.png "Return Comparision All Three")

<br />

# Correlation and covariance

## Correlation

<div style="text-align: justify"> The correlation matrix is used to estimate the linear relationship among multiple stocks. Each cell in the matrix is the correlation coefficient of its corresponding two stocks ranging from -1 to 1. If they are positive correlated, they tend to move together, which means they either increase together or decrease together. If they are negative correlated, they tend to move oppositely. The following figure presents the heatmap of the correlation matrix, which can easily visualize relationships. We can see that most stocks have positive correlations with other stocks. It may reflect that companies with the positive correlation may be in the same industry, or they may have mutual business. </div>

<br />

```python
# correlation
correlation_matrix = StockReturns.corr()

# seaborn
import seaborn as sns
plt.figure(figsize=(20,12))
# Heatmap
corr_map = sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu", 
            linewidths=0.3,
            annot_kws={"size": 8})

bottom, top=corr_map.get_ylim()
corr_map.set_ylim(bottom+0.5, top-0.5)

plt.xticks(rotation=90)
plt.yticks(rotation=0) 
plt.show()
```

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/return_heatmap.png "Return Heatmap")

<br />

## Covariance

<div style="text-align: justify"> The correlation coefficient only reflects the linear relationship between stocks, but it does not tell us the volatility of stocks, and the covariance matrix contains this information. Covariance is a measure to assess the risk of an asset relative to another asset. It is the correlation of yield between the two assets. If it is positive, it indicates that the return of one asset rises, the other is also rising. If the value of the covariance is negative, it indicates that they move in the opposite direction. In addition, if the absolute value of the covariance is bigger, the returns of the two assets have a strong relationship. On the contrary, if the absolute value is relatively small, it means that the returns of these two assets have a distant relationship. </div>

## Standard deviation

<div style="text-align: justify"> Standard deviation is another measure of the risk, and it is also known as volatility. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/statistics_three_portfolios.png "Statistics Of Three Portfolios")

<br />

<div style="text-align: justify"> This table presents some statistical results of discussed portfolios above. The portfolio with the equal weights generates the maximum return, and the portfolio with the given weights generated the minimum risk. The portfolio based on the market capitalization is not recommended in our case. </div>





