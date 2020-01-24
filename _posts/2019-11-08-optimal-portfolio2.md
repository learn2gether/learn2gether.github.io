---
title: "Optimal Portfolio Part II"
date: 2019-11-08
# author: Yifei Luo
tags: [finance analysis, data science, forecasting, stocks portfolio]
# header:
#     image: "images/posts/titanic.jpeg"
excerpt: "Portfolio optimization analysis applied by Monte Carlo simulation, VaR model and Portfolio Beta."
---

![alt text](https://learn2gether.github.io/images/posts/portfolio/optimal_portfolio.jpg "OPTIMAL PORTFOLIO")


- [Portfolio Optimization](#portfolio-optimization)
  * [Monte Carlo Simulation](#monte-carlo-simulation)
  * [Portfolio with the minimum volatility](#portfolio-with-the-minimum-volatility)
  * [Portfolio with the maximum return](#portfolio-with-the-maximum-return)
  * [Portfolio with the best Sharpe Ratio](#portfolio-with-the-best-sharpe-ratio)
  * [Comparison of three optimal portfolios](#comparison-of-three-optimal-portfolios)
- [Assessing Risk by VaR](#assessing-risk-by-var)
  * [The Variance-covariance method](#the-variance-covariance-method)
  * [The Historical Method](#the-historical-method)
- [Portfolio Beta](#portfolio-beta)


# Portfolio Optimization

<div style="text-align: justify"> In order to implement the portfolio optimization, the next question to consider is: What is the best combination of weights? Does the customer care more about the biggest gain or the least risk? We need to consider the trade-off between the risk and the return. We can apply the meanvariance analysis and efficient frontier to locate the optimal portfolio. </div>

## Monte Carlo Simulation

<div style="text-align: justify"> The Monte Carlo simulation is to generate a list of weights for stocks randomly with the expected return and the volatility. Then, we repeat this method hundred thousand times to generate hundred thousand portfolios. The expected return and the volatility of these portfolios can be used to draw a scatter plot of the mean-variance analysis. According to the following figure, we are able to visualize expected return and volatility easily. The horizontal axis represents the volatility reflecting the risk, and the vertical axis represents the expected return. Each point represents a portfolio. Markowitz's portfolio theory argued that a rational investor is always attempting to maximize the return at a given level of the risk or minimize the risk at a given level of the return. There is always a unique portfolio with minimum risk for each of target return. Thus, we can define a curve to represent a number of portfolios for different target return. This curve is the effective frontier shown by the red curve, only the point on the effective frontier is the most effective portfolio. </div>

<br />

```python
number = 100000
random_p = np.empty((number, 27))
np.random.seed(123)

for i in range(number):
    random9 = np.random.random(25)
    random_weight = random9 / np.sum(random9)
    
    mean_return = StockReturns.mul(random_weight, axis=1).sum(axis=1).mean()
    annual_return = (1 + mean_return)**252 - 1

    random_volatility = np.sqrt(np.dot(random_weight.T, 
                                       np.dot(cov_mat_annual, random_weight)))

    random_p[i][:25] = random_weight
    random_p[i][25] = annual_return
    random_p[i][26] = random_volatility
    
RandomPortfolios = pd.DataFrame(random_p)
RandomPortfolios.columns = [ticker + "_weight" for stock in StockReturns.columns.tolist()]  \
                         + ['Returns', 'Volatility']
```

![alt text](https://learn2gether.github.io/images/posts/portfolio/mean_variance_analysis.png "Mean Variance Analysis")

## Portfolio with the minimum volatility

<div style="text-align: justify"> One strategy is to select the portfolio with the minimum volatility. This portfolio is the global minimum volatility portfolio (GMV). The GMV portfolio is marked by the red point. </div>

<br />

```python
min_index = RandomPortfolios.Volatility.idxmin()

RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[min_index,'Volatility']
y = RandomPortfolios.loc[min_index,'Returns']
plt.scatter(x, y, color='red')   
plt.show()
```

![alt text](https://learn2gether.github.io/images/posts/portfolio/gmv_portfolio.png "GMV Portfolio")

<div style="text-align: justify"> Then, we can calculate the cumulative return of the GMV portfolio to compare with the equal weight portfolio and the given weight portfolio. The green line represents the GMV portfolio. Obviously, its return is relatively smaller than the other two portfolios, which reflects the reality. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/gmv_portfolio_comparision.png "GMV Portfolio Comparision")

## Portfolio with the maximum return

<div style="text-align: justify"> One strategy is to select the portfolio with the maximum return. This portfolio is the global maximum return portfolio (GMR). The GMR portfolio is marked by the red point. </div>

<br />

```python
max_index_return = RandomPortfolios.Returns.idxmax()

RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[max_index_return,'Volatility']
y = RandomPortfolios.loc[max_index_return,'Returns']
plt.scatter(x, y, color='red')   
plt.show()
```

![alt text](https://learn2gether.github.io/images/posts/portfolio/gmr_portfolio.png "GMR Portfolio")

<div style="text-align: justify"> Then, we can calculate the cumulative return of the GMR portfolio to compare with other portfolios. The oriange line represents the GMV portfolio. Obviously, its return is relatively higher than the other portfolios, but its risk is also higher, which also reflects the reality. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/gmr_portfolio_comparison.png "GMR Portfolio Comparison")

## Portfolio with the best Sharpe Ratio

<div style="text-align: justify"> In fact, if we want to find a balance between the return and the risk, the Sharpe ratio can help make a better choice as it calculates the excess return by increasing each unit of risk. </div>

<br />

<div style="text-align: justify"> First, we calculate the Sharpe ratio for each portfolio generating by the Monte Carlo simulation. Then, generating a scatter plot to present each portfolio, and the color represents the value of Sharpe ratio shown as the following figure. The higher the value, the better the portfolio. </div>

<br />

```python
risk_free = 0

RandomPortfolios['Sharpe'] = (RandomPortfolios.Returns - risk_free)   \
                            / RandomPortfolios.Volatility

plt.scatter(RandomPortfolios.Volatility, RandomPortfolios.Returns, 
            c=RandomPortfolios.Sharpe)
plt.colorbar(label='Sharpe Ratio')
plt.show()
```

![alt text](https://learn2gether.github.io/images/posts/portfolio/sharp_ratio.png "Sharp Ratio Plot")

<div style="text-align: justify"> In the following figure, the portfolio with the largest Sharpe ratio is marked by the red point. </div>

<br />

```python
max_index_sharp = RandomPortfolios.Sharpe.idxmax()

RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[max_index_sharp,'Volatility']
y = RandomPortfolios.loc[max_index_sharp,'Returns']
plt.scatter(x, y, color='red')   
plt.show()
```

![alt text](https://learn2gether.github.io/images/posts/portfolio/portfolio_best_sharpe_ratio.png "Best Sharp Ratio")

<div style="text-align: justify"> Then, we can calculate the cumulative return of the portfolio with the largest Sharpe ratio to compare with other portfolios. The green line represents the MSR portfolio. Its return is relatively higher than the GMV portfolio, but lower than the GMR portfolio. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/msr_portfolio_comparison.png "MSR Portfolio Comparison")

## Comparison of three optimal portfolios

<div style="text-align: justify"> The following table presents some statistical results of discussed portfolios above. The GMV portfolio provides the lowest volatility but it also yields the lowest. The GMR portfolio generates the maximum return but it also comes with the highest risk. The MST portfolio implements the risk-return trade-off. These three portfolios are all optimal, and the decision is made depends on the risk preference of the client. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/statistics_optimal_portfolios.png "Statistics Optimal Portfolios")

<div style="text-align: justify"> There are normally three common types of investors with different risk preference including the risk averse investors, the risk neutral investors and the risk lover investors. A risk averse investor is going to exhibit diminishing marginal utility of wealth. When all portfolios generate the same expected return, they prefer the low-risk portfolio. When all portfolios have the same risk, they prefer the high-return portfolio. A risk neutral investor is going to exhibit constant marginal utility of wealth. They usually do not evade risk or actively pursue risk, and they select the portfolio according to their expected return, regardless of the risk level. A risk seeker is going to exhibit increasing marginal utility of wealth. They usually take the initiative to pursue risk. When two portfolios have the same rate of return, they prefer the riskier portfolio since this usually brings more utility. </div>

<br />

# Assessing Risk by VaR

<div style="text-align: justify"> VaR can be used to assess the risk of the portfolio. The reason to apply VaR is that variance cannot give the direction of the investing movement as the volatility of the stock can be caused due to the rocket growth rate in its price. Risk is usually about losing money for the investor, and the VaR model is able to identify the worst case. The VaR consists of three components including a confidence level, a set of time period and amount of loss which can be expressed by amount of money or a certain percentage. There are two common methods to calculate VaR by the variance-covariance matrix or the historical data. We are going to use these two methods to assess the optimal portfolios discussed above. </div>

## The Variance-covariance method

<div style="text-align: justify"> This method is to assess the VaR of the portfolio using the correlation and volatility of stocks. Based on these two values, we can generate the normal distribution of the portfolio, and then the risk level is determined according to the confidence interval. </div>

<br />

<div style="text-align: justify"> The following figure presents the normal distribution of the optimal portfolio with the minimum volatility. If we use the confidence level of 95%, the rate of return is 0.987%. It means that there is a 95% chance that this portfolio will lose no more than 0.987% in a single day. If the investor invests 10 million to calculate the absolute amount, the 5% VaR for a single day is 0.00987 * 10 million = 98,700 dollars. </div>

<br />

```python
import scipy.stats

mean_GMV = stockReturns['Portfolio_GMV'].mean()
sigma_GMV = stockReturns['Portfolio_GMV'].std()
tdf, tmean, tsigma = scipy.stats.t.fit(stockReturns['Portfolio_GMV'])
support = np.linspace(stockReturns['Portfolio_GMV'].min(), stockReturns['Portfolio_GMV'].max(), 100)

plt.figure(figsize=(15, 5))
plt.hist(stockReturns['Portfolio_GMV'],bins=50, density=True)
plt.plot(support, scipy.stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-")

plt.title("Rate of Return (%)", weight='bold')
```

![alt text](https://learn2gether.github.io/images/posts/portfolio/var_gmv.png "VaR GMV")

<div style="text-align: justify"> The following figure presents the normal distribution of the optimal portfolio with the maximum return. If we use the confidence level of 95%, the rate of return is 1.43%. It means that there is a 95% chance that this portfolio will lose no more than 1.43% in a single day. If the investor invests 10 million to calculate the absolute amount, the 5% VaR for a single day is 0.0143 * 10 million = 143,000 dollars. </div>

<br />

```python
import scipy.stats

mean_GMR = stockReturns['Portfolio_GMR'].mean()
sigma_GMR = stockReturns['Portfolio_GMR'].std()
tdf, tmean, tsigma = scipy.stats.t.fit(stockReturns['Portfolio_GMR'])
support = np.linspace(stockReturns['Portfolio_GMR'].min(), stockReturns['Portfolio_GMR'].max(), 100)

plt.figure(figsize=(15, 5))
plt.hist(stockReturns['Portfolio_GMR'],bins=50, density=True)
plt.plot(support, scipy.stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-")

plt.title("Rate of Return (%)", weight='bold')
```

![alt text](https://learn2gether.github.io/images/posts/portfolio/var_gmr.png "VaR GMR")

<!-- <br /> -->

<div style="text-align: justify"> The following figure presents the normal distribution of the optimal portfolio with the maximum Sharpe Ratio. If we use the confidence level of 95%, the rate of return is 1.05%. It means that there is a 95% chance that this portfolio will lose no more than 1.05% in a single day. If the investor invests 10 million to calculate the absolute amount, the 5% VaR for a single day is 0.0105 * 10 million = 105,000 dollars. </div>

<br />

```python
import scipy.stats

mean_MSR = stockReturns['Portfolio_MSR'].mean()
sigma_MSR = stockReturns['Portfolio_MSR'].std()
tdf, tmean, tsigma = scipy.stats.t.fit(stockReturns['Portfolio_MSR'])
support = np.linspace(stockReturns['Portfolio_MSR'].min(), stockReturns['Portfolio_MSR'].max(), 100)

plt.figure(figsize=(15, 5))
plt.hist(stockReturns['Portfolio_MSR'],bins=50, density=True)
plt.plot(support, scipy.stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-")

plt.title("Rate of Return (%)", weight='bold')
```

![alt text](https://learn2gether.github.io/images/posts/portfolio/var_msr.png "VaR MSR")

<div style="text-align: justify"> By comparing the VaR of these three portfolios, the portfolio with the minimum volatility has the minimum loss, but it results in a smaller return. </div>

## The Historical Method

<div style="text-align: justify"> This method is to calculate the overall rate of return and the volatility for a certain period of time, and then determine the maximum loss according to the confidence interval such as 10%, 5% and 1%. </div>

<br />

<div style="text-align: justify"> The following figure shows the VaR on percentage and dollars respectively for the GMV portfolio. As for the confidence interval of 95%, there is a 95% chance that this portfolio will lose no more than 0.993% in a single day. If the investor invests 10 million to calculate the absolute amount, the 5% VaR for a single day is 99,300 dollars. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/var_gmv_loss.png "VaR GMV Loss")

<div style="text-align: justify"> According to the following figure, there is a 95% chance that this portfolio will lose no more than 0.993% in a single day, which may happen 38 days over the three years. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/var_gmv_loss_plot.png "VaR GMV Loss Plot")

<div style="text-align: justify"> The following figure shows the VaR on percentage and dollars respectively the GMR portfolio. As for the confidence interval of 95%, there is a 95% chance that this portfolio will lose no more than 1.52% in a single day. If the investor invests 10 million to calculate the absolute amount, the 5% VaR for a single day is 152,000 dollars. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/var_gmr_loss.png "VaR GMR Loss")

<div style="text-align: justify"> According to the following figure, there is a 95% chance that this portfolio will lose no more than 1.52% in a single day, which may happen 38 days over the three years. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/var_gmr_loss_plot.png "VaR GMR Loss Plot")

<div style="text-align: justify"> The following figure shows the VaR on percentage and dollars respectively the MSR portfolio. As for the confidence interval of 95%, there is a 95% chance that this portfolio will lose no more than 1.09% in a single day. If the investor invests 10 million to calculate the absolute amount, the 5% VaR for a single day is 109,000 dollars. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/var_msr_loss.png "VaR MSR Loss")

<div style="text-align: justify"> According to the following figure, there is a 95% chance that this portfolio will lose no more than 1.09% in a single day, which may happen 38 days over the three years. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/var_msr_loss_plot.png "VaR MSR Loss Plot")

<div style="text-align: justify"> By comparing the VaR of these three portfolios, this method generates almost the identical result, and the portfolio with the minimum volatility still has the minimum loss. </div>

<br />

# Portfolio Beta

<div style="text-align: justify"> Beta is another measure to evaluate the risk of the portfolio. The S&P 500 index is normally considered as a benchmark portfolio, and it can be also defined as a market portfolio. The systematic risk is exhibited by the market portfolio, and Beta is used to assess this kind of risk. In other words, it assesses the volatility of any stocks against the volatility of the overall market. For example, we assume that the stock of Google has a beta value of 1.5. If the rate of return for the market is 10%, then the return of Google stock will be 15%. In other words, the return of Google stock exceeds 50% of the market. Now, if the return of the market falls by 5%, then Google stock will fall by 7.5%, which is also 50% more than the market. Therefore, Google is a high beta stock. Similarly, if any stock has a beta of 0.75, then its volatility will be lower than the market. If the return of the market is 10%, then low beta stocks will only receive 7.5%. However, stocks with lower value of the beta can help reduce the market risk because if the return of the market falls by 5%, the stock will only fall by 3.75%, which is important when the market has a downtrend. </div>

<br />

<div style="text-align: justify"> According to the following figure, we can see that the portfolio of the minimum risk has a small volatility compared to the market portfolio, and it has a beta value of 0.657. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/gmv_volatility.png "GMV Volatility")

<div style="text-align: justify"> According to the following figure, we can see that the portfolio of the maximum return has a relatively large volatility compared to the market portfolio, and it has a beta value of 1.011. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/gmr_volatility.png "GMR Volatility")

<div style="text-align: justify"> According to the following figure, we can see that the portfolio of the largest Sharpe Ratio has a small volatility compared to the market portfolio, and it has a beta value of 0.729. </div>

<br />

![alt text](https://learn2gether.github.io/images/posts/portfolio/msr_volatility.png "MSR Volatility")

<br />

<div style="text-align: justify"> By comparing these three portfolios, the GMV portfolio is a better choice since it has the smallest beta, which is able to reduce the market risk. However, it may not reflect that its return meets the client’s expectation.</div>




