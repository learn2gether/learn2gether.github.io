---
title: "Optimal Portfolio"
date: 2019-11-01
tags: [finance analysis, data science, forecasting, stocks portfolio]
# header:
#     image: "images/posts/titanic.jpeg"
excerpt: "A case study on investment by selecting stocks from S&P 500 for optimal portfolio."
---

![alt text](https://learn2gether.github.io/images/posts/portfolio/optimal_portfolio.jpg "OPTIMAL PORTFOLIO")

<br />

# Introduction

<div style="text-align: justify"> Every investor should know the acronym “TINSTAAFL”, which represents “There is no such thing as a free lunch”. This phrase emphasizes the concept of the trade-off between the risk and the return. The relatively high return also brings the high risk, and a lower risk is also likely to result in the lower return. Markowitz published a research about Portfolio Selection in 1952, and it introduced a Modern Portfolio Theory (MPT). This theory demonstrated several main points (Markowitz 1952). First, the return of a portfolio is a weighted average of the components of each asset, and its volatility is also affected by the correlation among assets. In addition, investors should not assess asset benefits independently, and they should understand how each asset is going to affect the effectiveness of the portfolio. Furthermore, investors should allocate their capital and resources into multiple assets rather than concentrating on one or a small number of assets, which is able to greatly reduce the volatility of the portfolio. </div>

<div style="text-align: justify"> The MPT attempts to explain the trade-off relationship between obtaining the maximum return and avoiding the excessive risk. Investors combine distinct assets in a certain proportion to achieve the expected return. Under the premise, the risk is minimized or the yield is maximized. The MPT assumes that investors prefer risk-neutral, which means that if there are two portfolios with the same return, investors are more likely to select the portfolio with lower risk. Thus, if there exists a portfolio with high return, investors are more likely to choose a high-risk portfolio. In order to reduce the volatility, the MPT introduced the diversity. The benefit of the diversification is that investors can optimize the asset allocation to obtain the maximum return under their risk preference. </div>