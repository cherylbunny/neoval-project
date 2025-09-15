# About the Project 

Understanding how macroeconomic forces such as mining investment, immigration and lifestyle shifts shape housing prices in Australia is critical for policymakers, investors, residents and real estate professionals. Traditional analyses often overlook both the broader macroeconomic drivers and the fine-grained dynamics at regional levels such as SA4.

This project addresses that gap by applying Principal Component Analysis to uncover key patterns in housing price movements. By examining loading scores we identify the macroeconomic factors behind each component and construct proxies through linear combinations of PC time series. We then develop forecasting models to project housing indexes up to a decade ahead and employ multivariate regressions with time varying coefficients to capture evolving regional dynamics.

The outcome is a factor model that provides a comprehensive framework to explain the drivers of Australia’s housing market at both national and regional scales.

# Built With 

Built with R version 4.4.3. Core libraries include `tidyverse` for data wrangling and visualisation, `fpp3` and `forecast` for time series modelling, `tseries` and `urca` for statistical testing, and `sf` for spatial analysis.

# Getting Started 

This is a research based project and does not include a user interface. 

To replicate the analysis, clone the repository and run the R scripts in sequence: 

```r
git clone https://github.com/cherylbunny/neoval-project.git
```

To set up the R environment, ensure you have R version 4.4.3 (or higher).
This project uses `renv` to manage dependencies. To recreate the exact environment, run:
```r
install.packages("renv")
renv::restore()
```
To access the datasets: The input datasets include city-level and regional housing price indexes, PCA results, and regression outputs.
Data files are located in the `/data` folder.

Run the analysis: Open the R scripts which is the `.qmd` file. 
Scripts are organised by analysis stage, including proxy reconstruction, explortary data analysis, forecasting models, and factor regressions.

To view the outputs: Model outputs, plots, and summary tables are saved in the `/output` folder.



# Methodologies 

# Further Enhauncement 

Future work may involve developing additional proxies from the full set of 20 principal components, applying advanced machine learning methods for forecasting, incorporating a broader range of macroeconomic variables, and extending the analysis to finer regional levels to enable richer comparative insights.

# License 

This project is proprietary. Copyright © 2025 Neoval. All rights reserved. See the LICENSE file for details. 

# Contributor 

The foundational work for this project was developed by Will Silp, under whose supervision Yiran Yao made key contributions.

