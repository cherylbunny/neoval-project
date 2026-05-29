# Data Sources

This file documents all external macro and housing data downloaded and cleaned for the housing market analysis project.

Each entry covers what the variable measures, where it comes from, the series ID used in `raw_data_cleaning.py`, and the date range of the downloaded data.

## Variables cleaned

- `mortgage_rate`: standard variable mortgage rate (RBA F5)
- `cash_rate`: overnight cash rate target (RBA A2)
- `bond_yield_10y`: 10-year government bond yield (RBA F2)
- `housing_lending_rate`: average rate on outstanding owner-occupier loans (RBA F6)
- `housing_credit_growth`: monthly growth in housing credit outstanding (RBA D2)
- `unemployment_rate`: unemployment rate, seasonally adjusted (ABS 6202.0)
- `trimmed_mean_inflation`: year-ended trimmed mean inflation (RBA G1)
- `household_disposable_income`: gross household disposable income, seasonally adjusted (ABS 5206.0)
- `net_overseas_migration`: net overseas migration, Australia (ABS 3101.0)
- `building_approvals`: total dwelling approvals, seasonally adjusted (ABS 8731.0)
- `dwelling_commencements`: total dwelling commencements, seasonally adjusted (ABS 8752.0)
- `cpi_rents`: CPI rents index, seasonally adjusted (ABS 6401.0)

---

## mortgage_rate

**Description**: Standard variable mortgage rate advertised by banks for owner-occupier home loans.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: F5, Indicator Lending Rates  
**Series ID**: FILRHLBVS  
**Units**: Percent per annum (% p.a.)  
**Frequency**: Monthly (no resampling needed)  
**Start**: 1959-02-01  
**End**: 2026-04-01  

---

## cash_rate

**Description**: The RBA's overnight cash rate target, Australia's primary monetary policy instrument.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: A2, Monetary Policy Changes  
**Series ID**: ARBAMPCNCRT  
**Units**: Percent per annum (% p.a.)  
**Frequency**: Event-based, resampled to monthly by forward filling  
**Start**: 1990-08-01  
**End**: 2026-05-01  

---

## bond_yield_10y

**Description**: Yield on 10-year Australian Government bonds, a market-determined long-term interest rate.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: F2, Capital Market Yields  
**Series ID**: FCMYGBAG10D  
**Units**: Percent per annum (% p.a.)  
**Frequency**: Daily, averaged to monthly  
**Start**: 2013-05-01  
**End**: 2026-05-01  

---

## housing_lending_rate

**Description**: Average interest rate actually paid on outstanding owner-occupier variable-rate home loans.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: F6, Housing Lending Rates  
**Series ID**: FLRHOOVL  
**Units**: Percent per annum (% p.a.)  
**Frequency**: Monthly (no resampling needed)  
**Start**: 2019-08-01  
**End**: 2026-03-01  

---

## housing_credit_growth

**Description**: Monthly percentage change in total housing credit outstanding, covering owner-occupier and investor loans.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: D2, Lending and Credit Aggregates  
**Series ID**: DLCACOHS  
**Units**: Percent change per month (%)  
**Frequency**: Monthly (no resampling needed)  
**Start**: 1990-02-01  
**End**: 2026-03-01  

---

## unemployment_rate

**Description**: Share of the labour force unemployed and actively seeking work, seasonally adjusted, all persons.

**Source**: ABS Labour Force Survey (6202.0)  
**Table**: Labour Force Table 12, Data2 sheet  
**Series ID**: A84423050A  
**Units**: Percent (%)  
**Frequency**: Monthly (no resampling needed)  
**Start**: 1978-02-01  
**End**: 2026-03-01  

---

## trimmed_mean_inflation

**Description**: The RBA's preferred underlying inflation measure, removing the top and bottom 15% of price changes each quarter.

**Source**: Reserve Bank of Australia (RBA), compiled from ABS CPI  
**Table**: G1, Consumer Price Inflation  
**Series ID**: GCPIOCPMTMYP  
**Units**: Percent, year-ended change (%)  
**Frequency**: Quarterly, resampled to monthly by forward filling  
**Start**: 1983-03-01  
**End**: 2026-03-01  

---

## household_disposable_income

**Description**: Total gross income available to Australian households after taxes and transfers, seasonally adjusted.

**Source**: ABS Australian National Accounts (5206.0)  
**Table**: Household Income Account, Data1 sheet  
**Series ID**: A2302939L  
**Units**: Millions of Australian dollars ($ millions, nominal)  
**Frequency**: Quarterly, resampled to monthly by forward filling  
**Start**: 1959-09-01  
**End**: 2025-12-01  

---

## net_overseas_migration

**Description**: Net flow of people arriving in Australia from overseas on a permanent or long-term basis.

**Source**: ABS Demographic Statistics (3101.0)  
**Table**: National, State and Territory Population, Data1 sheet  
**Series ID**: A2133254C  
**Units**: Thousands of persons per quarter (000)  
**Frequency**: Quarterly, resampled to monthly by forward filling  
**Start**: 1981-06-01  
**End**: 2025-09-01  

---

## building_approvals

**Description**: Total number of new dwellings approved for construction across all sectors, seasonally adjusted.

**Source**: ABS Building Approvals (8731.0)  
**Table**: Building Approvals, Data1 sheet  
**Series ID**: A422070J  
**Units**: Number of dwellings per month  
**Frequency**: Monthly (no resampling needed)  
**Start**: 1983-07-01  
**End**: 2026-03-01  

---

## dwelling_commencements

**Description**: Total number of new dwellings on which construction began each quarter, across all sectors, seasonally adjusted.

**Source**: ABS Building Activity (8752.0)  
**Table**: Building Activity, Data1 sheet  
**Series ID**: A83801544L  
**Units**: Number of dwellings per quarter  
**Frequency**: Quarterly, resampled to monthly by forward filling  
**Start**: 1984-09-01  
**End**: 2025-12-01  

---

## cpi_rents

**Description**: CPI rents sub-index, seasonally adjusted, measuring change in the price of renting a private dwelling.

**Source**: ABS Consumer Price Index (6401.0)  
**Table**: CPI Selected Living Cost Indexes, Data1 sheet  
**Series ID**: A130400542R  
**Units**: Index number (base period 2017-18 = 100)  
**Frequency**: Monthly (no resampling needed)  
**Start**: 2022-07-01  
**End**: 2026-03-01  
**Note**: Short history, only 45 months as of download. Treat as supplementary; do not impute pre-2022 values.  
