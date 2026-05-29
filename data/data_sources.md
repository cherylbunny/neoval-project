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

**Description**: The standard variable mortgage rate charged by banks to owner-occupier borrowers. This is the advertised "headline" rate, the rate that banks officially quote for variable-rate home loans. It moves broadly in line with the cash rate, but banks add a margin on top, so it is always higher than the cash rate. It is a key measure of the cost of borrowing for housing and directly influences how much buyers can afford to borrow.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: F5, Indicator Lending Rates  
**Series ID**: FILRHLBVS  
**Units**: Percent per annum (% p.a.)  
**Frequency**: Monthly (no resampling needed)  
**Data start**: 1959-02-01  
**Data end**: 2026-04-01  


---

## cash_rate

**Description**: The overnight cash rate target set by the Reserve Bank of Australia at its monthly board meetings. This is Australia's primary monetary policy tool. When the RBA wants to slow inflation it raises the cash rate, and when it wants to stimulate the economy it cuts it. All other interest rates in the economy, including mortgage rates and savings rates, tend to move in the same direction. Because it is event-based (it only changes when the RBA decides to move it), we forward fill the last known rate across each month.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: A2, Monetary Policy Changes  
**Series ID**: ARBAMPCNCRT  
**Units**: Percent per annum (% p.a.)  
**Frequency**: Event-based, resampled to monthly by forward filling  
**Data start**: 1990-08-01  
**Data end**: 2026-05-01  


---

## bond_yield_10y

**Description**: The yield (effective interest rate) on 10-year Australian Government bonds. This is a market-determined long-term rate. Unlike the cash rate, which the RBA sets directly, bond yields are set by buyers and sellers in financial markets. They reflect the market's expectations of future inflation and economic growth. Long-term fixed mortgage rates are priced off bond yields, so this variable captures a different dimension of borrowing costs than the cash rate. The raw data is daily; we average it to monthly.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: F2, Capital Market Yields  
**Series ID**: FCMYGBAG10D  
**Units**: Percent per annum (% p.a.)  
**Frequency**: Daily, averaged to monthly  
**Data start**: 2013-05-01  
**Data end**: 2026-05-01  


---

## housing_lending_rate

**Description**: The average interest rate actually being paid on all outstanding owner-occupier variable-rate home loans. Unlike the mortgage_rate (which is the advertised standard variable rate), this is the effective rate across the whole book of existing loans, accounting for discounts banks give to keep customers, refinancing activity, and borrower negotiation. It is the closest measure we have to what the average homeowner with a variable loan is actually paying each month.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: F6, Housing Lending Rates  
**Series ID**: FLRHOOVL  
**Units**: Percent per annum (% p.a.)  
**Frequency**: Monthly (no resampling needed)  
**Data start**: 2019-08-01  
**Data end**: 2026-03-01  


---

## housing_credit_growth

**Description**: The monthly percentage change in the total stock of housing credit outstanding across Australia, meaning how fast the total amount Australians owe on home loans is growing each month. This covers both owner-occupier and investor loans and is seasonally adjusted to remove regular calendar-based patterns. A higher number means more credit is being extended, which is associated with stronger housing demand and upward pressure on prices. The raw series is a level (total dollars); we compute the month-on-month percentage change.

**Source**: Reserve Bank of Australia (RBA)  
**Table**: D2, Lending and Credit Aggregates  
**Series ID**: DLCACOHS  
**Units**: Percent change per month (%)  
**Frequency**: Monthly (no resampling needed)  
**Data start**: 1990-02-01  
**Data end**: 2026-03-01  


---

## unemployment_rate

**Description**: The share of the labour force that is unemployed and actively looking for work, seasonally adjusted, for all persons across Australia. This is the standard measure of labour market health published by the ABS each month. A low unemployment rate generally means households have more income security and confidence to buy homes, supporting housing demand. Conversely, rising unemployment tends to dampen demand and can trigger forced sales.

**Source**: ABS Labour Force Survey (6202.0)  
**Table**: Labour Force Table 12, Data2 sheet  
**Series ID**: A84423050A  
**Units**: Percent (%)  
**Frequency**: Monthly (no resampling needed)  
**Data start**: 1978-02-01  
**Data end**: 2026-03-01  


---

## trimmed_mean_inflation

**Description**: The year-ended trimmed mean inflation rate, which is the Reserve Bank of Australia's preferred measure of underlying inflation. Unlike headline CPI (which can spike due to one-off items like fuel or fruit), the trimmed mean removes the most extreme price changes (the top and bottom 15% of items by price change each quarter) before averaging what is left. This gives a cleaner picture of sustained, broad-based inflation pressure. The RBA's official inflation target of 2-3% is assessed against this measure. The data is quarterly; we forward fill to monthly.

**Source**: Reserve Bank of Australia (RBA), compiled from ABS CPI  
**Table**: G1, Consumer Price Inflation  
**Series ID**: GCPIOCPMTMYP  
**Units**: Percent, year-ended change (%)  
**Frequency**: Quarterly, resampled to monthly by forward filling  
**Data start**: 1983-03-01  
**Data end**: 2026-03-01  


---

## household_disposable_income

**Description**: The total gross income available to Australian households after taxes and transfers have been accounted for, but before spending. This includes wages, investment income, and government payments such as welfare. It is expressed in nominal dollars and is seasonally adjusted to remove regular calendar-based patterns such as Christmas bonuses. A growing disposable income means households can afford higher repayments and larger mortgages, which supports house prices. The data is quarterly; we forward fill to monthly.

**Source**: ABS Australian National Accounts (5206.0)  
**Table**: Household Income Account, Data1 sheet  
**Series ID**: A2302939L  
**Units**: Millions of Australian dollars ($ millions, nominal)  
**Frequency**: Quarterly, resampled to monthly by forward filling  
**Data start**: 1959-09-01  
**Data end**: 2025-12-01  


---

## net_overseas_migration

**Description**: The net flow of people arriving in Australia from overseas on a permanent or long-term basis, meaning arrivals minus departures. This is the single largest driver of population growth in Australia and therefore a direct driver of housing demand, as every additional person who arrives needs somewhere to live. During the COVID period (2020-2021), net overseas migration turned negative for the first time in decades, removing a major demand driver. The post-COVID surge to record highs in 2022-2023 significantly amplified rental and purchase demand. The data is quarterly; we forward fill to monthly.

**Source**: ABS Demographic Statistics (3101.0)  
**Table**: National, State and Territory Population, Data1 sheet  
**Series ID**: A2133254C  
**Units**: Thousands of persons per quarter (000)  
**Frequency**: Quarterly, resampled to monthly by forward filling  
**Data start**: 1981-06-01  
**Data end**: 2025-09-01  


---

## building_approvals

**Description**: The total number of new dwellings approved for construction across all sectors (private and public) each month, seasonally adjusted. Building approvals are a leading indicator of future housing supply, as a dwelling must be approved before it can be built. When approvals fall, fewer homes will be completed in the months ahead, tightening supply. However, there is typically a lag of 12-24 months between approval and completion, so this variable leads dwelling completions.

**Source**: ABS Building Approvals (8731.0)  
**Table**: Building Approvals, Data1 sheet  
**Series ID**: A422070J  
**Units**: Number of dwellings per month  
**Frequency**: Monthly (no resampling needed)  
**Data start**: 1983-07-01  
**Data end**: 2026-03-01  


---

## dwelling_commencements

**Description**: The total number of new dwellings on which construction actually began each quarter, across all sectors, seasonally adjusted. Unlike building approvals (which measure intention), commencements measure when building actually starts. Not all approved dwellings are commenced; some are cancelled or deferred. This variable therefore gives a more accurate picture of actual construction activity and near-term additions to housing supply. The data is quarterly; we forward fill to monthly.

**Source**: ABS Building Activity (8752.0)  
**Table**: Building Activity, Data1 sheet  
**Series ID**: A83801544L  
**Units**: Number of dwellings per quarter  
**Frequency**: Quarterly, resampled to monthly by forward filling  
**Data start**: 1984-09-01  
**Data end**: 2025-12-01  


---

## cpi_rents

**Description**: The rents component of the Consumer Price Index (CPI), seasonally adjusted. This measures how much the price of renting a private dwelling has changed over time. It is distinct from market rents (which reflect what new tenants pay). The CPI rents index captures the average across all rental agreements, including longstanding leases where rent may not have been reviewed recently. It therefore tends to move more slowly than advertised rents. This monthly series was only introduced by the ABS in mid-2022 when CPI moved to a monthly release; pre-2022 data is not available in monthly form.

**Source**: ABS Consumer Price Index (6401.0)  
**Table**: CPI Selected Living Cost Indexes, Data1 sheet  
**Series ID**: A130400542R  
**Units**: Index number (base period 2017-18 = 100)  
**Frequency**: Monthly (no resampling needed)  
**Data start**: 2022-07-01  
**Data end**: 2026-03-01  
**Note**: Short history, only 45 months as of download. Treat as supplementary; do not impute pre-2022 values.  
