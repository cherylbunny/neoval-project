# About the Project 

This project extends the Neoval housing factor framework to provide a dynamic and interpretable analysis of regional housing price movements across Australia. Using a panel of regional price indexes derived from state valuer general data, the study applies Principal Component Analysis (PCA) to extract the dominant market, mining, and lifestyle factors driving long-term housing trends.

Building on this foundation, the analysis incorporates ARIMAX residual control and expanding-window estimation to address autocorrelation and capture evolving regional sensitivities through time. Forecast models for the market and mining factors are developed to assess temporal persistence, revealing a national trend characterised by sustained growth and a mining factor that exhibits bounded, mean-reverting dynamics.

The result is a refined three-factor model that explains the structural drivers of Australian housing markets at both national and regional levels, providing a framework that is statistically robust, interpretable, and adaptable to future data updates.

# Built With 

* R version 4.4.3 (or later)

* Core libraries: `tidyverse`, `tsibble`, `fpp3`, `forecast`, `broom`, `strucchange`, `zoo`, `tseries`, `urca`, `sf`

* Documented and visualised using Quarto

* Dependency management via renv for reproducibility

# Getting Started 

This is a research based project and does not include a user interface. 

## Clone the repository

To replicate the analysis, clone the repository and run the R scripts in sequence: 

```r
git clone https://github.com/cherylbunny/neoval-project.git
```

## Recreate the environment
To set up the R environment, ensure you have R version 4.4.3 (or higher).

This project uses `renv` to manage dependencies. To recreate the exact environment, run:

```r
install.packages("renv")
renv::restore()
```

## Project structure

**Folders**

* `_extensions/quarto-monash/report/` Custom Quarto extension for Monash branded styling used in the final report and presentation.

* `data/` Contains all datasets used and generated in the project. 

* `final_report_presentation/` Quarto files, figures, and supporting materials for the final report and presentation slides.

* `python_code/` Python scripts authored by Willem P. Sijp for related analyses. These scripts are not directly part of the `R/Quarto` workflow but are occasionally referenced for methodological comparison.

* `renv/` Environment folder managed by the renv package for reproducibility of the R environment.

**Files**

* `.gitignore` Specifies which files and folders are excluded from version control.

* `LICENSE.txt` Custom licence file (proprietary, Neoval Pty Ltd.).

* `README.md` Main project documentation.

* `neoval-project.Rproj` RStudio project file for setup.

* `project-main.Rmd` Original R Markdown version of the analysis, later migrated into the Quarto template for consistency with university formatting requirements.

* `renv.lock` Snapshot of package versions ensuring reproducible R environments.

* `styles.css` Custom stylesheet used for styling the R Markdown html.

**Notes:**

* The final analysis is contained in the Quarto files under `final_report_presentation/`, which replicate the full workflow originally developed in `project-main.Rmd`. The .qmd version is formatted according to the official university report template and serves as the definitive version of the study.

* The Python code in `python_code/` is developed by Will Sijp as part of his parallel analysis within the same project, and is referenced where relevant but not directly executed by the R workflow.

* All datasets in `data/` are version-controlled to ensure transparency and reproducibility.

## Running the analysis

Open the main analysis file (`project-main.Rmd`) or the Quarto documents under `final_report_presentation/`.

All results are reproducible from the included data and code.

The workflow proceeds through:

* Data preparation and PCA-based factor extraction

* Regression analysis and factor loading interpretation

* Forecast generation for national and mining trends

* ARIMAX residual control and expanding-window estimation

* Output visualisation and interpretation


# Methodologies 

* Data construction: Regional housing price indexes compiled from ~3 million detached house transactions (1995 - 2024).

* Factor extraction: PCA applied to regional indexes to identify national, mining, and lifestyle trends.

* Forecasting: market and mining factors projected forward to assess persistence and reversion dynamics.

* Factor modelling: regional regressions to estimate sensitivities to each factor.

* Dynamic estimation: ARIMAX residual control and expanding-window estimation to address autocorrelation and time variation.

# Further Enhauncement 

Future work could integrate macroeconomic variables to deepen interpretability. 

Or extending the model to enable real-time tracking of changing housing sensitivities and improve long-horizon forecasting precision.

# License 

This project is proprietary. Copyright © 2025 Neoval. All rights reserved. See the LICENSE file for details. 

# Contributor 

The foundational framework was developed by Willem P. Sijp, under whose supervision Yiran Yao extended the analysis to include dynamic estimation, forecasting, and temporal stability assessment.

