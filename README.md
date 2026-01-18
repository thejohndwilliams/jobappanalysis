# Job Application Analysis

ML-powered analysis of job application data to identify rejection patterns and generate actionable insights.

## Features

- **Rejection Prediction** — Random Forest classification with 98%+ accuracy
- - **Anomaly Detection** — Identifies unusual application patterns
  - - **Automated Insights** — Quantifies impact of referrals, timing, and sources
    - - **Visualizations** — Feature importance, industry trends, timeline analysis
     
      - ## Quick Start
     
      - ```bash
        pip install -r requirements.txt
        python jobappanalysis.py -i your_data.xlsx -o results.xlsx
        ```

        ## Input Format

        Excel file with sheets `2024`, `2025` containing columns:
        - Company, Job Title, Application Date
        - - Screening, Interview, Final Round, Rejected, Offer (dates)
          - - Interval, Application Source, Referral, Resume Version
            - - Industry, Seniority Level, Cover Letter
             
              - ## Output
             
              - - `JobAppResults.xlsx` — Summary metrics, insights, predictions, anomalies
                - - `plots/` — Feature importance, outcomes by source, rejection by industry
                 
                  - ## CLI Options
                 
                  - ```
                    -i, --input         Input Excel path
                    -o, --output        Output Excel path
                    --plots-dir         Visualization directory
                    --no-plots          Skip plot generation
                    --contamination     Anomaly sensitivity (default: 0.1)
                    ```

                    ## Requirements

                    Python 3.10+, pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl

                    ## License

                    MIT
