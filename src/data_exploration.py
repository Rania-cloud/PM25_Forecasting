# ============================================================
# PM2.5 FORECASTING PROJECT
# Mission 1: Data Exploration and Preprocessing
# Step 1: Loading and Initial Dataset Inspection
# ============================================================

import pandas as pd
import numpy as np
import os

# Define the paths of our datasets
DATA_PATH = r"C:\Users\rania\Desktop\PM25_Forecasting\data\raw"

combined_path = os.path.join(DATA_PATH, "combined_gee_pm25.csv")
features_path = os.path.join(DATA_PATH, "gee_features_daily.csv")
pm25_path = os.path.join(DATA_PATH, "pm25.csv")

# Load the three datasets
combined_df = pd.read_csv(combined_path)
features_df = pd.read_csv(features_path)
pm25_df = pd.read_csv(pm25_path)

print("All datasets loaded successfully!")

# Display the dimensions of each dataset

print("\n========== DATASET DIMENSIONS ==========")

print("Combined dataset:", combined_df.shape)
print("Features dataset:", features_df.shape)
print("PM2.5 dataset:", pm25_df.shape)

# Display column names and data types

datasets = {
    "Combined": combined_df,
    "Features": features_df,
    "PM25": pm25_df
}

for name, df in datasets.items():

    print(f"\n========== {name} DATASET ==========")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

for name, df in datasets.items():

    print(f"\n========== {name} DATASET ==========")

    print("\nFirst 5 observations:")
    print(df.head())

    print("\nLast 5 observations:")
    print(df.tail())

# Convert date columns temporarily for inspection

for name, df in datasets.items():

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print(f"\n========== {name} DATASET ==========")

    print("Earliest date:", df["date"].min())
    print("Latest date:", df["date"].max())

    if "id" in df.columns:
        print("Number of unique station IDs:", df["id"].nunique())

    elif "sensor_name" in df.columns:
        print("Number of unique stations:", df["sensor_name"].nunique())


# Create the output directory if it does not exist

os.makedirs("results/tables", exist_ok=True)

# Create a summary table

summary = []

for name, df in datasets.items():

    summary.append({
        "Dataset": name,
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Start_Date": df["date"].min(),
        "End_Date": df["date"].max(),
        "Missing_Values": df.isna().sum().sum(),
        "Duplicate_Rows": df.duplicated().sum()
    })

summary_df = pd.DataFrame(summary)

print("\n========== INITIAL DATASET SUMMARY ==========")
print(summary_df)

# Export the summary
summary_df.to_csv(
    "results/tables/initial_dataset_summary.csv",
    index=False
)

print("\nInitial summary exported successfully!")


# ============================================================
# PM2.5 FORECASTING PROJECT
# MISSION 1: DATA EXPLORATION AND PREPROCESSING
# STEP 2: DATA QUALITY AUDIT
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

# Create output folders
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ============================================================
# STEP 2.1: DATASET STRUCTURE
# ============================================================

print("\n========== STEP 2: DATASET STRUCTURE ==========")

structure_results = []

for name, df in datasets.items():

    for column in df.columns:

        structure_results.append({
            "Dataset": name,
            "Variable": column,
            "Data_Type": str(df[column].dtype),
            "Unique_Values": df[column].nunique(),
            "Missing_Values": df[column].isna().sum()
        })

structure_df = pd.DataFrame(structure_results)

structure_df.to_csv(
    "results/tables/dataset_structure.csv",
    index=False
)

print("Dataset structure exported successfully.")


# ============================================================
# STEP 2.2: MISSING VALUES ANALYSIS
# ============================================================

print("\n========== MISSING VALUES ANALYSIS ==========")

missing_results = []

for name, df in datasets.items():

    for column in df.columns:

        missing_count = df[column].isna().sum()

        missing_percentage = (
            missing_count / len(df) * 100
        )

        missing_results.append({
            "Dataset": name,
            "Variable": column,
            "Missing_Count": missing_count,
            "Missing_Percentage": round(
                missing_percentage, 2
            )
        })

missing_df = pd.DataFrame(missing_results)

missing_df.to_csv(
    "results/tables/missing_values_audit.csv",
    index=False
)

print(missing_df.to_string(index=False))


# Visualize missing values in the combined dataset

plt.figure(figsize=(12, 6))

sns.heatmap(
    combined_df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title("Missing Values - Combined Dataset")
plt.xlabel("Variables")

plt.tight_layout()

plt.savefig(
    "results/figures/missing_values_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 2.3: DUPLICATE ANALYSIS
# ============================================================

print("\n========== DUPLICATE ANALYSIS ==========")

duplicate_results = []

for name, df in datasets.items():

    exact_duplicates = df.duplicated().sum()

    if "id" in df.columns:
        key_columns = ["date", "id"]

    elif "sensor_name" in df.columns:
        key_columns = ["date", "sensor_name"]

    else:
        key_columns = ["date"]

    repeated_records = df.duplicated(
        subset=key_columns,
        keep=False
    ).sum()

    duplicate_results.append({
        "Dataset": name,
        "Exact_Duplicates": exact_duplicates,
        "Repeated_Date_Station_Records": repeated_records,
        "Key_Columns": ", ".join(key_columns)
    })

duplicate_df = pd.DataFrame(duplicate_results)

duplicate_df.to_csv(
    "results/tables/duplicate_analysis.csv",
    index=False
)

print(duplicate_df.to_string(index=False))


# ============================================================
# STEP 3: PM2.5 TARGET ANALYSIS
# ============================================================

print("\n========== STEP 3: TARGET ANALYSIS ==========")

target = "pm25"

target_data = combined_df[target]

total_observations = len(target_data)

missing_target = target_data.isna().sum()

nonpositive_target = (
    target_data.notna() & (target_data <= 0)
).sum()

high_target = (
    target_data.notna() & (target_data >= 500)
).sum()

valid_target = (
    target_data.notna()
    & (target_data > 0)
    & (target_data < 500)
).sum()

target_summary = pd.DataFrame({
    "Metric": [
        "Total observations",
        "Missing PM2.5",
        "PM2.5 <= 0",
        "PM2.5 >= 500",
        "Valid PM2.5 observations"
    ],

    "Count": [
        total_observations,
        missing_target,
        nonpositive_target,
        high_target,
        valid_target
    ]
})

target_summary["Percentage"] = (
    target_summary["Count"] / total_observations * 100
).round(2)

target_summary.to_csv(
    "results/tables/target_analysis.csv",
    index=False
)

print(target_summary.to_string(index=False))


# ============================================================
# STEP 4: DESCRIPTIVE STATISTICS
# ============================================================

print("\n========== STEP 4: DESCRIPTIVE STATISTICS ==========")

numeric_columns = combined_df.select_dtypes(
    include=np.number
).columns.tolist()

descriptive_df = combined_df[
    numeric_columns
].describe().T

descriptive_df["Skewness"] = (
    combined_df[numeric_columns].skew()
)

descriptive_df["Kurtosis"] = (
    combined_df[numeric_columns].kurtosis()
)

descriptive_df.to_csv(
    "results/tables/descriptive_statistics.csv"
)

print(descriptive_df.round(3))


# ============================================================
# STEP 5: CATEGORICAL VARIABLES AND STATIONS
# ============================================================

print("\n========== STEP 5: CATEGORICAL ANALYSIS ==========")

categorical_columns = combined_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for column in categorical_columns:

    frequency_table = (
        combined_df[column]
        .value_counts(dropna=False)
        .reset_index()
    )

    frequency_table.columns = [
        column,
        "Count"
    ]

    frequency_table["Percentage"] = (
        frequency_table["Count"]
        / len(combined_df) * 100
    ).round(2)

    frequency_table.to_csv(
        f"results/tables/frequency_{column}.csv",
        index=False
    )


# Number of observations per station

station_counts = (
    combined_df.groupby(
        ["id", "sensor_name"],
        dropna=False
    )
    .size()
    .reset_index(name="Observation_Count")
    .sort_values(
        "Observation_Count",
        ascending=False
    )
)

station_counts.to_csv(
    "results/tables/station_observations.csv",
    index=False
)

print(station_counts.to_string(index=False))


# ============================================================
# STEP 6: PM2.5 VISUALIZATION
# ============================================================

print("\n========== STEP 6: PM2.5 VISUALIZATION ==========")

valid_pm25 = combined_df.loc[
    combined_df["pm25"].notna()
    & (combined_df["pm25"] > 0)
    & (combined_df["pm25"] < 500),
    "pm25"
]

# Histogram

plt.figure(figsize=(10, 6))

sns.histplot(
    valid_pm25,
    bins=50,
    kde=True
)

plt.title("Distribution of PM2.5")
plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "results/figures/pm25_distribution.png",
    dpi=300
)

plt.close()


# Boxplot

plt.figure(figsize=(10, 5))

sns.boxplot(x=valid_pm25)

plt.title("PM2.5 Boxplot")
plt.xlabel("PM2.5 (µg/m³)")

plt.tight_layout()

plt.savefig(
    "results/figures/pm25_boxplot.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 7: OUTLIER ANALYSIS
# ============================================================

print("\n========== STEP 7: OUTLIER ANALYSIS ==========")

outlier_results = []

for column in numeric_columns:

    values = combined_df[column].dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (values < lower_bound)
        | (values > upper_bound)
    ).sum()

    outlier_results.append({
        "Variable": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": int(outlier_count),
        "Outlier_Percentage": round(
            outlier_count / len(values) * 100, 2
        )
    })

outlier_df = pd.DataFrame(outlier_results)

outlier_df.to_csv(
    "results/tables/outlier_analysis.csv",
    index=False
)

print(outlier_df.to_string(index=False))


# ============================================================
# STEP 8: CORRELATION ANALYSIS
# ============================================================

print("\n========== STEP 8: CORRELATION ANALYSIS ==========")

# Use valid PM2.5 observations for target correlations

correlation_data = combined_df.loc[
    combined_df["pm25"].notna()
    & (combined_df["pm25"] > 0)
    & (combined_df["pm25"] < 500),
    numeric_columns
].copy()

# Remove station ID and geographic coordinates from
# the correlation analysis of physical variables.

excluded_columns = [
    "id",
    "latitude",
    "longitude"
]

correlation_columns = [
    column for column in correlation_data.columns
    if column not in excluded_columns
]

correlation_data = correlation_data[
    correlation_columns
]

pearson_corr = correlation_data.corr(
    method="pearson"
)

spearman_corr = correlation_data.corr(
    method="spearman"
)

pearson_corr.to_csv(
    "results/tables/pearson_correlation.csv"
)

spearman_corr.to_csv(
    "results/tables/spearman_correlation.csv"
)


# Pearson heatmap

plt.figure(figsize=(12, 9))

sns.heatmap(
    pearson_corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Pearson Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "results/figures/pearson_correlation.png",
    dpi=300
)

plt.close()


# Spearman heatmap

plt.figure(figsize=(12, 9))

sns.heatmap(
    spearman_corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Spearman Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "results/figures/spearman_correlation.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 9: TEMPORAL ANALYSIS
# ============================================================

print("\n========== STEP 9: TEMPORAL ANALYSIS ==========")

# Keep only valid PM2.5 observations

temporal_df = combined_df.loc[
    combined_df["pm25"].notna()
    & (combined_df["pm25"] > 0)
    & (combined_df["pm25"] < 500)
].copy()

temporal_df = temporal_df.sort_values("date")

temporal_df["Year"] = temporal_df["date"].dt.year
temporal_df["Month"] = temporal_df["date"].dt.month
temporal_df["DayOfWeek"] = temporal_df["date"].dt.dayofweek


# Daily descriptive summary across stations

daily_summary = (
    temporal_df.groupby("date")["pm25"]
    .agg(["mean", "median", "std", "count"])
    .reset_index()
)

daily_summary.columns = [
    "Date",
    "Mean_PM25",
    "Median_PM25",
    "Standard_Deviation",
    "Number_of_Observations"
]

daily_summary.to_csv(
    "results/tables/daily_pm25_summary.csv",
    index=False
)


# Monthly analysis

monthly_summary = (
    temporal_df.groupby("Month")["pm25"]
    .agg(["mean", "median", "std", "count"])
    .reset_index()
)

monthly_summary.to_csv(
    "results/tables/monthly_pm25_summary.csv",
    index=False
)


# Yearly analysis

yearly_summary = (
    temporal_df.groupby("Year")["pm25"]
    .agg(["mean", "median", "std", "count"])
    .reset_index()
)

yearly_summary.to_csv(
    "results/tables/yearly_pm25_summary.csv",
    index=False
)


# Daily trend

plt.figure(figsize=(15, 6))

plt.plot(
    daily_summary["Date"],
    daily_summary["Mean_PM25"]
)

plt.title("Daily Average PM2.5")
plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")

plt.tight_layout()

plt.savefig(
    "results/figures/daily_pm25_trend.png",
    dpi=300
)

plt.close()


# Monthly distribution

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=temporal_df,
    x="Month",
    y="pm25"
)

plt.title("PM2.5 Distribution by Month")
plt.xlabel("Month")
plt.ylabel("PM2.5 (µg/m³)")

plt.tight_layout()

plt.savefig(
    "results/figures/pm25_monthly_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 10: TEMPORAL REGULARITY AND MISSING DATES
# ============================================================

print("\n========== STEP 10: TEMPORAL REGULARITY ==========")

temporal_audit = []

for station_id, group in temporal_df.groupby("id"):

    station_dates = (
        group["date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(station_dates) == 0:
        continue

    expected_dates = pd.date_range(
        start=station_dates.min(),
        end=station_dates.max(),
        freq="D"
    )

    missing_dates = expected_dates.difference(
        pd.DatetimeIndex(station_dates)
    )

    temporal_audit.append({
        "Station_ID": station_id,
        "Station_Name": group["sensor_name"].iloc[0],
        "First_Date": station_dates.min(),
        "Last_Date": station_dates.max(),
        "Observed_Dates": len(station_dates),
        "Expected_Daily_Dates": len(expected_dates),
        "Missing_Dates": len(missing_dates),
        "Missing_Percentage": round(
            len(missing_dates) / len(expected_dates) * 100,
            2
        )
    })

temporal_audit_df = pd.DataFrame(temporal_audit)

temporal_audit_df.to_csv(
    "results/tables/Exploratory_temporal.csv",
    index=False
)

print(temporal_audit_df.to_string(index=False))


# ============================================================
# STEP 11: STATIONARITY TEST - ADF
# ============================================================

print("\n========== STEP 11: STATIONARITY TEST ==========")

# Select the station with the largest number of valid observations.

station_counts_valid = (
    temporal_df.groupby("id")
    .size()
    .sort_values(ascending=False)
)

adf_results = []

if len(station_counts_valid) > 0:

    selected_station_id = station_counts_valid.index[0]

    selected_station = temporal_df[
        temporal_df["id"] == selected_station_id
    ].copy()

    # One observation per station and day
    selected_station = (
        selected_station.groupby("date")["pm25"]
        .mean()
        .sort_index()
    )

    # Reindex to daily frequency.
    # Do not fill missing dates with future information.

    full_dates = pd.date_range(
        start=selected_station.index.min(),
        end=selected_station.index.max(),
        freq="D"
    )

    selected_station = selected_station.reindex(full_dates)

    # Select the longest uninterrupted daily segment
    valid_groups = (
        selected_station.notna()
        .groupby(selected_station.isna().cumsum())
    )

    segments = [
        group.dropna()
        for _, group in valid_groups
        if group.notna().any()
    ]

    if segments:

        longest_segment = max(segments, key=len)

        if len(longest_segment) >= 30:

            adf_result = adfuller(
                longest_segment,
                autolag="AIC"
            )

            adf_results.append({
                "Station_ID": selected_station_id,
                "Observations": len(longest_segment),
                "ADF_Statistic": adf_result[0],
                "P_Value": adf_result[1],
                "Lags_Used": adf_result[2],
                "Critical_Value_1%": adf_result[4]["1%"],
                "Critical_Value_5%": adf_result[4]["5%"],
                "Critical_Value_10%": adf_result[4]["10%"],
                "Conclusion": (
                    "Stationary"
                    if adf_result[1] < 0.05
                    else "Non-stationary"
                )
            })

            # ACF and PACF plots
            lags = min(
                40,
                len(longest_segment) // 2 - 1
            )

            if lags >= 2:

                fig, axes = plt.subplots(
                    2, 1,
                    figsize=(12, 8)
                )

                plot_acf(
                    longest_segment,
                    lags=lags,
                    ax=axes[0]
                )

                plot_pacf(
                    longest_segment,
                    lags=lags,
                    ax=axes[1],
                    method="ywm"
                )

                axes[0].set_title("ACF - Selected Station")
                axes[1].set_title("PACF - Selected Station")

                plt.tight_layout()

                plt.savefig(
                    "results/figures/acf_pacf.png",
                    dpi=300
                )

                plt.close()

adf_df = pd.DataFrame(adf_results)

adf_df.to_csv(
    "results/tables/stationarity_adf.csv",
    index=False
)

print(adf_df.to_string(index=False))


# ============================================================
# STEP 12: CLEAN INVALID TARGET OBSERVATIONS
# ============================================================

print("\n========== STEP 12: TARGET CLEANING ==========")

cleaned_df = combined_df.copy()

initial_rows = len(cleaned_df)

# Apply the teacher's target validity conditions:
# Remove missing PM2.5, PM2.5 <= 0 and PM2.5 >= 500.

cleaned_df = cleaned_df[
    cleaned_df["pm25"].notna()
    & (cleaned_df["pm25"] > 0)
    & (cleaned_df["pm25"] < 500)
].copy()

cleaned_df = cleaned_df.sort_values(
    ["id", "date"]
).reset_index(drop=True)

removed_rows = initial_rows - len(cleaned_df)

cleaning_summary = pd.DataFrame({
    "Metric": [
        "Initial observations",
        "Final observations",
        "Removed observations",
        "Missing PM2.5",
        "PM2.5 <= 0",
        "PM2.5 >= 500"
    ],

    "Count": [
        initial_rows,
        len(cleaned_df),
        removed_rows,
        missing_target,
        nonpositive_target,
        high_target
    ]
})

cleaning_summary.to_csv(
    "results/tables/cleaning_summary.csv",
    index=False
)

print(cleaning_summary.to_string(index=False))


# ============================================================
# STEP 13: MISSINGNESS INDICATORS
# ============================================================

print("\n========== STEP 13: MISSINGNESS INDICATORS ==========")

# Create indicators for satellite variables.
# Do not impute values using the complete dataset.

for column in ["NO2", "CO", "O3"]:

    if column in cleaned_df.columns:

        cleaned_df[f"{column}_missing"] = (
            cleaned_df[column].isna().astype(int)
        )


remaining_missing = pd.DataFrame({
    "Variable": cleaned_df.columns,
    "Missing_Count": [
        cleaned_df[column].isna().sum()
        for column in cleaned_df.columns
    ]
})

remaining_missing["Missing_Percentage"] = (
    remaining_missing["Missing_Count"]
    / len(cleaned_df) * 100
).round(2)

remaining_missing = remaining_missing[
    remaining_missing["Missing_Count"] > 0
]

remaining_missing.to_csv(
    "results/tables/remaining_missing_values.csv",
    index=False
)


# ============================================================
# STEP 14: EXPORT CLEANED DATASET
# ============================================================

cleaned_df.to_csv(
    "data/processed/cleaned_pm25_dataset.csv",
    index=False
)

print("\nCleaned dataset exported successfully.")


# ============================================================
# STEP 15: EXPORT EXPLORATORY_DATA.CSV
# ============================================================

exploratory_data = []

for column in cleaned_df.select_dtypes(
    include=np.number
).columns:

    values = cleaned_df[column].dropna()

    if len(values) == 0:
        continue

    exploratory_data.append({
        "Variable": column,
        "Count": len(values),
        "Missing": cleaned_df[column].isna().sum(),
        "Mean": values.mean(),
        "Median": values.median(),
        "Standard_Deviation": values.std(),
        "Minimum": values.min(),
        "Q1": values.quantile(0.25),
        "Q3": values.quantile(0.75),
        "Maximum": values.max(),
        "Skewness": values.skew(),
        "Kurtosis": values.kurtosis()
    })

exploratory_data_df = pd.DataFrame(exploratory_data)

exploratory_data_df.to_csv(
    "results/tables/Exploratory_Data.csv",
    index=False
)


# ============================================================
# STEP 16: FINAL DATA QUALITY VERIFICATION
# ============================================================

print("\n========== FINAL DATA QUALITY CHECK ==========")

final_quality = pd.DataFrame({
    "Check": [
        "Final number of rows",
        "Final number of columns",
        "Invalid PM2.5 <= 0",
        "Invalid PM2.5 >= 500",
        "Exact duplicate rows",
        "Missing predictor values"
    ],

    "Result": [
        len(cleaned_df),
        len(cleaned_df.columns),
        (cleaned_df["pm25"] <= 0).sum(),
        (cleaned_df["pm25"] >= 500).sum(),
        cleaned_df.duplicated().sum(),
        cleaned_df.drop(columns=["pm25"]).isna().sum().sum()
    ]
})

final_quality.to_csv(
    "results/tables/final_data_quality_check.csv",
    index=False
)

print(final_quality.to_string(index=False))


# ============================================================
# STEP 17: FINAL MISSION SUMMARY
# ============================================================

summary_text = f"""
PM2.5 FORECASTING PROJECT
MISSION 1: DATA EXPLORATION AND PREPROCESSING

DATASET:
Initial observations: {initial_rows}
Final observations: {len(cleaned_df)}
Number of variables: {len(cleaned_df.columns)}

TARGET:
PM2.5 (µg/m³)

INVALID TARGET OBSERVATIONS:
Missing: {missing_target}
PM2.5 <= 0: {nonpositive_target}
PM2.5 >= 500: {high_target}

CLEANING:
Removed observations: {removed_rows}

STATIONS:
{cleaned_df['id'].nunique()}

DATE RANGE:
{cleaned_df['date'].min()} to {cleaned_df['date'].max()}

MISSING PREDICTORS:
Missing values have been documented.
They have not been globally imputed.
Imputation must be fitted on training data only.

TIME SERIES:
Stationarity was investigated using the ADF test.
ACF and PACF plots were generated when sufficient
consecutive daily observations were available.

IMPORTANT:
Daily averages across stations are descriptive summaries.
They must not be confused with station-specific forecasting.

The final modeling station and temporal validation strategy
must be established before model training.
"""

with open(
    "results/Mission1_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(summary_text)


print("\n" + "=" * 65)
print("MISSION 1 ANALYSIS FINISHED")
print("=" * 65)

print("\nAll tables saved in: results/tables/")
print("All figures saved in: results/figures/")
print("Cleaned dataset: data/processed/cleaned_pm25_dataset.csv")
print("Summary: results/Mission1_summary.txt")

print("\nPlease verify the exported results before submission.")

print("=" * 65)

# ============================================================
# STEP 18: STATION SELECTION AND CONTINUITY DIAGNOSTICS
# ============================================================

print("\n========== STATION CONTINUITY DIAGNOSTICS ==========")

station_diagnostics = []

for station_id, group in cleaned_df.groupby("id"):

    group = group.sort_values("date")

    station_name = group["sensor_name"].iloc[0]

    # One PM2.5 observation per date
    series = (
        group.groupby("date")["pm25"]
        .mean()
        .sort_index()
    )

    # Expected daily calendar
    full_dates = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="D"
    )

    regular_series = series.reindex(full_dates)

    # Missing dates
    missing_dates = regular_series.isna().sum()

    expected_dates = len(regular_series)

    missing_percentage = (
        missing_dates / expected_dates * 100
    )

    # Identify consecutive non-missing segments
    groups = regular_series.isna().cumsum()

    segments = [
        segment.dropna()
        for _, segment in regular_series.groupby(groups)
        if segment.notna().any()
    ]

    if segments:

        longest_segment = max(segments, key=len)

        longest_segment_length = len(longest_segment)

        segment_start = longest_segment.index.min()

        segment_end = longest_segment.index.max()

    else:

        longest_segment_length = 0

        segment_start = None

        segment_end = None

    station_diagnostics.append({

        "Station_ID": station_id,

        "Station_Name": station_name,

        "Valid_Observations": len(series),

        "Expected_Daily_Observations": expected_dates,

        "Missing_Dates": missing_dates,

        "Missing_Percentage": round(
            missing_percentage, 2
        ),

        "Longest_Continuous_Segment": longest_segment_length,

        "Segment_Start": segment_start,

        "Segment_End": segment_end

    })


# Convert results to DataFrame

station_diagnostics_df = pd.DataFrame(
    station_diagnostics
)

# Rank stations by longest continuous segment

station_diagnostics_df = (
    station_diagnostics_df.sort_values(
        by="Longest_Continuous_Segment",
        ascending=False
    )
)

# Save results

station_diagnostics_df.to_csv(
    "results/tables/station_continuity_ranking.csv",
    index=False
)

print("\nTOP 15 STATIONS BY CONTINUOUS OBSERVATIONS")

print(
    station_diagnostics_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# STEP 19: VISUALIZE STATION AVAILABILITY
# ============================================================

top_stations = station_diagnostics_df.head(15)

plt.figure(figsize=(13, 7))

sns.barplot(
    data=top_stations,
    x="Longest_Continuous_Segment",
    y="Station_Name"
)

plt.title(
    "Top 15 Stations by Longest Continuous Daily Segment"
)

plt.xlabel("Number of Consecutive Daily Observations")

plt.ylabel("Station")

plt.tight_layout()

plt.savefig(
    "results/figures/station_continuity_ranking.png",
    dpi=300
)

plt.close()


# ============================================================
# STEP 20: IDENTIFY POTENTIAL STATIONS FOR SARIMA
# ============================================================

# This is a diagnostic threshold, not a final modeling decision.
# We require at least 90 consecutive daily observations.

potential_stations = station_diagnostics_df[
    station_diagnostics_df[
        "Longest_Continuous_Segment"
    ] >= 90
].copy()

potential_stations.to_csv(
    "results/tables/potential_sarima_stations.csv",
    index=False
)

print("\nPOTENTIAL STATIONS FOR SARIMA")

if len(potential_stations) > 0:

    print(
        potential_stations.to_string(index=False)
    )

else:

    print(
        "No station currently has a continuous segment "
        "of at least 90 days."
    )


print("\nStation continuity diagnostics completed.")

print(
    "\nResults saved in "
    "results/tables/station_continuity_ranking.csv"
)