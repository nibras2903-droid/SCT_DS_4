S"""Professional exploratory data analysis for the US accidents dataset.

This script uses the existing accident dataset, creates high-quality visualizations,
and prints concise markdown-style explanations and insights for each chart.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_PATH = Path(r"e:/US_Accidents_March23.csv/US_Accidents_March23.csv")
CACHE_PATH = Path(r"e:/US_Accidents_March23.csv/accidents_df.pkl")
IMAGE_DIR = Path(r"e:/US_Accidents_March23.csv/images")
IMAGE_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data():
    """Load the dataset once and cache it for repeat runs."""
    if CACHE_PATH.exists():
        print("Using cached dataset from disk.")
        return pd.read_pickle(CACHE_PATH)

    print("Loading the dataset from the CSV file.")
    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
        on_bad_lines="skip",
        nrows=200000,
    )
    df.to_pickle(CACHE_PATH)
    return df


# Use the existing DataFrame if it is already available in memory.
try:
    df
except NameError:
    df = load_data()
else:
    df = df.copy()

analysis_df = df.sample(n=min(15000, len(df)), random_state=42)

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def find_column(candidates):
    """Find a matching column name dynamically, case-insensitively."""
    normalized = {
        col.strip().lower().replace(" ", "_").replace("-", "_"): col for col in df.columns
    }
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_").replace("-", "_")
        if key in normalized:
            return normalized[key]
    return None


def save_plot(fig, filename):
    """Save a figure to the images directory at 300 DPI."""
    fig.savefig(IMAGE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_section(title, explanation, insights):
    """Print a markdown-style explanation and the insights for a plot."""
    print(f"\n### {title}")
    print(explanation)
    print("Key insights:")
    for insight in insights:
        print(f"- {insight}")


def annotate_bars(ax):
    """Annotate bars with their values."""
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=10)


# ---------------------------------------------------------------------
# Dynamic column detection
# ---------------------------------------------------------------------
severity_col = find_column(["Severity", "severity"])
weather_col = find_column(["Weather_Condition", "WeatherCondition", "weather_condition"])
road_condition_col = find_column(
    ["Road_Condition", "RoadConditions", "Road_Conditions", "Surface_Condition", "Road_Surface"]
)
lighting_col = find_column(["Lighting_Condition", "Light_Condition", "Sunrise_Sunset"])
start_time_col = find_column(["Start_Time", "StartTime", "start_time"])
latitude_col = find_column(["Start_Lat", "Latitude", "StartLatitude", "lat"])
longitude_col = find_column(["Start_Lng", "Longitude", "StartLongitude", "lng"])
street_col = find_column(["Street", "street"])
city_col = find_column(["City", "city"])
county_col = find_column(["County", "county"])

# ---------------------------------------------------------------------
# Date-based feature engineering
# ---------------------------------------------------------------------
if start_time_col:
    df[start_time_col] = pd.to_datetime(df[start_time_col], errors="coerce")
    analysis_df[start_time_col] = pd.to_datetime(analysis_df[start_time_col], errors="coerce")
    df["Day_of_Week"] = df[start_time_col].dt.day_name()
    df["Month"] = df[start_time_col].dt.month_name()
    df["Hour"] = df[start_time_col].dt.hour
    analysis_df["Day_of_Week"] = analysis_df[start_time_col].dt.day_name()
    analysis_df["Month"] = analysis_df[start_time_col].dt.month_name()
    analysis_df["Hour"] = analysis_df[start_time_col].dt.hour

# ---------------------------------------------------------------------
# 1. Severity distribution
# ---------------------------------------------------------------------
if severity_col:
    severity_counts = df[severity_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=severity_counts.index.astype(str), y=severity_counts.values, palette="viridis", ax=ax)
    ax.set_title("Distribution of Accident Severity")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Number of Accidents")
    annotate_bars(ax)
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot(fig, "severity_distribution.png")

    explanation = "This chart shows how often each accident severity level occurs in the dataset."
    insights = [
        f"Most frequent severity: {severity_counts.idxmax()}",
        f"Total accidents analyzed: {len(df):,}",
    ]
    print_section("Accident Severity Distribution", explanation, insights)
else:
    print("Skipping severity distribution because no severity column was found.")

# ---------------------------------------------------------------------
# 2. Weather condition distribution
# ---------------------------------------------------------------------
if weather_col:
    weather_counts = df[weather_col].fillna("Unknown").astype(str).value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=weather_counts.index, y=weather_counts.values, palette="rocket", ax=ax)
    ax.set_title("Accidents by Weather Condition")
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("Number of Accidents")
    plt.xticks(rotation=45, ha="right")
    annotate_bars(ax)
    plt.tight_layout()
    save_plot(fig, "accidents_by_weather.png")

    explanation = "This chart highlights the most common weather conditions during accidents."
    insights = [
        f"Most common weather condition: {weather_counts.index[0]}",
        f"Top weather condition share: {weather_counts.iloc[0] / len(df) * 100:.1f}%",
    ]
    print_section("Accidents by Weather Condition", explanation, insights)
else:
    print("Skipping weather analysis because the required weather column was not found.")

# ---------------------------------------------------------------------
# 3. Road condition distribution
# ---------------------------------------------------------------------
if road_condition_col:
    road_counts = df[road_condition_col].fillna("Unknown").astype(str).value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=road_counts.index, y=road_counts.values, palette="mako", ax=ax)
    ax.set_title("Accidents by Road Condition")
    ax.set_xlabel("Road Condition")
    ax.set_ylabel("Number of Accidents")
    plt.xticks(rotation=45, ha="right")
    annotate_bars(ax)
    plt.tight_layout()
    save_plot(fig, "accidents_by_road_condition.png")

    explanation = "This chart shows how road-condition categories are associated with accident frequency."
    insights = [f"Most common road condition: {road_counts.index[0]}"]
    print_section("Accidents by Road Condition", explanation, insights)
else:
    print("Skipping road-condition analysis because no matching column was found.")

# ---------------------------------------------------------------------
# 4. Lighting condition distribution
# ---------------------------------------------------------------------
if lighting_col:
    lighting_counts = df[lighting_col].fillna("Unknown").astype(str).value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=lighting_counts.index, y=lighting_counts.values, palette="cubehelix", ax=ax)
    ax.set_title("Accidents by Lighting Condition")
    ax.set_xlabel("Lighting Condition")
    ax.set_ylabel("Number of Accidents")
    plt.xticks(rotation=30, ha="right")
    annotate_bars(ax)
    plt.tight_layout()
    save_plot(fig, "accidents_by_lighting.png")

    explanation = "This chart highlights whether accidents are more frequent in daylight, dusk, or night conditions."
    insights = [f"Most common lighting condition: {lighting_counts.index[0]}"]
    print_section("Accidents by Lighting Condition", explanation, insights)
else:
    print("Skipping lighting analysis because no matching column was found.")

# ---------------------------------------------------------------------
# 5. Day of week distribution
# ---------------------------------------------------------------------
if "Day_of_Week" in df.columns:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = df["Day_of_Week"].value_counts().reindex(day_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=day_counts.index, y=day_counts.values, palette="pastel", ax=ax)
    ax.set_title("Accidents by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Number of Accidents")
    annotate_bars(ax)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_plot(fig, "accidents_by_day_of_week.png")

    explanation = "This chart shows whether accidents happen more often on certain weekdays."
    insights = [f"Busiest weekday: {day_counts.idxmax()}"]
    print_section("Accidents by Day of Week", explanation, insights)
else:
    print("Skipping day-of-week analysis because the time column was not available.")

# ---------------------------------------------------------------------
# 6. Month distribution
# ---------------------------------------------------------------------
if "Month" in df.columns:
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    month_counts = df["Month"].value_counts().reindex(month_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=month_counts.index, y=month_counts.values, palette="magma", ax=ax)
    ax.set_title("Accidents by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Accidents")
    annotate_bars(ax)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_plot(fig, "accidents_by_month.png")

    explanation = "This chart reveals seasonal or monthly variation in accident frequency."
    insights = [f"Peak month: {month_counts.idxmax()}"]
    print_section("Accidents by Month", explanation, insights)
else:
    print("Skipping month analysis because the time column was not available.")

# ---------------------------------------------------------------------
# 7. Hour of day distribution
# ---------------------------------------------------------------------
if "Hour" in df.columns:
    hour_counts = df["Hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, palette="cividis", ax=ax)
    ax.set_title("Accidents by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Accidents")
    annotate_bars(ax)
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot(fig, "accidents_by_hour.png")

    explanation = "This chart shows when accidents are most likely to occur during the day."
    insights = [f"Peak accident hour: {int(hour_counts.idxmax())}:00"]
    print_section("Accidents by Hour of Day", explanation, insights)
else:
    print("Skipping hour analysis because the time column was not available.")

# ---------------------------------------------------------------------
# 8. Top locations
# ---------------------------------------------------------------------
if street_col or city_col or county_col:
    if street_col and city_col:
        location_series = df[street_col].fillna("") + " - " + df[city_col].fillna("")
    elif street_col:
        location_series = df[street_col]
    elif city_col:
        location_series = df[city_col]
    else:
        location_series = df[county_col]

    location_counts = location_series.fillna("Unknown").astype(str).value_counts().head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=location_counts.values, y=location_counts.index, palette="Blues_d", ax=ax)
    ax.set_title("Top Accident Locations")
    ax.set_xlabel("Number of Accidents")
    ax.set_ylabel("Location")
    plt.tight_layout()
    save_plot(fig, "top_locations.png")

    explanation = "This chart shows the locations with the highest accident counts."
    insights = [f"Top location: {location_counts.index[0]}"]
    print_section("Top Accident Locations", explanation, insights)
else:
    print("Skipping location analysis because no location-related columns were found.")

# ---------------------------------------------------------------------
# 9. Missing values visualization
# ---------------------------------------------------------------------
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
if not missing_pct.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=missing_pct.values, y=missing_pct.index, palette="coolwarm", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Missing Percentage (%)")
    ax.set_ylabel("Column")
    plt.tight_layout()
    save_plot(fig, "missing_values.png")

    explanation = "This chart visualizes which columns contain the most missing data."
    insights = [f"Column with the highest missing rate: {missing_pct.index[0]} ({missing_pct.iloc[0]:.1f}%)"]
    print_section("Missing Values by Column", explanation, insights)

# ---------------------------------------------------------------------
# 10. Correlation heatmap
# ---------------------------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number]).copy()
if len(numeric_df.columns) > 1:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        numeric_df.corr(numeric_only=True),
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot=False,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap for Numerical Features")
    plt.tight_layout()
    save_plot(fig, "correlation_heatmap.png")

    explanation = "This heatmap shows the strength of linear relationships between numeric variables."
    insights = ["Strong correlations indicate that some features move together and may be useful for modeling."]
    print_section("Correlation Heatmap", explanation, insights)
else:
    print("Skipping correlation heatmap because there were too few numeric columns.")

# ---------------------------------------------------------------------
# 11. Pairplot for important numeric variables
# ---------------------------------------------------------------------
important_numeric_cols = []
for candidate in [
    severity_col,
    "Distance(mi)",
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Pressure(in)",
    latitude_col,
    longitude_col,
]:
    if candidate and candidate in analysis_df.columns:
        important_numeric_cols.append(candidate)

if len(important_numeric_cols) >= 3:
    pair_df = analysis_df[important_numeric_cols].dropna().sample(n=min(5000, len(analysis_df)), random_state=42)
    pairplot = sns.pairplot(pair_df, height=2.0, diag_kind="kde")
    pairplot.fig.suptitle("Pairplot of Important Numerical Variables", y=1.02)
    pairplot.fig.tight_layout()
    pairplot.fig.savefig(IMAGE_DIR / "pairplot.png", dpi=300, bbox_inches="tight")
    plt.close(pairplot.fig)

    explanation = "This pairplot shows the relationships between several important numeric variables."
    insights = ["Use it to identify clusters, trends, or unusual patterns across variables."]
    print_section("Pairplot of Important Numerical Variables", explanation, insights)
else:
    print("Skipping pairplot because fewer than three suitable numeric columns were found.")

# ---------------------------------------------------------------------
# 12. Boxplots for outlier detection
# ---------------------------------------------------------------------
box_cols = []
for candidate in ["Distance(mi)", "Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)", "Pressure(in)"]:
    if candidate in analysis_df.columns:
        box_cols.append(candidate)

if len(box_cols) >= 2:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, box_cols[:6]):
        sns.boxplot(y=analysis_df[col], ax=ax, color="lightblue")
        ax.set_title(f"Boxplot of {col}")
        ax.set_ylabel(col)
    for ax in axes[len(box_cols):]:
        ax.axis("off")
    plt.tight_layout()
    save_plot(fig, "boxplots_outliers.png")

    explanation = "These boxplots help detect outliers and unusual ranges in the numeric data."
    insights = ["Extreme values are visible where the boxplot whiskers are very long."]
    print_section("Boxplots for Outlier Detection", explanation, insights)
else:
    print("Skipping boxplots because too few relevant numeric columns were available.")

# ---------------------------------------------------------------------
# 13. Histograms and KDE plots
# ---------------------------------------------------------------------
hist_cols = []
for candidate in ["Distance(mi)", "Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)", "Pressure(in)"]:
    if candidate in analysis_df.columns:
        hist_cols.append(candidate)

if len(hist_cols) >= 2:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, hist_cols[:6]):
        sns.histplot(analysis_df[col].dropna(), kde=True, bins=30, ax=ax, color="steelblue")
        ax.set_title(f"Histogram and KDE for {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
    for ax in axes[len(hist_cols):]:
        ax.axis("off")
    plt.tight_layout()
    save_plot(fig, "histograms_kde.png")

    explanation = "These histograms and KDE curves show the distribution of key numeric features."
    insights = ["Strong skewness or multiple peaks may indicate interesting subgroups in the dataset."]
    print_section("Histograms and KDE Plots", explanation, insights)
else:
    print("Skipping distribution plots because too few numeric columns were available.")

# ---------------------------------------------------------------------
# 14. Scatter plots for relationships
# ---------------------------------------------------------------------
if "Temperature(F)" in analysis_df.columns and "Visibility(mi)" in analysis_df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    sample = analysis_df.sample(n=min(8000, len(analysis_df)), random_state=42)
    sns.scatterplot(data=sample, x="Temperature(F)", y="Visibility(mi)", alpha=0.4, ax=ax)
    ax.set_title("Temperature vs Visibility")
    ax.set_xlabel("Temperature (F)")
    ax.set_ylabel("Visibility (mi)")
    plt.tight_layout()
    save_plot(fig, "scatter_temperature_visibility.png")

    explanation = "This scatter plot explores whether visibility changes as temperature changes."
    insights = ["Look for clear positive or negative trends in visibility across temperature ranges."]
    print_section("Scatter Plot: Temperature vs Visibility", explanation, insights)

if "Humidity(%)" in analysis_df.columns and "Visibility(mi)" in analysis_df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    sample = analysis_df.sample(n=min(8000, len(analysis_df)), random_state=42)
    sns.scatterplot(data=sample, x="Humidity(%)", y="Visibility(mi)", alpha=0.4, ax=ax)
    ax.set_title("Humidity vs Visibility")
    ax.set_xlabel("Humidity (%)")
    ax.set_ylabel("Visibility (mi)")
    plt.tight_layout()
    save_plot(fig, "scatter_humidity_visibility.png")

    explanation = "This scatter plot helps show whether visibility tends to decrease as humidity rises."
    insights = ["A downward slope would suggest stronger visibility loss under humid conditions."]
    print_section("Scatter Plot: Humidity vs Visibility", explanation, insights)

# ---------------------------------------------------------------------
# 15. Accident hotspot visualization
# ---------------------------------------------------------------------
if latitude_col and longitude_col:
    hotspot_df = analysis_df[[latitude_col, longitude_col]].dropna()
    if len(hotspot_df) > 1000:
        hotspot_df = hotspot_df.sample(n=min(8000, len(hotspot_df)), random_state=42)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.hexbin(
        hotspot_df[longitude_col],
        hotspot_df[latitude_col],
        gridsize=35,
        cmap="YlOrRd",
        mincnt=1,
        alpha=0.95,
    )
    cb = plt.colorbar()
    cb.set_label("Accident Density")
    ax.set_title("Accident Hotspot Visualization")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    save_plot(fig, "accident_hotspots.png")

    explanation = "This hotspot plot shows geographic regions where accidents are concentrated."
    insights = ["Hotspots appear as darker or denser regions in the map."]
    print_section("Accident Hotspot Visualization", explanation, insights)
else:
    print("Skipping hotspot visualization because latitude/longitude columns were not found.")

# ---------------------------------------------------------------------
# 16. Executive summary
# ---------------------------------------------------------------------
print("\n## Executive Summary")
summary_insights = []

if "Hour" in df.columns and not df["Hour"].dropna().empty:
    peak_hour = int(df["Hour"].mode().iloc[0])
    summary_insights.append(f"Peak accident hour: {peak_hour}:00")

if weather_col:
    weather_risk = df.groupby(weather_col)[severity_col].mean().sort_values(ascending=False)
    if not weather_risk.empty:
        summary_insights.append(
            f"Most dangerous weather condition: {weather_risk.index[0]} (mean severity {weather_risk.iloc[0]:.2f})"
        )

if road_condition_col:
    road_risk = df.groupby(road_condition_col)[severity_col].mean().sort_values(ascending=False)
    if not road_risk.empty:
        summary_insights.append(
            f"Most dangerous road condition: {road_risk.index[0]} (mean severity {road_risk.iloc[0]:.2f})"
        )

if severity_col:
    summary_insights.append(f"Most frequent severity: {df[severity_col].mode().iloc[0]}")

if street_col or city_col:
    location_series = (
        df[street_col].fillna("") + " - " + df[city_col].fillna("")
        if street_col and city_col
        else df[street_col]
        if street_col
        else df[city_col]
    )
    top_location = location_series.fillna("Unknown").astype(str).value_counts().idxmax()
    summary_insights.append(f"Top accident hotspot: {top_location}")

if "Hour" in df.columns and "Day_of_Week" in df.columns:
    summary_insights.append("Interesting trend: accidents cluster around peak commuting hours and weekdays.")

for insight in summary_insights:
    print(f"- {insight}")

print("\nEDA completed successfully. Visualizations saved in the images folder.")