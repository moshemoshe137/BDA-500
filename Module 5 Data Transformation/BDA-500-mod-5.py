# %%
import numpy as np
import pandas as pd

# %%
# Read the COVID Testing Data Excel file.
df = pd.read_excel(
    "BDA 500 - COVID Testing Data.xlsx", sheet_name="COVID_Testing_Date"  # sic
)

# %% [markdown]
# # Processing from Module 4
#

# %%
# Clean the "Temperature" column.
valid_temps = df.eval("90 < Temperature < 110 or Temperature.isna()")
invalid_temps = ~valid_temps

df.loc[invalid_temps, "Temperature"] = np.nan

# %%
# Clean the "Age" column.
df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")

valid_ages = df.eval("0 < Age < 120 or Age.isna()")
invalid_ages = ~valid_ages

df.loc[invalid_ages, "Age"] = np.nan

# %%
# Clean the "Test_Administrator" column.
df.loc[df["Test_Administrator"] == "ID-", "Test_Administrator"] = np.nan

# %%
# Recalculate the "Age_60_And_Above" column for accuracy
df["Age_60_And_Above"] = (df["Age"] >= 60).fillna(False)

# %% [markdown]
# # 1. Bin the Age variable into a new variable using your chosen bins, and describe your process.
#

# %%
# Cut the "Age" column into 4 quantiles (quartiles). Use `precision=0` to force rounding
# the bins to a whole number.
pd.qcut(df["Age"], 4, precision=0).value_counts(dropna=False).sort_index()

# %%
# Add an "age_quartile" column to our DataFrame. Instead of putting the range of ages,
# we'll use the labels `["Q1", "Q2", "Q3", "Q4"]`.
df["age_quartile"] = pd.qcut(df["Age"], 4, precision=0, labels=["Q1", "Q2", "Q3", "Q4"])

# View a random sample of the table to ensure accuracy.
df[["Age", "age_quartile"]].sample(10)

# %% [markdown]
# # Transform the Date variable into a new variable with "Weekday" or "Weekend" bins and describe your process.
#

# %%
# Create a helper column called "DOW" that uses `strftime` to convert each datetime
# to a string. The code `"%a"` converts each date to its abbreviated name of the day of
# the week (i.e. Sun, Mon, ..., Sat). (Locale dependent!!)
DOW = df["Date"].dt.strftime("%a").rename("DOW")
DOW

# %%
# Pass two simple regular expressions to `.replace`. `"Sat|Sun"` matches either "Sat" or
# "Sun". The regex `"Mon|Tue|Wed|Thu|Fri"` works similarly.
df["weekend_weekday"] = DOW.replace(
    regex={"Sat|Sun": "Weekend", "Mon|Tue|Wed|Thu|Fri": "Weekday"}
)

# View our date column and our new weekend_weekday column together. We'll also `concat`
# our helper column, `DOW`, to help confirm accuracy.
pd.concat([df[["Date", "weekend_weekday"]], DOW], axis="columns")

# %%
# Before we start plotting, let's separate out the Boolean columns that won't look super
# nice with traditional numeric plotting.
bool_cols = [
    "Cough",
    "Fever",
    "Sore_Throat",
    "Shortness_Of_Breath",
    "Headache",
    "Age_60_And_Above",
    "Contact",
]

# The `bool_cols`, plus a few others, make up all of our categories.
categorical_cols = bool_cols + [
    "Sex",
    "Result",
    "Test_Administrator",
    "Patient_Experience_Survey",
    "Test_Type",
    "age_quartile",
    "weekend_weekday",
]

# Create a subset called `columns_good_for_plotting` that has no Boolean columns.
columns_good_for_plotting = df.drop(columns=categorical_cols)

# %%
# Import plotting libraries `matplotlib` and `seaborn`.
import matplotlib.pyplot as plt
import seaborn as sns

# Make a pair-plot for each categorical variable in the list we created above.
for category in categorical_cols:
    print(category)

    # Decide on the Palette
    # ---------------------
    # This is just a few rules to give us a nice palette for certain plots instead of
    # relying on the default colormap. It colors 0/False/"negative" as green for "good"
    # outcomes, while 1/True/"positive" are colored red for "bad".
    # Additionally, it uses the traditional blue/pink colorscheme for male and female
    # respectively.
    if set(df[category].dropna()) == {0, 1}:
        palette = {0: "green", 1: "red"}
    elif set(df[category].dropna()) == {True, False}:
        palette = {False: "green", True: "red"}
    elif set(df[category].dropna()) == {"positive", "negative"}:
        palette = {"negative": "green", "positive": "red"}
    elif set(df[category].dropna()) == {"male", "female"}:
        palette = {"male": "blue", "female": "pink"}
    else:
        # If it's none of those situations, just let `seaborn` use whatever color
        # palette it wants.
        palette = None

    # Make sure our categorical column is part of our subset.
    cols_to_plot = columns_good_for_plotting.columns.union([category])
    pairplot = sns.pairplot(df[cols_to_plot], hue=category, palette=palette)

    # Give each plot a title
    pairplot.figure.suptitle(
        f"COVID 19 Data by {category.replace('_', ' ').title()}", y=1.015
    )
    # and show the plot.
    plt.show()

    print("\n\n")

# %%
for category in categorical_cols:
    if category == "Result":
        # Can't graph result and result. Just skip this category.
        continue

    ax = (
        df[["Result", category]]
        .value_counts()
        .sort_index()
        .unstack()
        .plot.bar(stacked=True)
    )

    # Add labels to the bars so we can actually see what's going on.
    for bar in ax.containers:
        label_style = {
            "fc": "lightgrey",
            "edgecolor": "none",
            "pad": 0.1,
            "boxstyle": "round",
        }
        ax.bar_label(bar, label_type="center", bbox=label_style)

    # Add a title.
    ax.set_title(f"COVID 19 Data: {category} by test result")
