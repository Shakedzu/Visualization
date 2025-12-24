import pandas as pd
from tkinter import Tk
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tkinter.filedialog import askopenfilename, asksaveasfilename
import webbrowser
from pathlib import Path




##Read file

# Tk().withdraw()
#
# file_path = askopenfilename(
#     title="choose an Excel file",
#     filetypes=[("Excel files", "*.xlsx *.xls")]
# )
#
# if not file_path:
#     raise SystemExit("No file selected.")
#
# print("\nThe selected file:")
# print(file_path)
#
# df = pd.read_excel(file_path)
#



###Data preparation

# missing_strings = ["Unknown", "unknown", "Missing Data", "missing data", "N/A", "NA"]
#
# missing_mask = df.isna() | df.apply(lambda col: col.astype(str).str.strip().isin(missing_strings))
#
# missing_counts = missing_mask.sum()
#
# print("\n Num of missing values:")
# print(missing_counts)
#
# rows_with_missing = missing_mask.any(axis=1).sum()
# print(f"\nNumber of records with at least one missing value: {rows_with_missing}")
#
# df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#
# df['Day_num'] = df['Date'].dt.day
# df['Month'] = df['Date'].dt.month
# df['Year'] = df['Date'].dt.year
#
#
# invalid_month = df[(df['Month'] < 1) | (df['Month'] > 12)]
# print("\n invalid month :", len(invalid_month))
# if not invalid_month.empty:
#     print(invalid_month[['Date', 'Month']].head())
#
# invalid_day = df[(df['Day_num'] < 1) | (df['Day_num'] > 31)]
# print("\n invalid day :", len(invalid_day))
# if not invalid_day.empty:
#     print(invalid_day[['Date', 'Day_num']].head())
#

# numeric_cols = df.select_dtypes(include=['number']).columns
# negative_rows = df[(df[numeric_cols] < 0).any(axis=1)]
#
# print("\n records with negative values:", len(negative_rows))
# if not negative_rows.empty:
#     print(negative_rows[numeric_cols].head())
#
# zero_vehicles = df[df['Number_of_Vehicles'] == 0]
# print("\n records with Number_of_Vehicles = 0 (Number_of_Vehicles == 0):", len(zero_vehicles))
# if not zero_vehicles.empty:
#     print(zero_vehicles[['Accident_Index', 'Number_of_Vehicles', 'Number_of_Casualties']].head())
#
#
# invalid_speed = df[(df['Speed_limit'] <= 0) | (df['Speed_limit'] > 80)]
# print("\n invalid Speed_limit :", len(invalid_speed))
#
#
# if not invalid_speed.empty:
#     df = df.drop(invalid_speed.index)
#
#     df['Time'] = df['Time'].astype(str).str.strip()
#
#     time_parsed = pd.to_datetime(df['Time'], format="%H:%M:%S", errors='coerce')
#
#
#     def categorize_time_from_dt(t):
#         if pd.isna(t):
#             return "Unknown"
#         hour = t.hour
#         if 4 <= hour < 10:
#             return "Morning"
#         if 10 <= hour < 16:
#             return "Noon"
#         if 16 <= hour < 22:
#             return "Evening"
#         return "Night"  # 22:00–03:59
#
#     df['Time_Category'] = time_parsed.apply(categorize_time_from_dt)
#
#     print(df[['Time', 'Time_Category']].head())
#

###save updated data
#     print("\n save updated data")
#
#     save_path = asksaveasfilename(
#         title="choose location to save the updated data",
#         defaultextension=".xlsx",
#         filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
#     )

#     if not save_path:
#         print("save cancelled.")
#     else:
#         if save_path.endswith(".csv"):
#             df.to_csv(save_path, index=False)
#         else:
#             df.to_excel(save_path, index=False)
#
#         print(f"\n save success:\n{save_path}")



#visualization_1

Tk().withdraw()
file_path = askopenfilename(
    title=" Excel/CSV",
    filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")]
)
if not file_path:
    raise SystemExit("No file selected.")

df = pd.read_csv(file_path) if file_path.lower().endswith(".csv") else pd.read_excel(file_path)


required_cols = ["Time_Category", "Severity"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required_cols].dropna().copy()
df["Time_Category"] = df["Time_Category"].astype(str).str.strip()
df["Severity"] = df["Severity"].astype(str).str.strip()


time_order = ["Morning", "Noon", "Evening", "Night"]

grouped = (
    df.groupby(["Time_Category", "Severity"])
      .size()
      .reset_index(name="Accident_Count")
)

pivot = (
    grouped.pivot(index="Time_Category", columns="Severity", values="Accident_Count")
           .reindex(time_order)
           .fillna(0)
           .astype(int)
)

severities = pivot.columns.tolist()


fig = go.Figure()

for sev in severities:
    y = pivot[sev].values
    fig.add_trace(go.Bar(
        x=pivot.index.tolist(),
        y=y,
        name=sev,
        text=y,
        textposition="outside",
        hovertemplate=(
            "Time: %{x}<br>"
            f"Severity: {sev}<br>"
            "Number of accidents: %{y}<extra></extra>"
        )
    ))


fig.update_layout(
    template="plotly_white",
    title=dict(
        text="Bicycle accidents by time of day and injury severity",
        x=0.5,
        xanchor="center"
    ),
    xaxis_title="Time of day (Time_Category)",
    yaxis_title="Number of accidents",
    barmode="group",

    legend=dict(
        x=0.98,
        y=0.98,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.65)",
        bordercolor="rgba(180,180,180,0.6)",
        borderwidth=1
    )
)

fig.show()





##visualization_2
Tk().withdraw()
file_path = askopenfilename(
    title=" Excel/CSV",
    filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")]
)
if not file_path:
    raise SystemExit("No file selected.")


df = pd.read_csv(file_path) if file_path.lower().endswith(".csv") else pd.read_excel(file_path)


required_cols = ["Date", "Time", "Day"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Required columns are missing in the file. {missing}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).copy()

df["Time"] = df["Time"].astype(str).str.strip()
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
df = df.dropna(subset=["Hour"]).copy()
df["Hour"] = df["Hour"].astype(int)

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["Day"] = df["Day"].astype(str).str.strip()
df = df[df["Day"].isin(day_order)].copy()

df["Weekend"] = df["Day"].isin(["Saturday", "Sunday"])


daily = (
    df.groupby(["Weekend", df["Date"].dt.date, "Hour"])
      .size()
      .reset_index(name="Count")
)

mean_per_hour = (
    daily.groupby(["Weekend", "Hour"])["Count"]
         .mean()
         .reset_index()
)

weekday = mean_per_hour[mean_per_hour["Weekend"] == False]
weekend = mean_per_hour[mean_per_hour["Weekend"] == True]


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=weekday["Hour"],
    y=weekday["Count"],
    mode="lines+markers",
    name="Weekday (Monday–Friday)",
    line=dict(color="rgba(0,120,255,0.85)", width=2),
    marker=dict(
        symbol="triangle-up",
        size=9,
        color="rgba(0,120,255,0.85)"
    ),
    hovertemplate="Type=Weekday<br>Hour=%{x:02d}:00<br>Avg Accidents=%{y:.2f}<extra></extra>",
    hoverlabel=dict(
        bgcolor="rgba(235,245,255,0.95)",
        bordercolor="rgba(160,190,220,0.9)",
        font=dict(color="rgba(25,25,25,1)", size=13)
    )
))


fig.add_trace(go.Scatter(
    x=weekend["Hour"],
    y=weekend["Count"],
    mode="lines+markers",
    name="Weekend (Saturday–Sunday)",
    line=dict(color="rgba(255,90,90,0.85)", width=2),
    marker=dict(
        symbol="circle",
        size=7,
        color="rgba(255,90,90,0.85)"
    ),
    hovertemplate="Type=Weekend<br>Hour=%{x:02d}:00<br>Avg Accidents=%{y:.2f}<extra></extra>",
    hoverlabel=dict(
        bgcolor="rgba(255,235,235,0.95)",
        bordercolor="rgba(220,160,160,0.9)",
        font=dict(color="rgba(25,25,25,1)", size=13)
    )
))

fig.update_layout(
    template="plotly_white",
    title=dict(
        text="Average Number of Accidents per Hour — Weekend vs Weekday",
        x=0.5,
        xanchor="center"
    ),
    xaxis_title="Hour of the Day",
    yaxis_title="Average Number of Accidents",
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(24)),
        ticktext=[f"{h:02d}:00" for h in range(24)]
    ),
    hovermode="closest",
    legend=dict(
        x=0.98,
        y=0.98,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.65)",
        bordercolor="rgba(180,180,180,0.6)",
        borderwidth=1
    ),
    annotations=[
        dict(
            x=0.5,
            y=-0.22,
            xref="paper",
            yref="paper",
            text="Weekday = Monday–Friday | Weekend = Saturday–Sunday",
            showarrow=False,
            font=dict(size=12, color="gray"),
            align="center"
        )
    ]
)

fig.show()


####visualization_3- power bi



##visualization_4

Tk().withdraw()
file_path = askopenfilename(
    title=" Excel/CSV",
    filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")]
)
if not file_path:
    raise SystemExit("No file selected.")

df = pd.read_csv(file_path) if file_path.lower().endswith(".csv") else pd.read_excel(file_path)


required_cols = ["Speed_limit", "Severity"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df_box = df[required_cols].dropna().copy()
df_box["Severity"] = df_box["Severity"].astype(str).str.strip()
df_box["Speed_limit"] = pd.to_numeric(df_box["Speed_limit"], errors="coerce")
df_box = df_box.dropna(subset=["Speed_limit"])


severity_order_base = ["Slight", "Serious", "Fatal"]
severity_order = [s for s in severity_order_base if s in df_box["Severity"].unique()]
if not severity_order:
    raise ValueError("No matching Severity values found (expected: Slight/Serious/Fatal).")


fig = go.Figure()

x_positions = list(range(len(severity_order)))


medians = []

for i, sev in enumerate(severity_order):
    vals = df_box.loc[df_box["Severity"] == sev, "Speed_limit"].values
    if vals.size == 0:
        medians.append(np.nan)
        continue

    medians.append(float(np.median(vals)))

    fig.add_trace(go.Box(
        y=vals,
        x=[i] * len(vals),
        name=sev,
        fillcolor="rgba(0,0,0,0)",
        line=dict(color="black", width=1.5),
        boxpoints="outliers",
        marker=dict(color="black", size=6, line=dict(color="black", width=1)),


        showlegend=False
    ))


half_width = 0.22

for i, m in enumerate(medians):
    if np.isnan(m):
        continue
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=i - half_width,
        x1=i + half_width,
        y0=m,
        y1=m,
        line=dict(color="orange", width=3)
    )

fig.update_layout(
    template="plotly_white",
    title=dict(text="Speed Limit Distribution by Injury Severity", x=0.5, xanchor="center"),
    xaxis=dict(
        title="Injury Severity",
        tickmode="array",
        tickvals=x_positions,
        ticktext=severity_order
    ),
    yaxis_title="Speed Limit (mph)"
)

fig.show()








####visualization_5- power bi





####visualization_6

Tk().withdraw()
file_path = askopenfilename(
    title=" Excel/CSV",
    filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")]
)
if not file_path:
    raise SystemExit("No file selected.")


df = pd.read_csv(file_path) if file_path.lower().endswith(".csv") else pd.read_excel(file_path)


required_cols = ["Road_type", "Time_Category", "Severity", "Number_of_Casualties"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Required columns are missing in the file. {missing}")

for c in ["Road_type", "Time_Category", "Severity"]:
    df[c] = df[c].astype(str).str.strip()

df["Number_of_Casualties"] = pd.to_numeric(df["Number_of_Casualties"], errors="coerce")

ignore_vals = {"Unknown", "Missing Data", "nan", "None", ""}
df = df[
    (~df["Road_type"].isin(ignore_vals)) &
    (~df["Time_Category"].isin(ignore_vals)) &
    (~df["Severity"].isin(ignore_vals)) &
    (df["Number_of_Casualties"].notna()) &
    (df["Number_of_Casualties"] > 0)
].copy()

if df.empty:
    raise ValueError("No data left after filtering. Check values/columns in the file.")


time_order_base = ["Morning", "Noon", "Evening", "Night"]
uniq_times = df["Time_Category"].unique().tolist()
time_order = [t for t in time_order_base if t in uniq_times] + [t for t in uniq_times if t not in time_order_base]


total = (
    df.groupby(["Road_type", "Time_Category"])["Number_of_Casualties"]
      .sum()
      .unstack(fill_value=0)
)


road_order = total.sum(axis=1).sort_values(ascending=False).index.tolist()

total = total.reindex(index=road_order, columns=time_order).fillna(0)

den = total.replace(0, np.nan)


severities = ["Slight", "Serious", "Fatal"]

count_tables = {}
pct_tables = {}

for sev in severities:
    ctab = (
        df[df["Severity"] == sev]
        .groupby(["Road_type", "Time_Category"])["Number_of_Casualties"]
        .sum()
        .unstack(fill_value=0)
    ).reindex(index=road_order, columns=time_order).fillna(0)

    count_tables[sev] = ctab
    pct_tables[sev] = (ctab / den) * 100


colorscale = [
    [0.00, "rgb(255,255,255)"],
    [0.20, "rgb(220,235,255)"],
    [0.40, "rgb(180,210,245)"],
    [0.60, "rgb(120,170,230)"],
    [0.80, "rgb(60,120,210)"],
    [1.00, "rgb(0,45,120)"],
]

hoverlabel_style = dict(
    bgcolor="rgba(235,245,255,0.95)",
    bordercolor="rgba(160,190,220,0.9)",
    font=dict(color="rgba(25,25,25,1)", size=13)
)

def make_title(sev: str) -> str:
    return f"Percentage of {sev}/Total casualties by Road Type and Time Category"

def make_cbar_title(sev: str) -> str:
    return f"% <b>{sev}</b>/Total casualties"


fig = go.Figure()

for sev in severities:
    z = pct_tables[sev].values


    custom = np.dstack([count_tables[sev].values, total.values])

    fig.add_trace(go.Heatmap(
        z=z,
        x=time_order,
        y=road_order,
        colorscale=colorscale,
        zsmooth=False,
        xgap=1,
        ygap=1,
        colorbar=dict(title=make_cbar_title(sev)),
        customdata=custom,
        hovertemplate=(
            "Road type: %{y}<br>"
            "Time: %{x}<br>"
            f"Severity: {sev}<br>"
            "% of total casualties: %{z:.2f}%<br>"
            "Casualties (severity): %{customdata[0]:.0f}<br>"
            "Total casualties: %{customdata[1]:.0f}"
            "<extra></extra>"
        ),
        hoverlabel=hoverlabel_style,
        visible=(sev == "Serious")
    ))


buttons = []
for idx, sev in enumerate(severities):
    visible = [False] * len(severities)
    visible[idx] = True

    buttons.append(dict(
        label=f"{sev}/Total",
        method="update",
        args=[
            {"visible": visible},
            {"title": {"text": make_title(sev), "x": 0.5, "xanchor": "center"}}
        ]
    ))

fig.update_layout(
    template="plotly_white",
    title=dict(text=make_title("Serious"), x=0.5, xanchor="center"),
    xaxis_title="Time Category",
    yaxis_title="Road Type",
    updatemenus=[dict(
        type="dropdown",
        x=0.0,
        y=1.15,
        xanchor="left",
        yanchor="top",
        buttons=buttons
    )],
)

fig.show()

