import pandas as pd
import numpy as np
import os

# 1. Create a DateTime index for year 2025 at 5-minute intervals (no DST adjustment).
date_range = pd.date_range(start="2025-01-01 00:00:00", end="2025-12-31 23:55:00", freq="5T")
# Note: '5T' frequency means 5 minutes.

# 2. Initialize DataFrame with the timestamp index.
df = pd.DataFrame(index=date_range)
df.index.name = "Timestamp"

# 3. Determine day type for each date.
# Prepare holiday list for 2025 (unique dates):
holidays_2025 = pd.to_datetime([
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day (and Inauguration Day, same date)
    "2025-02-17",  # Washington's Birthday (Presidents' Day)
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-10-13",  # Columbus Day
    "2025-11-11",  # Veterans Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas Day
])
holiday_dates = [d.date() for d in holidays_2025]  # convert to python date objects

# Set random seed for reproducibility
np.random.seed(42)

# Randomly assign each holiday as 'party' or 'vacation', with at most 10 parties.
holiday_types = {}
holiday_list = list(holiday_dates)
np.random.shuffle(holiday_list)
party_count = min(10, len(holiday_list))
parties = set(holiday_list[:party_count])
for h in holiday_dates:
    holiday_types[h] = "party" if h in parties else "vacation"

# Now determine work-from-home weekdays.
all_dates = pd.Series(df.index.date)  # Series of all dates corresponding to each timestamp
unique_dates = pd.to_datetime(all_dates.unique())  # This creates a DatetimeIndex
# Create a boolean mask for weekdays (Monday=0 to Friday=4)
weekday_mask = unique_dates.weekday < 5
# Filter out holidays from weekdays.
nonholiday_weekdays = unique_dates[(unique_dates.weekday < 5) & (~pd.Index(unique_dates.date).isin(holiday_dates))]

# Choose ~5-10% of these weekdays as WFH.
total_weekdays = len(nonholiday_weekdays)
min_wfh = int(np.floor(0.05 * total_weekdays))
max_wfh = int(np.floor(0.10 * total_weekdays))
if max_wfh < min_wfh:
    max_wfh = min_wfh
num_wfh = np.random.randint(min_wfh, max_wfh + 1)
wfh_days = set(np.random.choice(nonholiday_weekdays.date, size=num_wfh, replace=False))

# Now assign day type for each unique date.
day_type_map = {}
for d in unique_dates.date:
    if d in holiday_types:
        day_type_map[d] = holiday_types[d]
    elif pd.Timestamp(d).weekday() >= 5:
        day_type_map[d] = "weekend"
    else:
        day_type_map[d] = "work-from-home" if d in wfh_days else "work"

# Map each timestamp's date to the day type.
df["DayType"] = [day_type_map[d] for d in all_dates]

# 4. Simulate outdoor temperature.
# Seasonal component (yearly cosine wave)
day_of_year = (df.index - pd.Timestamp("2025-01-01")).days  # number of days from Jan 1
avg_temp = 15.0    # average temperature in Celsius
season_amp = 15.0  # seasonal amplitude
seasonal = avg_temp - season_amp * np.cos(2 * np.pi * day_of_year / 365)
# Daily component (cosine daily cycle, peak at noon)
hour_fraction = df.index.hour + df.index.minute / 60.0
daily_amp = 5.0
daily = daily_amp * np.cos(2 * np.pi * (hour_fraction - 12) / 24)
# Total outdoor temperature plus some random noise.
outdoor_temp = seasonal + daily + np.random.normal(scale=1.0, size=len(df))
df["OutdoorTemp"] = outdoor_temp.round(2)

# 5. Assign energy cost based on time of day (TOU pricing).
current_hour = df.index.hour
cost = np.select(
    [
        ((current_hour >= 6) & (current_hour < 10)) | ((current_hour >= 17) & (current_hour < 21)),
        (current_hour >= 10) & (current_hour < 17)
    ],
    [0.30, 0.20],
    default=0.10
)
df["EnergyCost"] = cost

# 6. Simulate lights brightness and AC set temperature.
brightness = np.zeros(len(df))
ac_set_temp = np.zeros(len(df))
all_dates_array = df.index.date

# Iterate over each unique date and assign baseline values based on day type.
for d, dtype in day_type_map.items():
    day_idx = (all_dates_array == d)
    hours = df.index[day_idx].hour  # hours for timestamps of day d
    morning = (hours >= 6) & (hours < 9)
    workday = (hours >= 9) & (hours < 17)
    evening = (hours >= 17) & (hours < 23)
    night = ~(morning | workday | evening)
    
    bright_day = np.zeros(np.sum(day_idx))
    ac_day = np.zeros(np.sum(day_idx))
    
    if dtype == "work":
        bright_day[morning] = 5.0
        bright_day[workday] = 0.0
        bright_day[evening] = 7.0
        bright_day[night] = 0.0
        
        day_outtemp = df.loc[day_idx, "OutdoorTemp"].values
        home_set = 22.0
        away_cold = 20.0
        away_hot = 26.0
        ac_day[morning] = home_set
        # Use noon hours as indicator (12 <= hour < 15)
        noon_mask = (df.loc[day_idx].index.hour >= 12) & (df.loc[day_idx].index.hour < 15)
        mid_out = np.mean(day_outtemp[noon_mask]) if np.any(noon_mask) else home_set
        ac_away = away_cold if mid_out < home_set else away_hot if mid_out > home_set else home_set
        ac_day[workday] = ac_away
        ac_day[evening] = home_set
        ac_day[night] = home_set
        
    elif dtype == "work-from-home":
        bright_day[morning] = 5.0
        bright_day[workday] = 3.0
        bright_day[evening] = 7.0
        bright_day[night] = 1.0
        
        home_set = 22.0
        day_outtemp = df.loc[day_idx, "OutdoorTemp"].values
        noon_mask = (df.loc[day_idx].index.hour >= 12) & (df.loc[day_idx].index.hour < 15)
        mid_out = np.mean(day_outtemp[noon_mask]) if np.any(noon_mask) else home_set
        home_set_day = 21.0 if mid_out < home_set - 5 else 24.0 if mid_out > home_set + 5 else home_set
        ac_day[:] = home_set_day
        
    elif dtype == "weekend":
        bright_day[morning] = 2.0
        bright_day[(hours >= 9) & (hours < 12)] = 4.0
        bright_day[(hours >= 12) & (hours < 17)] = 3.0
        bright_day[evening] = 8.0
        bright_day[night] = 1.0
        ac_day[:] = 22.0
        
    elif dtype == "vacation":
        bright_day[:] = 0.0
        bright_day[evening] = 1.0
        day_outtemp = df.loc[day_idx, "OutdoorTemp"].values
        ac_day[:] = 20.0 if np.mean(day_outtemp) < 22.0 else 26.0 if np.mean(day_outtemp) > 22.0 else 22.0
        
    elif dtype == "party":
        bright_day[morning] = 3.0
        bright_day[(hours >= 9) & (hours < 17)] = 4.0
        bright_day[evening] = 10.0
        bright_day[(hours >= 23)] = 9.0
        bright_day[night & (hours < 2)] = 9.0
        ac_day[:] = 22.0
        day_outtemp = df.loc[day_idx, "OutdoorTemp"].values
        if np.max(day_outtemp) > 25.0:
            ac_day[evening] = 20.0
        elif np.min(day_outtemp) < 10.0:
            ac_day[evening] = 24.0
    
    brightness[day_idx] = bright_day
    ac_set_temp[day_idx] = ac_day

# 7. Add Gaussian noise and apply occasional overrides.
rng = np.random.default_rng(seed=42)
brightness += rng.normal(loc=0.0, scale=1.0, size=len(brightness))
ac_set_temp += rng.normal(loc=0.0, scale=0.5, size=len(ac_set_temp))

num_b_override = int(0.002 * len(df))
override_idx = rng.choice(len(df), size=num_b_override, replace=False)
for idx in override_idx:
    brightness[idx] = 10.0 if brightness[idx] < 2 else 0.0

num_ac_override = int(0.001 * len(df))
override_idx = rng.choice(len(df), size=num_ac_override, replace=False)
for idx in override_idx:
    ac_set_temp[idx] = 26.0 if ac_set_temp[idx] < 22.0 else 20.0

brightness = np.clip(brightness, 0.0, 10.0)
ac_set_temp = np.clip(ac_set_temp, 20.0, 26.0)
brightness = np.round(brightness, 1)
ac_set_temp = np.round(ac_set_temp, 1)

df["Brightness"] = brightness
df["AC_Set_Temp"] = ac_set_temp

# 8. Save to CSV files.
os.makedirs("historical_data", exist_ok=True)
lights_df = df[["DayType", "OutdoorTemp", "EnergyCost", "Brightness"]]
ac_df = df[["DayType", "OutdoorTemp", "EnergyCost", "AC_Set_Temp"]]
lights_df.to_csv("historical_data/lights_2025.csv")
ac_df.to_csv("historical_data/ac_2025.csv")
print("CSV files generated: 'lights_2025.csv' and 'ac_2025.csv' in the 'historical_data' directory")
