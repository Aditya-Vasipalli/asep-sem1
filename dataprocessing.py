import pandas as pd

# Load the raw sheet
file_path = "data/cleaned_data.xlsx"
raw_df = pd.read_excel(file_path, sheet_name="raw")

# Initialize a new Date column
raw_df["Date"] = None

# Initialize current_date
current_date = None

# Fill the Date column with values from the "Time" column if they represent a date
for index, row in raw_df.iterrows():
    if " - " not in row["Time"]:  # Assuming date rows don't have " - "
        current_date = row["Time"]  # Assign the new date
    else:
        raw_df.at[index, "Date"] = current_date  # Add current_date to the row

# Forward-fill the Date column
raw_df["Date"] = raw_df["Date"].ffill()

# Drop rows where "Time" is just a date
cleaned_df = raw_df[raw_df["Time"].str.contains(" - ")]

# Save to the "clean" sheet
with pd.ExcelWriter(file_path, mode="a", if_sheet_exists="replace") as writer:
    cleaned_df.to_excel(writer, sheet_name="clean", index=False)

print("Date column added and data cleaned!")
