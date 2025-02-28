import pandas as pd

# Load dataset with correct encoding
file_path = r"D:\dev\Dataset\IMDb Movies India.csv"  # Update path if needed
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Drop rows where 'Name' is missing
df = df.dropna(subset=['Name'])

# Clean 'Year' column: Extract numeric year and convert to int, fill missing with 0
df['Year'] = df['Year'].str.extract('(\\d{4})')  # Extract only the year part
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

# Convert 'Duration' to numeric (minutes) and fill missing with the median
df['Duration'] = df['Duration'].str.extract('(\\d+)')  # Extract numeric part
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Duration'].fillna(df['Duration'].median(), inplace=True)

# Fill missing categorical values with 'Unknown'
categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# Fill missing ratings with the mean
df['Rating'].fillna(df['Rating'].mean(), inplace=True)

# Convert 'Votes' to numeric, fill missing with 0
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0).astype(int)

# Save cleaned dataset
df.to_csv("Cleaned_IMDb_Movies_India.csv", index=False)

print("Dataset cleaned and saved as 'Cleaned_IMDb_Movies_India.csv'")
