# Imports
import pandas as pd

# Load the data
data = pd.read_csv("data/titanic.csv")
# Verify the data is loaded correctly
print(data.head())

# Trim data to only include relevant columns
columnsToKeep = ["Survived", "Pclass", "Sex", "Age"]
trimmedData = data[columnsToKeep]

print(trimmedData.head())

trimmedData.to_csv("data/trimmedTitanic.csv", index=False)

