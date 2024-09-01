import pandas as pd

# Read the Excel file
excel_file = 'Downloads\\indian_products.xlsx'
df = pd.read_excel(excel_file)

# Convert to JSON
json_data = df.to_json(orient='records', indent=4)

# Write to a JSON file
json_file = 'Downloads\\output_file.json'
with open(json_file, 'w') as file:
    file.write(json_data)

print(f"Excel file converted to JSON and saved as {json_file}")
