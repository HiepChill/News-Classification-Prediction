import os
import pandas as pd

# Define the folder structure and category labels
categories = ["business", "entertainment", "politics", "sport", "tech"]
category_labels = {
    "business": 1,
    "entertainment": 2,
    "politics": 3,
    "sport": 4,
    "tech": 5,
}

# List to hold all the data to be written into CSV
data = []

# Base directory containing the 'bbc' folder
base_dir = "./bbc/"  # Replace this with the path to your 'bbc' directory

# Traverse through each category and its text files
for category in categories:
    category_path = os.path.join(base_dir, category)

    # Ensure the category directory exists
    if not os.path.exists(category_path):
        continue

    # Get list of .txt files in the category folder
    for idx, filename in enumerate(os.listdir(category_path)):
        if filename.endswith(".txt"):
            # Create the file ID based on the rule: category label + '0' + filename
            file_id = f"{category_labels[category]}0{filename.split('.')[0]}"

            # Read the content of the text file
            with open(
                os.path.join(category_path, filename), "r", encoding="ISO-8859-1"
            ) as file:
                content = file.read()

            # Append data (id, content, category, label)
            data.append([file_id, content, category, category_labels[category]])

# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=["id", "content", "category", "label"])

# Save the DataFrame to a CSV file
df.to_csv("bbc_data.csv", index=False)

print("Data preprocessing complete. CSV file created.")