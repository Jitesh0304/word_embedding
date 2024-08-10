import pandas as pd

# Load the Excel file
df = pd.read_excel(r"C:\Users\jites\Downloads\OPD Patient Consultation-Un Cleaned Data.xlsx")

delete_item = ["<p>", "</p>", "<ul>", "</ul>", "<li>", "</li>"]

# Function to clean the values
def clean_text(val):
    if isinstance(val, str):  # Ensure the value is a string
        for item in delete_item:
            if item in val:
                val = val.replace("<ul>", "")
                val = val.replace("</ul>", "")
                val = val.replace("<li>", "")
                val = val.replace("</li>", "")
                val = val.replace("<p>", "")
                val = val.replace("</p>", ".")
    return val

# Apply the function to all elements in the dataframe
return_df = df.applymap(clean_text)

# Save the cleaned dataframe to a new Excel file
# return_df.to_excel("abc.xlsx", index=False)


