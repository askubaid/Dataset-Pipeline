import os
import json
import pandas as pd

def parse_excel_themes():
    excel_path = "RAWThemes.xlsx"
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return

    # Read the excel sheet (default first sheet is Raw)
    df = pd.read_excel(excel_path)
    
    themes_list = []
    current_category = "General"

    for idx, row in df.iterrows():
        col1 = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        col2 = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        col3 = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""
        col4 = str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else ""
        col5 = str(row.iloc[5]).strip() if pd.notna(row.iloc[5]) else ""

        # Check if this row defines a Category
        if "Category:" in col1:
            current_category = col1.replace("Category:", "").strip()
            continue

        # Skip headers
        if col1 == "#" or col2 == "Themes" or not col1:
            continue

        # Try to parse index as number to verify it's a theme row
        try:
            theme_id = int(float(col1))
            theme_name = col2
            nc_level = col3
            
            # Format token count nicely
            try:
                tokens = int(float(col4))
            except ValueError:
                tokens = col4

            try:
                sp_value = int(float(col5))
            except ValueError:
                sp_value = col5

            themes_list.append({
                "id": theme_id,
                "category": current_category,
                "name": theme_name,
                "nc": nc_level,
                "tokens": tokens,
                "sp": sp_value
            })
        except ValueError:
            # Not a theme row
            continue

    # Ensure output directory exists
    # os.makedirs(os.path.join("src", "data"), exist_ok=True)
    output_path = os.path.join("themes.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(themes_list, f, indent=2, ensure_ascii=False)

    print(f"Success! Parsed {len(themes_list)} themes and saved them to {output_path}")

if __name__ == "__main__":
    parse_excel_themes()
