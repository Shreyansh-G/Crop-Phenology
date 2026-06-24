import json
import re
import pandas as pd
from bs4 import BeautifulSoup

def extract_dashboard_data(html_path, output_csv):
    print(f"Reading {html_path}...")
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    extracted_json = None

    for script in soup.find_all("script"):
        if script.string and "var data =" in script.string:
            match = re.search(r'var\s+data\s*=\s*(\{.*?\});\s*$', script.string, re.DOTALL | re.MULTILINE)
            if match:
                raw_json = match.group(1)
                try:
                    extracted_json = json.loads(raw_json)
                    print("Successfully located and parsed the GeoJSON data block!")
                    break
                except json.JSONDecodeError as e:
                    print(f"❌ Found the block but JSON parsing failed: {e}")

    if not extracted_json:
        print("❌ Could not find the 'var data =' block in the HTML.")
        return

    rows = []
    features = extracted_json.get("features", [])
    
    print(f"Processing {len(features)} spatial records...")
    
    for feature in features:
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [])
        
        raw_field_id = props.get("Field_ID", "0")
        clean_plot_id = int(re.sub(r'\D', '', raw_field_id)) if any(c.isdigit() for c in raw_field_id) else raw_field_id

        rows.append({
            "plot_id": clean_plot_id,
            "date": props.get("date"),
            "stage_name": props.get("Stage"),
            "NDVI": props.get("NDVI"),
            "stress": props.get("Stress"),
            "yield_est": props.get("Yield"),
            "confidence": props.get("Confidence"),
            "raw_coordinates": str(coords) 
        })

    df = pd.DataFrame(rows)

    df.to_csv(output_csv, index=False)
    print("-" * 40)
    print(f"Extraction Complete! Saved {len(df)} rows to {output_csv}")
    print(df.head())

extract_dashboard_data("../dump/ADVANCED_WHEAT_GEOAI_DASHBOARD.html", "../dataset/2025_2026_ground_truth.csv")