import pandas as pd
import os
from pathlib import Path


def create_dummy_data(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = []
    for i in range(1000):
        humidity_3pm = 40.0 + (i % 50)
        rain_tomorrow = "Yes" if humidity_3pm > 70 else "No"
        rain_today = "Yes" if i % 5 == 0 else "No"

        row = {
            "Date": f"2020-01-{(i % 30) + 1:02d}",
            "Location": "Albury",
            "MinTemp": 10.0 + (i % 20),
            "MaxTemp": 20.0 + (i % 20),
            "Rainfall": 0.0 if rain_today == "No" else 10.0,
            "Evaporation": 5.0,
            "Sunshine": 10.0,
            "WindGustDir": "W",
            "WindGustSpeed": 40.0,
            "WindDir9am": "W",
            "WindDir3pm": "W",
            "WindSpeed9am": 20.0,
            "WindSpeed3pm": 20.0,
            "Humidity9am": humidity_3pm,
            "Humidity3pm": humidity_3pm,
            "Pressure9am": 1010.0,
            "Pressure3pm": 1010.0,
            "Cloud9am": 5.0,
            "Cloud3pm": 5.0,
            "Temp9am": 15.0,
            "Temp3pm": 15.0,
            "RainToday": rain_today,
            "RainTomorrow": rain_tomorrow,
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dummy data created at {output_path}")


if __name__ == "__main__":
    output_file = Path("data/raw/weatherAUS.csv")
    if not output_file.exists():
        create_dummy_data(output_file)
    else:
        print(f"Data file already exists at {output_file}")
