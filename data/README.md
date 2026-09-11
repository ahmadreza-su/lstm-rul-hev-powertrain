# Dataset

The pipeline expects the dataset file:

```
EV_Predictive_Maintenance_Dataset_15min.csv
```

to be placed in this `data/` directory (i.e., at `data/EV_Predictive_Maintenance_Dataset_15min.csv`
relative to the repository root).

## Where to obtain the dataset

The dataset is publicly available on Kaggle:

> **EV IoT Predictive Maintenance Dataset**
> https://www.kaggle.com/datasets/datasetengineer/eviot-predictivemaint-dataset

Download the CSV file from the Kaggle page above and place it in this `data/`
directory before running the pipeline. A free Kaggle account is required to
download the file.

## Dataset description

- **Format:** CSV
- **Size:** 175,393 records × 30 columns
- **Sampling:** every 15 minutes
- **Time span:** January 2020 – January 2025 (5 years)
- **Source:** hybrid electric vehicle (HEV) operational logs (publicly released
  on Kaggle)

The 23 input sensor channels used by the pipeline are grouped into five
powertrain sub-systems:

| Sub-system          | Channels                                                                 |
|---------------------|--------------------------------------------------------------------------|
| Battery             | SoC, Battery_Voltage, Battery_Current, Battery_Temperature, Charge_Cycles |
| Electric motor      | Motor_Temperature, Motor_Vibration, Motor_Torque, Motor_RPM, Power_Consumption |
| Brake               | Brake_Pad_Wear, Brake_Pressure, Reg_Brake_Efficiency                     |
| Tire & suspension   | Tire_Pressure, Tire_Temperature, Suspension_Load                         |
| Operating & ambient | Ambient_Temperature, Ambient_Humidity, Load_Weight, Driving_Speed, Distance_Traveled, Idle_Time, Route_Roughness |

The dataset is not redistributed in this repository due to its size (~89 MB).
Please download it directly from Kaggle using the link above.
