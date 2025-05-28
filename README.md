The framework requires two types of data:
1. Visual Data: Daily images of shrimp in ponds
2. Sensor Data: Hourly readings of water quality parameters (DO, Temperature, pH, TAN)

Organize your data as follows:
data/
├── visual/
│   ├── pond1/
│   │   ├── day1/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── day2/
│   │   └── ...
│   └── ...
├── sensor/
│   ├── pond1.csv
│   ├── pond2.csv
│   └── ...
└── biomass.csv