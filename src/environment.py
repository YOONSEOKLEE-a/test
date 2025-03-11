import pandas as pd

class Environment:
    def __init__(self, data_file, building_number):
        data = pd.read_csv(data_file)
        self.time = data['time'].values
        self.costData = data['costData'].values
        self.BuildingLoad = data[f'Building {building_number}'].values  # Dynamically select building
        self.Emissions = data['Emissions'].values
        self.maxCost = max(self.costData)
        self.maxEmissions = max(self.Emissions)
        self.current_step = 0
        self.total_steps = len(self.time)  # Dynamically get dataset length