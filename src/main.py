#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.environment import Environment
from src.mpc_agent import MPCAgent
from src.utils import plot_soc, plot_mode, plot_energy_data, plot_results

data_file = r'C:\Asmaa\MINES\BESS Project\EMS_Script_New\python\Git-Hub\MPC-Controller\data\pvLoadPriceData6.csv'
results = []

# Storage for plotting
grid_energy = []
battery_power = []
load_power = []
electricity_price = []
soc_values = []
time_steps = []

for building_number in range(1, 2):  # Iterate through buildings
    print(f"Running optimization for Building {building_number}...")
    env = Environment(data_file, building_number)
    controller = MPCAgent(env, gamma=[0.15, 0.85], planning_steps=24)

    # Storage for results
    total_price_without_battery = 0
    total_price_with_battery = 0
    total_emissions_without_battery = 0
    total_emissions_with_battery = 0
    total_grid_energy_without_battery = 0
    total_grid_energy_with_battery = 0

    # ✅ Run the simulation loop **only for the first 72 hours**
    max_simulation_steps = min(72, env.total_steps - controller.planning_steps)
   # max_simulation_steps = env.total_steps - controller.planning_steps

    # Run the simulation loop
    for step in range(max_simulation_steps):
        results_opt = controller.predict()
        if results_opt:
            Pgrid_total, Pbatt_total, Ebatt = results_opt
            # Store values for plotting
            grid_energy.append(Pgrid_total[0])  
            battery_power.append(Pbatt_total[0])  
            load_power.append(env.BuildingLoad[step])  
            electricity_price.append(env.costData[step])  
            soc_values.append(Ebatt[0])  
            time_steps.append(step)            

            # Calculate prices and emissions without battery
            actual_grid_price = env.costData[step] * env.BuildingLoad[step]
            actual_emissions = env.Emissions[step] * env.BuildingLoad[step]
            total_price_without_battery += actual_grid_price
            total_emissions_without_battery += actual_emissions
            total_grid_energy_without_battery += env.BuildingLoad[step] * controller.dt

            # Calculate prices and emissions with battery
            optimized_grid_power = Pgrid_total[0]
            optimized_grid_price = env.costData[step] * optimized_grid_power
            optimized_emissions = env.Emissions[step] * optimized_grid_power

            total_price_with_battery += optimized_grid_price
            total_emissions_with_battery += optimized_emissions
            total_grid_energy_with_battery += optimized_grid_power * controller.dt

    # Calculate Total Savings
    price_saving = total_price_without_battery - total_price_with_battery
    emissions_saving_tons = (total_emissions_without_battery - total_emissions_with_battery)
    grid_energy_saving = total_grid_energy_without_battery - total_grid_energy_with_battery

    # Store results
    results.append([
        building_number,
        total_price_without_battery, total_price_with_battery, price_saving,
        total_emissions_without_battery, total_emissions_with_battery, emissions_saving_tons,
        total_grid_energy_without_battery, total_grid_energy_with_battery, grid_energy_saving
    ])



# Convert SOC to percentage
if soc_values:  # Avoid empty lists
    soc_percentage = np.clip([(soc / (controller.batteryEnergy)) * 100 for soc in soc_values], 0, 100)
else:
    soc_percentage = []

# Now call plotting functions
plot_soc(soc_percentage)
plot_energy_data(time_steps, grid_energy,soc_percentage, battery_power, load_power, electricity_price)



# Save results to CSV
results_df = pd.DataFrame(results, columns=[
    "Building", "Total Price Without Battery", "Total Price With Battery", "Price Saving",
    "Total Emissions Without Battery", "Total Emissions With Battery", "Emissions Saving (kg)",
    "Total Grid Energy Without Battery", "Total Grid Energy With Battery", "Grid Energy Saving"
])
results_df.to_csv('results/summary_results.csv', index=False)
print("Summary results saved to results/summary_results.csv")

# Display the full results for all buildings
from IPython.display import display
display(results_df)

# %%
