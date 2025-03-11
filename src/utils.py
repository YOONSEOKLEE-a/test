
import matplotlib.pyplot as plt
import pandas as pd

def plot_soc(soc_history):
    """Plots the State of Charge (SOC) over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(soc_history, label='SOC', color='orange')
    plt.title('Battery State of Charge (SOC)')
    plt.xlabel('Time (hours)')
    plt.ylabel('SOC (%)')
    plt.grid()
    plt.legend()
    plt.show()

def plot_energy_data(time_steps, grid_energy,soc_percentage, battery_power, load_power, electricity_price):
    """Plots energy usage data."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 16), sharex=True)

    axs[0].plot(time_steps, electricity_price, label="Electricity Price ($/kWh)", color='purple')
    axs[0].set_ylabel("Price ($/kWh)")
    axs[0].legend()
    axs[0].grid(True)

    # 🔹 **Second subplot: Battery SOC Percentage**
    axs[1].plot(time_steps, soc_percentage, label="Battery SOC (%)", color='orange')
    axs[1].set_ylabel("Battery SOC (%)")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time_steps, grid_energy, label="Grid Power (kW)", color='blue')
    axs[2].plot(time_steps, battery_power, label="Battery Power (kW)", color='red')
    axs[2].plot(time_steps, load_power, label="Load Power (kW)", color='green', linestyle="dashed")
    axs[2].set_xlabel("Time Steps (Hours)")
    axs[2].set_ylabel("Power (kW)")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def plot_results(results_df):
    """Plots MPC optimization results from summary dataframe."""
    results_df.plot(kind='bar', figsize=(12, 6))
    plt.title('MPC Optimization Results')
    plt.ylabel('Values')
    plt.show()

