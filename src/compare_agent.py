import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.environment import Environment
from src.mpc_agent import MPCAgent
from src.tinympc_agent import TinyMPCAgent

# Load dataset
data_file = 'data/pvLoadPriceData6.csv'

# Storage for results
soc_values_cvx = []
soc_values_tiny = []
pbatt_cvx = []
pbatt_tiny = []
pgrid_cvx = []
pgrid_tiny = []
time_steps = []

for building_number in range(1, 2):
    print(f"Running optimization for Building {building_number}...")
    env1 = Environment(data_file, building_number)
    env2 = Environment(data_file, building_number)

    controller_cvx = MPCAgent(env1, gamma=[0.15, 0.85], planning_steps=24)
    controller_tiny = TinyMPCAgent(env2, planning_steps=24)

    max_steps = min(72, env1.total_steps - controller_cvx.planning_steps)

    for step in range(max_steps):
        res_cvx = controller_cvx.predict()
        res_tiny = controller_tiny.predict()

        if res_cvx and res_tiny:
            pgrid_cvx_val, pbatt_cvx_val, ebatt_cvx = res_cvx
            pgrid_tiny_val, pbatt_tiny_val, ebatt_tiny = res_tiny

            soc_values_cvx.append((ebatt_cvx[0] / controller_cvx.batteryEnergy) * 100)
            soc_values_tiny.append((ebatt_tiny / controller_tiny.capacity) * 100)
            pbatt_cvx.append(pbatt_cvx_val[0])
            pbatt_tiny.append(pbatt_tiny_val)
            pgrid_cvx.append(pgrid_cvx_val[0])
            pgrid_tiny.append(pgrid_tiny_val)
            time_steps.append(step)

# # Plot comparisons
# plt.figure(figsize=(10, 4))
# plt.plot(time_steps, soc_values_cvx, label="CVXPY SOC (%)")
# plt.plot(time_steps, soc_values_tiny, label="TinyMPC SOC (%)", linestyle='--')
# plt.title("State of Charge Comparison")
# plt.xlabel("Time Step")
# plt.ylabel("SOC (%)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 4))
# plt.plot(time_steps, pbatt_cvx, label="CVXPY Pbatt (kW)")
# plt.plot(time_steps, pbatt_tiny, label="TinyMPC Pbatt (kW)", linestyle='--')
# plt.title("Battery Power Comparison")
# plt.xlabel("Time Step")
# plt.ylabel("Power (kW)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 4))
# plt.plot(time_steps, pgrid_cvx, label="CVXPY Pgrid (kW)")
# plt.plot(time_steps, pgrid_tiny, label="TinyMPC Pgrid (kW)", linestyle='--')
# plt.title("Grid Power Comparison")
# plt.xlabel("Time Step")
# plt.ylabel("Power (kW)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Print as matrix form
print("\n🔋 SOC Comparison Matrix:")
for i, (s1, s2) in enumerate(zip(soc_values_cvx, soc_values_tiny)):
    print(f"Step {i:2d}: CVXPY={s1:.2f}%  |  TinyMPC={s2:.2f}%")
