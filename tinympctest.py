import numpy as np
import tinympc

# System parameters
A = np.array([[1.0]])  # SOC(t+1) = SOC(t) + ...
B = np.array([[0.96, -1/0.95]])  # Charge / Discharge efficiency reflected
Q = np.array([[1.0]])  # State cost (not used effectively here)
R = np.diag([0.01, 0.01])  # Control input cost
N = 24  # Prediction horizon

# Initial SOC: 2 batteries of 7.8kWh each, starting at 50%
x0 = np.array([7.8 * 2 * 0.5])

# Initialize TinyMPC
mpc = tinympc.TinyMPC()
mpc.setup(
    A, B, Q, R, N,
    x_min=np.full((1, N), 0.05 * 15.6),    # Minimum SOC constraint (5% of total capacity)
    x_max=np.full((1, N), 15.6),           # Maximum SOC constraint (100% of total capacity)
    u_min=np.full((2, N - 1), 0.0),        # Min charge/discharge power (0 kW)
    u_max=np.full((2, N - 1), 9.6)         # Max charge/discharge power (9.6 kW)
)

soc = [x0[0]]
charge = []
discharge = []

# Receding horizon: solve MPC at each time step
for t in range(N):
    mpc.set_x0(np.array([soc[-1]]))  # Update current SOC as initial state
    sol = mpc.solve()

    current_control = sol["controls"]  # First-step control: shape (2,)
    charge.append(current_control[0])
    discharge.append(current_control[1])

    # Compute next SOC based on current inputs
    next_soc = A @ [soc[-1]] + B @ current_control
    soc.append(next_soc[0])

# Remove the final SOC for alignment with N steps
soc = soc[:-1]

# Print results
print("🔋 SOC (kWh):", soc)
print("⚡ Charge (kW):", charge)
print("⚡ Discharge (kW):", discharge)
