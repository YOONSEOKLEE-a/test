# main.py  ── CVX vs TinyMPC (72 h) 1-to-1 comparison
# ───────────────────────────────────────────────────────────
import os, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from environment   import Environment
from mpc_agent     import MPCAgent
from tinympc_agent import TinyMPCAgent
# If SOC / power timeline plots in utils are unnecessary, feel free to remove
# from utils         import plot_soc, plot_energy_data

# -----------------------------------------------------------
DATA_FILE   = r'data/pvLoadPriceData6.csv'
BUILD_RANGE = range(1, 2)          # {1}
STEPS       = 72                  # 72-hour simulation

summary_rows = []                 # Final summary rows for CSV

# ===========================================================
# ❶  Simulation loop
# ===========================================================
for bldg in BUILD_RANGE:
    print(f"\n▶ Building {bldg}")

    # ── Independent environment for each controller ----------------------------
    env_grid  = Environment(DATA_FILE, bldg)     # Baseline without battery
    env_cvx   = Environment(DATA_FILE, bldg)     # CVX-based MPC
    env_tiny  = Environment(DATA_FILE, bldg)     # TinyMPC (converted CVXPY)

    cvx_ctrl  = MPCAgent(env_cvx,  gamma=[0.15, 0.85], planning_steps=24)
    tiny_ctrl = TinyMPCAgent(env_tiny,              planning_steps=24)

    # Initialize dictionaries to store cumulative results ----------------------
    agg = {
        'CVX' : dict.fromkeys(
            ['price_no_batt', 'price_batt',
             'emiss_no_batt', 'emiss_batt',
             'grid_no_batt',  'grid_batt'], 0.0),
        'Tiny': dict.fromkeys(
            ['price_no_batt', 'price_batt',
             'emiss_no_batt', 'emiss_batt',
             'grid_no_batt',  'grid_batt'], 0.0),
    }

    # ── Step-by-step simulation (72 h) ----------------------------------------
    for k in range(STEPS):
        load_kW   = env_grid.BuildingLoad[k]
        price     = env_grid.costData[k]
        emis_rate = env_grid.Emissions[k]         # kg / kWh
        dt = 1.0                                  # Time interval (h)

        # --- 1) Grid only (No battery) → add to each controller's "No-batt"
        for tag in ('CVX', 'Tiny'):
            agg[tag]['price_no_batt'] += price * load_kW
            agg[tag]['emiss_no_batt'] += emis_rate * load_kW
            agg[tag]['grid_no_batt']  += load_kW * dt

        # --- 2) CVX MPC -------------------------------------------------------
        pg_cvx, _, _ = cvx_ctrl.predict()         # (grid kW, batt kW, SOC)
        pg_cvx = float(np.asarray(pg_cvx).ravel()[0])
        agg['CVX']['price_batt'] += price * pg_cvx
        agg['CVX']['emiss_batt'] += emis_rate * pg_cvx
        agg['CVX']['grid_batt']  += pg_cvx * dt

        # --- 3) Tiny MPC ------------------------------------------------------
        pg_tiny, _, _ = tiny_ctrl.predict()
        pg_tiny = float(np.asarray(pg_tiny).ravel()[0])
        agg['Tiny']['price_batt'] += price * pg_tiny
        agg['Tiny']['emiss_batt'] += emis_rate * pg_tiny
        agg['Tiny']['grid_batt']  += pg_tiny * dt

    # —— Append summary row ----------------------------------------------------
    for tag in ('CVX', 'Tiny'):
        row = {
            'Building'  : bldg,
            'Controller': tag,

            'Total Price Without Battery' : agg[tag]['price_no_batt'],
            'Total Price With Battery'    : agg[tag]['price_batt'],
            'Price Saving'                : agg[tag]['price_no_batt'] - agg[tag]['price_batt'],

            'Total Emissions Without Battery' : agg[tag]['emiss_no_batt'],
            'Total Emissions With Battery'    : agg[tag]['emiss_batt'],
            'Emissions Saving (kg)'           : agg[tag]['emiss_no_batt'] - agg[tag]['emiss_batt'],

            'Total Grid Energy Without Battery': agg[tag]['grid_no_batt'],
            'Total Grid Energy With Battery'   : agg[tag]['grid_batt'],
            'Grid Energy Saving'               : agg[tag]['grid_no_batt'] - agg[tag]['grid_batt'],
        }
        summary_rows.append(row)

# ===========================================================
# ❷  Save results & visualize
# ===========================================================
summary_df = pd.DataFrame(summary_rows)

# Save as CSV
out_dir = pathlib.Path('results')
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / 'summary_compare.csv'
summary_df.to_csv(csv_path, index=False)
print("\nSaved  ➜", csv_path)
print(summary_df, "\n")

# ------------------------------------------------------------------
# ❸  Bar charts for all metrics (2 Controllers × N-metrics)
# ------------------------------------------------------------------
# Exclude Building and Controller columns
metrics = [c for c in summary_df.columns
           if c not in ('Building', 'Controller')]

n_col   = 3                      # Plots per row
n_plot  = len(metrics)
n_row   = int(np.ceil(n_plot / n_col))

fig, axs = plt.subplots(n_row, n_col,
                        figsize=(4*n_col, 3*n_row),
                        squeeze=False)
axs = axs.ravel()

unit = {  # y-axis label (unit), if not found → no label
    'Total Price Without Battery'  : '$',
    'Total Price With Battery'     : '$',
    'Price Saving'                 : '$',
    'Total Emissions Without Battery' : 'kg',
    'Total Emissions With Battery'    : 'kg',
    'Emissions Saving (kg)'           : 'kg',
    'Total Grid Energy Without Battery': 'kWh',
    'Total Grid Energy With Battery'   : 'kWh',
    'Grid Energy Saving'               : 'kWh',
}

for i, col in enumerate(metrics):
    ax = axs[i]
    ax.bar(summary_df['Controller'], summary_df[col],
           color=['tab:blue', 'tab:orange'], width=.6)
    ax.set_title(col, fontsize=10)
    ax.set_ylabel(unit.get(col, ''), fontsize=9)
    ax.grid(axis='y', ls=':')
    # Label on top of each bar
    for x, v in enumerate(summary_df[col]):
        ax.text(x, v*1.01 if v>=0 else v*0.99,
                f'{v:.3f}', ha='center',
                va='bottom' if v>=0 else 'top',
                fontsize=7)

# Remove empty subplots
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

fig.suptitle('CVX vs TinyMPC — Comparison of All Metrics (72 h, Building 1)',
             fontsize=14)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.show()
