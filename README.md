# 📊 MPC-Controller for Building Energy Management

This project implements a **Model Predictive Control (MPC)** system to optimize energy management in buildings with battery storage. The controller minimizes **energy costs** and **emissions** by scheduling the charging and discharging of batteries based on TOU price, forecasted  building loads, and emissions data.

---

## 📂 Project Structure

```
MPC-Controller/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── data/
│   ├── pvLoadPriceData6.csv # Example data file 
├── src/
│   ├── main.py            # Main script to run the control system
│   ├── environment.py      # Environment class for data handling
│   ├── mpc_agent.py        # MPC Agent class for predictive control
│   ├── utils.py            # Utility functions (plotting, etc.)
├── results/
│   ├── summary_results.csv # Output CSV (generated after running)
```

---

## ⚙️ Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/PIC-Lab/MPC4DER
cd MPC-Controller
```

### **2️⃣ Install Dependencies**
Ensure you have Python installed (>=3.8) and install required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the MPC Controller

### **Run the Simulation**
To execute the MPC optimization for battery control:
```bash
python -m src.main
```
This will:
✅ Load electricity price, emissions, and demand data.
✅ Optimize battery scheduling (charge/discharge/idle).
✅ Save summary results to `results/summary_results.csv`.
✅ Generate energy usage and SOC plots.

---

## 📊 Features
- **Predictive Battery Control:** Uses MPC to optimize charge/discharge cycles.
- **Cost & Emission Reduction:** Minimizes grid energy costs and environmental impact.
- **SOC & Power Visualization:** Plots battery SOC, load demand, and grid power usage.
- **Historical Tracking:** Logs battery mode (`Charge`, `Discharge`, `Idle`).

---

## 📜 Configuration
Modify the following parameters inside `src/mpc_agent.py`:
```python
MPCAgent(
    environment, 
    gamma=[0.15, 0.85], # Weighting factors for cost vs. emissions
    num_batteries=2,    # Number of battery units
    inverter_limit=9.6,  # Max inverter power (kW)
    planning_steps=24    # Look-ahead prediction horizon
)
```

---

## 📈 Visualizations
The following plots are generated after running the simulation:
1️⃣ **Battery SOC Over Time**  
2️⃣ **Electricity Price Fluctuations**  
3️⃣ **Power Flow (Grid, Battery, Load)**  
4️⃣ **Battery Mode (Charge, Discharge, Idle)**  


---

## 🛠 Requirements
```bash
numpy
cvxpy
matplotlib
pandas
gurobipy
```

---

## 👨‍💻 Author
- **Asmaa Romia** - __
- 📧 Contact: romia@mines.edu

---
