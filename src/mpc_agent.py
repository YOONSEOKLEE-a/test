import cvxpy as cp
import numpy as np

# MPC Agent Class for Predictive Control
class MPCAgent:
    def __init__(self, environment, gamma, num_batteries=2, inverter_limit=9.6, safety_margin=0.9, planning_steps=24):
        self.environment = environment
        self.gamma = gamma
        self.safety_margin = safety_margin
        self.planning_steps = planning_steps
        self.dt = 1  # 1-hour time steps

        # Battery Configuration
        self.num_batteries = num_batteries
        self.charge_efficiency = 0.96
        self.discharge_efficiency = 0.95

        self.batteryEnergy = self.num_batteries * 7.8  # Total Usable capacity in kWh
        self.Einit = 0.5 * self.batteryEnergy  # 50% SOC initial

        # Min & Max SOC based on usable capacity
        self.min_soc = 0.05  # Min SOC (5%)
        self.max_soc = 1.0   # Max SOC (100%)

        # Theoretical Battery Power Based on Number of Modules
        theoretical_Pmax = 9.830  # Battery limit: 9.83 kW per module
        self.inverter_limit = inverter_limit  # Inverter power in kW
        self.Pmax = min(theoretical_Pmax, self.inverter_limit)
 # ✅ Ensure mode history exists
        self.soc_history = []    # To store State of Charge (SOC)
        self.mode_history = []   # To store charging, discharging, idle modes
      

    # Predictive Control (Optimization Model)
    def predict(self):
        k = self.environment.current_step
        numSteps = self.environment.total_steps
        forecastSteps = min(self.planning_steps, numSteps - k)

        # Extracts price, demand, and emissions forecasts
        Cost = self.environment.costData[k:k + forecastSteps]
        Phome = self.environment.BuildingLoad[k:k + forecastSteps]
        EmissionRate = self.environment.Emissions[k:k + forecastSteps]

        # Decision variables
        PgridV = cp.Variable(forecastSteps, nonneg=True)
        EbattV = cp.Variable(forecastSteps)
        PbattV = cp.Variable(forecastSteps)  # Net battery power

        # Define explicit charging and discharging components
        Pbatt_charge = cp.Variable(forecastSteps, nonneg=True)  # Charging power (≥0)
        Pbatt_discharge = cp.Variable(forecastSteps, nonneg=True)  # Discharging power (≥0)

        # Cost and emissions normalization
        emission_cost_rate = 0.051  # $ per kg of emissions

        # Objective function (Cost + Emissions)
        max_cost = self.environment.maxCost if self.environment.maxCost > 0 else 1
        max_emissions = self.environment.maxEmissions if self.environment.maxEmissions > 0 else 1

        Obj1 = cp.sum(cp.multiply(self.dt * Cost / max_cost, PgridV))  # Normalized grid cost
        Obj2 = cp.sum(cp.multiply(self.dt * EmissionRate * emission_cost_rate / max_emissions, PgridV))  # Normalized emissions cost
  # Emission cost
        objective = cp.Minimize(self.gamma[0] * Obj1 + self.gamma[1] * Obj2)

        # Constraints
        constraints = [
            EbattV[0] == self.Einit,  # Initial SOC
            EbattV >= self.min_soc * self.batteryEnergy,  # Min SOC
            EbattV <= self.max_soc * self.batteryEnergy,  # Max SOC
            PgridV - PbattV - Phome == 0,  # Grid balance equation
            PbattV == Pbatt_charge - Pbatt_discharge,  # Ensure one mode at a time
            Pbatt_charge >= 0,  # Charging power ≥ 0
            Pbatt_discharge >= 0,  # Discharging power ≥ 0
            Pbatt_charge <= self.Pmax,  # Charging limit
            Pbatt_discharge <= self.Pmax,  # Discharging limit
            EbattV[-1] >= 0.4 * self.batteryEnergy,  # Final SOC constraints
            EbattV[-1] <= 0.8 * self.batteryEnergy,
        ]

        if forecastSteps > 1:
            constraints += [
                EbattV[1:] == EbattV[:-1] 
                + cp.multiply(Pbatt_charge[:-1], self.charge_efficiency * self.dt)  
                - cp.multiply(Pbatt_discharge[:-1], self.dt / self.discharge_efficiency)
            ]

        # Solve optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GUROBI, verbose=False)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            Pgrid_total = PgridV.value
            Pbatt_total = PbattV.value
            Ebatt = EbattV.value
            self.Einit = Ebatt[1]  # Update SOC
            self.environment.current_step += 1
            return Pgrid_total, Pbatt_total, Ebatt
        else:
            print(f"Optimization failed at step: {self.environment.current_step}")
            return None


