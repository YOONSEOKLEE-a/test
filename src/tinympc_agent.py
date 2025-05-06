import numpy as np
import tinympc

class TinyMPCAgent:
    def __init__(self, environment, planning_steps=24, alpha=1000.0):
        self.env = environment
        self.N = planning_steps
        self.dt = 1.0

        self.n_bat = 2
        self.capacity = self.n_bat * 7.8
        self.soc = 0.5 * self.capacity
        self.soc_min = 0.05 * self.capacity
        self.soc_max = self.capacity
        self.p_max = 9.6

        self.eta_ch = 0.96
        self.eta_dis = 0.95

        self.A = np.array([[1.0]])
        self.B = np.array([[-self.dt]])
        self.Q = np.array([[0.0]])
        self.alpha = alpha

    def predict(self):
        k = self.env.current_step
        if k >= self.env.total_steps:
            raise RuntimeError("Environment step exceeds dataset length.")

        price_t = self.env.costData[k]
        load_t = self.env.BuildingLoad[k]

        # R 재계산
        R_scaled = np.array([[1e-8 + self.alpha * price_t]])

        # TinyMPC 객체 새로 생성
        mpc = tinympc.TinyMPC()
        mpc.setup(
            self.A, self.B, self.Q, R_scaled, self.N,
            x_min=np.full((1, self.N), self.soc_min),
            x_max=np.full((1, self.N), self.soc_max),
            u_min=np.full((1, self.N - 1), -self.p_max),
            u_max=np.full((1, self.N - 1),  self.p_max)
        )

        mpc.set_x0(np.array([self.soc]))
        sol = mpc.solve()
        u = float(sol["controls"][0])

        # SOC 업데이트
        if u >= 0:
            self.soc = max(self.soc - u / self.eta_dis, self.soc_min)
        else:
            self.soc = min(self.soc - u * self.eta_ch, self.soc_max)

        p_batt = u
        p_grid = load_t - p_batt
        self.env.current_step += 1

        return p_grid, p_batt, self.soc
