import numpy as np


def search_xy_around_init(self,
                          R0: np.ndarray,
                          t0: np.ndarray,
                          dx_range=(-0.02, 0.02),
                          dy_range=(-0.02, 0.02),
                          num_steps=9,
                          band: float = 0.005):

    # dx, dy の離散値を決める
    dxs = np.linspace(dx_range[0], dx_range[1], num_steps)
    dys = np.linspace(dy_range[0], dy_range[1], num_steps)

    best_score = -np.inf
    best_t = None

    for dx in dxs:
        for dy in dys:
            t = t0 + np.array([dx, dy, 0.0])  # まずは z は固定とする

            score = self.score_hand_pose_by_sdf(R0, t, band=band)

            if score > best_score:
                best_score = score
                best_t = t.copy()

    print("best_score:", best_score)
    print("best_t:", best_t)
    return best_score, R0, best_t
