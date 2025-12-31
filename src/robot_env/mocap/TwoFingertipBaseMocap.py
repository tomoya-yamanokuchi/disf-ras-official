from .SingleFingertipBaseMocap import SingleFingertipBaseMocap


class TwoFingertipBaseMocap:
    def __init__(self,
            right_fingertip_base_mocap : SingleFingertipBaseMocap,
            left_fingertip_base_mocap  : SingleFingertipBaseMocap,
        ):
        self.right = right_fingertip_base_mocap
        self.left  = left_fingertip_base_mocap

    def update(self):
        self.right.update()
        self.left.update()

    def get_right_points(self):
        return self.right.get_points_world()

    def get_left_points(self):
        return self.left.get_points_world()

