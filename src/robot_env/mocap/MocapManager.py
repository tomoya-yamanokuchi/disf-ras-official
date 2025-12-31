import numpy as np
from ..utils.BodyManager import BodyManager

class MocapManager:
    def __init__(self, model, data, body: BodyManager):
        self.model = model
        self.data  = data
        self.body  = body

    def name2id(self, mocap_body_name: str):
        mocap_general_body_id  = self.body.name2id(mocap_body_name)
        mocap_specific_body_id = self.model.body_mocapid[mocap_general_body_id]
        return mocap_specific_body_id

    def get_pos(self, mocap_body_id: int):
        return self.data.mocap_pos[mocap_body_id]

    def get_quat(self, mocap_body_id: int):
        return self.data.mocap_quat[mocap_body_id]

    def set_pos(self, mocap_body_id: int, mocap_pos: np.ndarray):
        self.data.mocap_pos[mocap_body_id] = mocap_pos

    def set_quat(self, mocap_body_id: int, mocap_quat: np.ndarray):
        self.data.mocap_quat[mocap_body_id] = mocap_quat
        if mocap_body_id == 0:
            print(f'mocap_body_id {mocap_body_id} | mocap_quat = {mocap_quat}')