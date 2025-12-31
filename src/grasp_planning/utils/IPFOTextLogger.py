import numpy as np
from domain_object.builder import DomainObject
from print_color import print
from service import format_vector
from value_object import IPFOParams
from value_object import ISFResult
from value_object import IPFOResult
from service import ExtendedRotation
from value_object import SourcePointSurfaceSet


class IPFOTextLogger:
    def __init__(self, domain_object: DomainObject):
        self.verbose   = domain_object.verbose
        # ---------------------
        self.criteria           = domain_object.verbose.textlog.criteria
        self.ipfo               = domain_object.verbose.textlog.ipfo
        self.isf                = domain_object.verbose.textlog.isf
        self.gpa                = domain_object.verbose.textlog.gpa
        self.Rt_update          = domain_object.verbose.textlog.Rt_update
        self.error              = domain_object.verbose.textlog.error
        # ----


    def error_En(self, En : float):
        if not self.error: return
        print(f"En = {En:.3f}", tag ='error', tag_color='yellow', color='blue')

    def error_Ea(self, Ea : float):
        if not self.error: return
        print(f"Ea = {Ea:.3f}", tag='error', tag_color='yellow', color='red')

    def error_Ep(self, Ep : float):
        if not self.error: return
        print(f"Ep = {Ep:.3f}", tag='error', tag_color='yellow', color='green')


    def maximum_isf_iteration(self, iter_count, max_iter):
        if not self.isf: return
        print(f"iteraton reached to maximum ({iter_count}/{max_iter})",
                tag = 'ISF', tag_color='yellow', color='yellow')

    def delta_e(self, delta_e):
        if not self.criteria: return
        print("delta_e = ", delta_e, tag = 'ISF Criteria', tag_color='cyan', color='cyan')

    # ==================== IPFO ====================
    def dipfo_finished(self, result: IPFOResult):
        if not self.ipfo: return
        tag_color = "red"
        color     = "magenta"
        print(f"---------------------------------------------------------",                              tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f"       error = {format_vector(result.e_p_sum, decimal_places=4, format='decimal_exp')}", tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f"      rotvec = {ExtendedRotation.from_matrix(result.R_sum).as_rotvec()}",                tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f" translation = {result.t_sum}",                                                          tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f"     delta_d = {result.delta_d_sum}",                                                    tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f"---------------------------------------------------------",                              tag = 'DIPFO', tag_color=tag_color, color=color)

    def pfo_finished(self, result: IPFOResult):
        if not self.ipfo: return
        tag_color = "red"
        color     = "magenta"
        print(f"---------------------------------------------------------",                              tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f"       error = {format_vector(result.e_p_sum, decimal_places=4, format='decimal_exp')}", tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f"      rotvec = {ExtendedRotation.from_matrix(result.R_sum).as_rotvec()}",                tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f" translation = {result.t_sum}",                                                          tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f"     delta_d = {result.delta_d_sum}",                                                    tag = 'DIPFO', tag_color=tag_color, color=color)
        print(f"---------------------------------------------------------",                              tag = 'DIPFO', tag_color=tag_color, color=color)

    def palm_opt_finished(self, rotation_est, translation_est):
        if not self.ipfo: return
        print(f"palm optimization and update surface with rodrigues: rotvec={format_vector(rotation_est.as_rotvec())}, t={format_vector(translation_est)}",
                tag = 'IPFO', tag_color='red', color='magenta')

    def palm_trans_centroid_finished(self, t_center_refine):
        if not self.ipfo: return
        print(f"translation  center shit: tc={format_vector(t_center_refine, decimal_places=4)}",
                tag = 'IPFO', tag_color='red', color='magenta')

    def palm_trans_gripper_finished(self, translation_shift):
        if not self.ipfo: return
        print(f"translation gripper shit: tg={format_vector(translation_shift, decimal_places=4)}",
                tag = 'IPFO', tag_color='red', color='magenta')

    def finger_opt_finished(self, delta_d_est):
        if not self.ipfo: return
        print(f"fingertip displacement optimization: delta_d={format_vector([delta_d_est], decimal_places=4)}",
                tag = 'IPFO', tag_color='red', color='magenta')

    def error_diff_in_ipfo(self,
            source_set0       : SourcePointSurfaceSet,
            aligned_source_set: SourcePointSurfaceSet,
        ):
        if not self.ipfo: return
        source_error = np.linalg.norm(
            source_set0.correspondence.points - aligned_source_set.correspondence.points
        )
        print(f"source error (norm) = {source_error}",
                tag = '[class:IPFO]', tag_color='red', color='magenta')

    def error_diff_in_pfo(self, aligned_source_finger_opt, aligned_source_rigid):
        if not self.ipfo: return
        source_error = np.linalg.norm(
            aligned_source_finger_opt.points - aligned_source_rigid.points
        )
        print(" source_error (PFO) = ", source_error)

    # ==================== ISF ====================
    def loop_count(self, count: int):
        if not self.isf: return
        print(f" ------------ loop: i={count} ------------")

    def eta(self, eta: float):
        if not self.isf: return
        print(" eta = ", eta)

    def isf_finished(self, result: ISFResult):
        if not self.isf: return
        print(f"---------------------------------------------------------",                                                tag = 'ISF', tag_color='blue', color='blue')
        print(f"                  error = {format_vector(result.error, decimal_places=4, format='decimal_exp')}",          tag = 'ISF', tag_color='blue', color='blue')
        print(f" rotation (rotvec[deg]) = {format_vector(np.rad2deg(result.rotation.as_rotvec()),decimal_places=2)}",      tag = 'ISF', tag_color='blue', color='blue')
        print(f" rotation (rotvec[rad]) = {format_vector(result.rotation.as_rotvec(),            decimal_places=4)}",      tag = 'ISF', tag_color='blue', color='blue')
        print(f"      rotation (quat)   = {format_vector(result.rotation.as_quat_scalar_first(), decimal_places=4)}",      tag = 'ISF', tag_color='blue', color='blue')
        print(f"       translation      = {format_vector(result.translation,                     decimal_places=4)}",      tag = 'ISF', tag_color='blue', color='blue')
        print(f"                delta_d = {result.delta_d}",                                                               tag = 'ISF', tag_color='blue', color='blue')
        print(f"---------------------------------------------------------",                                                tag = 'ISF', tag_color='blue', color='blue')

    # ==================== GPA ====================
    def gpa_finished(self, result_cma, result: ISFResult):
        if not self.gpa: return
        print(f"---------------------------------------------------------------------")
        print(f"Best solution found: {repr(result_cma.xbest)}")
        print(f"Value of the objective function at this point: {result_cma.fbest}")
        print(f"---------------------------------------------------------",                                         tag = 'GPA', tag_color='green', color='green')
        print(f"             error = {format_vector(result.error, decimal_places=4, format='decimal_exp')}",        tag = 'GPA', tag_color='green', color='green')
        print(f" rotation (rotvec) = {format_vector(result.rotation.as_rotvec(),            decimal_places=4)}",    tag = 'GPA', tag_color='green', color='green')
        print(f" rotation (quat)   = {format_vector(result.rotation.as_quat_scalar_first(), decimal_places=4)}",    tag = 'GPA', tag_color='green', color='green')
        print(f"  translation_palm = {format_vector(result.translation_palm,                     decimal_places=4)}",    tag = 'GPA', tag_color='green', color='green')
        print(f"           delta_d = {result.delta_d}",                                                             tag = 'GPA', tag_color='green', color='green')
        print(f"---------------------------------------------------------",                                         tag = 'GPA', tag_color='green', color='green')


    # ==================== Rt_update ====================
    def set_R0(self, R0):
        if not self.Rt_update: return
        print(f"{R0}", tag = 'set_R0', tag_color='yellow', color='yellow')

    def update_Rt(self, Rt):
        if not self.Rt_update: return
        print(f"{Rt}", tag = 'update_Rt', tag_color='cyan', color='cyan')

    def error_diff(self, error_diff):
        if not self.Rt_update: return
        print(f"---------------------------------------------------------", tag = 'Rigid_Transform', tag_color='yellow', color='yellow')
        print(f" error_diff = {error_diff}", tag = 'Rigid_Transform', tag_color='yellow', color='yellow')
        print(f"---------------------------------------------------------", tag = 'Rigid_Transform', tag_color='yellow', color='yellow')

