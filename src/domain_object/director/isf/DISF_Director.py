from domain_object.builder import SelfContainedDomainObjectBuilder
from .PFOCommonDirector import PFOCommonDirector



class DISF_Director:
    @staticmethod
    def construct(builder: SelfContainedDomainObjectBuilder, config_name: str):
        # ---
        # builder = PFOCommonDirector.construct(
        #     builder       = builder,
        #     isf_model     = "disf",
        #     object_name   = config_name
        # )
        # ---
        builder.build_isf_visualizer()
        builder.build_set_error()
        builder.build_isf_loop_criteria()
        # ------------- Fingertip Dataset --------------
        builder.build_paired_finger_indices()
        # --------------- Least Square  ----------------
        builder.build_disf_palm_R_ls()
        builder.build_disf_finger_ls_Ep()
        # --------------- Optimization ------------------
        builder.build_disf_palm_R_opt()
        builder.build_disf_trans_centroid()
        builder.build_disf_finger_opt()
        # ------------------- IPFO ----------------------
        builder.build_disf()
        # -----------------------------------------------
        return builder.get_domain_object()

