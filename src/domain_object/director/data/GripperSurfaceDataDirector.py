from domain_object.builder import SelfContainedDomainObjectBuilder


class GripperSurfaceDataDirector:
    @staticmethod
    def construct(
            builder    : SelfContainedDomainObjectBuilder,
        ):
        # ---
        builder.build_paired_finger_source()
        # builder.build_hand_origin()
        # builder.build_surface_visualizer()
        # ---
        return builder.get_domain_object()

