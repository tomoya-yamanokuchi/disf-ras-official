from domain_object.builder import SelfContainedDomainObjectBuilder

class ICP:
    @staticmethod
    def construct(
            builder: SelfContainedDomainObjectBuilder,
        ):
        # ---

        builder.build_config_icp()
        builder.build_icp_matcher()
        # ---
        return builder.get_domain_object()

