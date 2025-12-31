from domain_object.builder import SelfContainedDomainObjectBuilder
from args import parse_args


def run(robot_name: str = "panda"):
    # ----
    object_name = "006_mustard_bottle"
    # ---------------------------------
    builder       = SelfContainedDomainObjectBuilder()
    # ----
    builder.set_robot_name(robot_name)
    builder.build_config_gripper_surface(config_name=robot_name)
    builder.build_paired_finger_source()
    builder.build_config_isf(config_name="disf")
    builder.build_ipfo_parameters()
    builder.build_config_point_cloud_data(config_name=object_name)
    builder.build_isf_visualizer()
    # ---- get domain object -----------
    domain_object  = builder.get_domain_object()
    isf_visualizer = domain_object.isf_visualizer
    source         = domain_object.source
    # ---------------------------------
    # domain_object.config_isf.visualize.point_normal.color.source.contact
    domain_object.config_isf.visualize.point_normal.point_size.source.contact = 4


    # ---------------------------------
    isf_visualizer.figsize = (6.5, 6.5)
    isf_visualizer.use_fixed_limit = True
    isf_visualizer.elev = +20.0
    isf_visualizer.azim = 230.0


    # --- set axis limit ---
    isf_visualizer.x_min = -0.06
    isf_visualizer.x_max =  0.06
    # ---
    isf_visualizer.y_min = -0.06
    isf_visualizer.y_max =  0.06
    # ---
    isf_visualizer.z_min = -0.06
    isf_visualizer.z_max =  0.06

    isf_visualizer.reset_figure()
    isf_visualizer.mode = 1  # save mode
    isf_visualizer.plot_source_contact(source)

    isf_visualizer.axis_label = True
    # isf_visualizer.empty_ticks = False
    isf_visualizer.label_fontsize = 30 # 20
    isf_visualizer.set_parameters()

    isf_visualizer.show_or_save(filename="gripper_pc_visualization.png")

if __name__ == '__main__':
    args = parse_args()
    run(robot_name = args.robot_name)
