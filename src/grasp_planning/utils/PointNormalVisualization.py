from domain_object.builder import DomainObject
from service import visualize_n_point_clouds
from service import visualize_n_point_clouds_with_hand_origin_and_approach_direction
from service import visualize_n_point_clouds_icp
from value_object import PointNormalUnitPairs
import numpy as np
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet


class PointNormalVisualization:
    def __init__(self, domain_object: DomainObject):
        self.domain_object             = domain_object
        self.visualize_call            = domain_object.verbose.visualize_call
        # ----
        self.mode                      = domain_object.config_isf.visualize.mode
        self.elev                      = domain_object.config_isf.visualize.elev
        self.azim                      = domain_object.config_isf.visualize.azim
        # ---
        self.hand_origin_point_size    = domain_object.config_isf.visualize.hand_origin_point_size
        self.hand_origin_normal_length = domain_object.config_isf.visualize.hand_origin_normal_length
        self.figsize                   = domain_object.config_isf.visualize.figsize

    def plot_point_normal(self,
            source : PointNormalUnitPairs,
            target : PointNormalUnitPairs,
            title     : str,
            call_level: int,
        ):
        if self.visualize_call == -1              : return
        if not (self.visualize_call <= call_level): return
        # -----
        visualize_n_point_clouds(
            point_normals = [source, target],
            labels        = ["source", "target"],
            colors        = ["r", "b"],
            title         = title,
            save_path     = self.save_path,
            mode          = self.mode,
            point_size    = self.point_size,
            normal_length = self.normal_length,
            elev          = self.elev,
            azim          = self.azim,
            figsize       = self.figsize,
        )


    def plot_2set(self,
            source_set : SourcePointSurfaceSet,
            target_set : TargetPointSurfaceSet,
            title     : str,
            call_level: int,
        ):
        if self.visualize_call == -1              : return
        if not (self.visualize_call <= call_level): return
        # -----
        visualize_n_point_clouds(
            point_normals = [source_set.correspondence, target_set.correspondence],
            labels        = ["source", "target"],
            colors        = ["r", "b"],
            title         = title,
            save_path     = self.save_path,
            mode          = self.mode,
            point_size    = self.point_size,
            normal_length = self.normal_length,
            elev          = self.elev,
            azim          = self.azim,
        )


    def plot_icp_skew_5set(self,
            source_set_skew : SourcePointSurfaceSet,
            source_set_rod  : SourcePointSurfaceSet,
            target_set      : TargetPointSurfaceSet,

            # source_skew   : PointNormalUnitPairs,
            # source_rod    : PointNormalUnitPairs,
            # target        : PointNormalUnitPairs,
            # source_surface: PointNormalUnitPairs,
            # target_surface: PointNormalUnitPairs,
            title         : str,
            call_level    : int,
        ):
        if self.visualize_call == -1              : return
        if not (self.visualize_call <= call_level): return
        # ---
        # import ipdb ; ipdb.set_trace()
        visualize_n_point_clouds_icp(
            point_normals         = [source_set_skew.correspondence, source_set_rod.correspondence, target_set.correspondence],
            labels                = ["skew", "rodrigues", "target"],
            colors                = ['gray', 'r', 'b'],
            # ----
            surface_point_normals = [source_set_rod.surface, target_set.contact_surface],
            surface_labels        = ["source_surface", "target_surface"],
            surface_colors        = ['plum', 'skyblue'],
            # ---
            title         = title,
            save_path     = self.save_path,
            mode          = self.mode,
            point_size    = self.point_size,
            normal_length = self.normal_length,
        )


    def plot_icp_4set(self,
            source_set : SourcePointSurfaceSet,
            target_set : TargetPointSurfaceSet,
            # source        : PointNormalUnitPairs,
            # target        : PointNormalUnitPairs,
            # source_surface: PointNormalUnitPairs,
            # target_surface: PointNormalUnitPairs,
            title         : str,
            call_level    : int,
        ):
        # import ipdb ; ipdb.set_trace()
        if self.visualize_call == -1              : return
        if not (self.visualize_call <= call_level): return
        # ---
        # import ipdb ; ipdb.set_trace()
        visualize_n_point_clouds_icp(
            point_normals         = [source_set.correspondence, target_set.correspondence],
            labels                = ["source", "target"],
            colors                = ['r', 'b'],
            # ----
            surface_point_normals = [source_set.surface, target_set.contact_surface],
            surface_labels        = ["source_surface", "target_surface"],
            surface_colors        = ['plum', 'skyblue'],
            # ---
            title         = title,
            save_path     = self.save_path,
            mode          = self.mode,
            point_size    = self.point_size,
            normal_length = self.normal_length,
        )



    def plot_3set(self,
            aligned_source_skew     : PointNormalUnitPairs,
            aligned_source_rodrigues: PointNormalUnitPairs,
            target                  : PointNormalUnitPairs,
            title                   : str,
            call_level              : int,
        ):
        if self.visualize == -1             : return
        if not (self.visualize < call_level): return
        # ---
        visualize_n_point_clouds(
            point_normals = [aligned_source_skew, aligned_source_rodrigues, target],
            labels        = ["skew", "rodrigues", "target"],
            colors        = ['r', 'g', 'b'],
            title         = title,
            save_path     = self.save_path,
            mode          = self.mode,
            point_size    = self.point_size,
            normal_length = self.normal_length,
        )


    def plot_3set_approach(self,
            source_skew: PointNormalUnitPairs,
            source_rod : PointNormalUnitPairs,
            target     : PointNormalUnitPairs,
            n_z        : np.ndarray,
            title      : str,
            call_level : int,
        ):
        if self.visualize == -1              : return
        if not (self.visualize <= call_level): return
        # ---
        self.n_app = self.domain_object.n_approach
        visualize_n_point_clouds_with_hand_origin_and_approach_direction(
            point_normals             = [source_skew, source_rod, target],
            labels                    = ["skew", "rodrigues", "target"],
            colors                    = ['gray', 'r', 'b'],
            n_z                       = n_z,
            n_app                     = self.n_app,
            title                     = title,
            save_path                 = self.save_path,
            mode                      = self.mode,
            point_size                = self.point_size,
            hand_origin_point_size    = self.hand_origin_point_size,
            normal_length             = self.normal_length,
            hand_origin_normal_length = self.hand_origin_normal_length,
        )


    def plot_2set_approach(self,
            source     : PointNormalUnitPairs,
            target     : PointNormalUnitPairs,
            n_z        : np.ndarray,
            title      : str,
            call_level : int,
        ):
        if self.visualize == -1              : return
        if not (self.visualize <= call_level): return
        # ---
        self.n_app = self.domain_object.n_approach
        visualize_n_point_clouds_with_hand_origin_and_approach_direction(
            point_normals             = [source, target],
            labels                    = ["source", "target"],
            colors                    = ['r', 'b'],
            n_z                       = n_z,
            n_app                     = self.n_app,
            title                     = title,
            save_path                 = self.save_path,
            mode                      = self.mode,
            point_size                = self.point_size,
            hand_origin_point_size    = self.hand_origin_point_size,
            normal_length             = self.normal_length,
            hand_origin_normal_length = self.hand_origin_normal_length,
        )
