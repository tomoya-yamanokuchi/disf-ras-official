import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
matplotlib.use('TkAgg')

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List
from domain_object.builder import DomainObject
from value_object import PointNormalUnitPairs
from value_object import PointNormalIndexUnitPairs
from value_object import SourcePointSurfaceSet, TargetPointSurfaceSet
# from value_object import PointNormalUnitPairs
from .axis_point_normal_plot import axis_point_normal_plot
from .axis_point_normal_plot_with_different_finger_color import axis_point_normal_plot_with_different_finger_color
from service import set_aspect_equal_3d


class SurfaceVisualization:
    def __init__(self, domain_object: DomainObject):
        self.figsize                   = domain_object.config_isf.visualize.figsize
        self.save_path                 = domain_object.config_isf.visualize.save_path
        self.mode                      = domain_object.config_isf.visualize.mode
        self.elev                      = domain_object.config_isf.visualize.elev
        self.azim                      = domain_object.config_isf.visualize.azim
        self.point_size                = domain_object.config_isf.visualize.point_size
        self.normal_length             = domain_object.config_isf.visualize.normal_length
        self.point_alpha               = domain_object.config_isf.visualize.point_alpha
        self.normal_alpha              = domain_object.config_isf.visualize.normal_alpha
        self.hand_origin_point_size    = domain_object.config_isf.visualize.hand_origin_point_size
        self.hand_origin_normal_length = domain_object.config_isf.visualize.hand_origin_normal_length
        self.label_fontsize            = domain_object.config_isf.visualize.label_fontsize
        self.n_app                     = domain_object.n_app
        # -----
        self.visualize_call = domain_object.config_isf.verbose.visualize_call
        # self.call_level = 1



    def plot_n_z(self, n_z: np.ndarray):
        p0 = np.zeros(3)
        self.ax.quiver(
            *p0,
            *n_z,
            # ---
            color  = "red",
            length = self.normal_length*10,
            alpha  = self.normal_alpha,
        )

    def plot_n_app(self):
        p0 = np.zeros(3)
        self.ax.quiver(
            *p0,
            *self.n_app,
            # ---
            color  = "blue",
            length = self.normal_length*10,
            alpha  = self.normal_alpha,
        )


    def set_target_information(self,
            object_surface  : PointNormalUnitPairs,
            contact_indices : np.ndarray,
        ):
        self.object_surface  = object_surface
        self.contact_indices = contact_indices


    def plot_correspondence(self,
            correspondence: PointNormalUnitPairs,
            data_type     : str, # source or target
            color         : str,
        ):
        axis_point_normal_plot(
            ax            = self.ax,
            point_normal  = correspondence,
            label         = f"{data_type} (correspondence)",
            color         = color,
            point_size    = self.point_size*3,
            normal_length = self.normal_length*3,
            point_alpha   = self.point_alpha,
            normal_alpha  = self.normal_alpha,
        )


    def plot_source_surface(self,
            surface   : PointNormalUnitPairs,
            data_type : str, # source or target
            color     : str,
        ):
        # import ipdb; ipdb.set_trace()
        # axis_point_normal_plot(
        axis_point_normal_plot_with_different_finger_color(
            ax            = self.ax,
            point_normal  = surface,
            label         = f"{data_type} (surface)",
            color         = color,
            point_size    = self.point_size*0.6,
            normal_length = self.normal_length*2,
            point_alpha   = self.point_alpha,
            normal_alpha  = self.normal_alpha,
        )

    def plot_object_contact_surface(self,
            target_contact : TargetPointSurfaceSet,
            data_type : str, # source or target
            color     : str,
        ):
        axis_point_normal_plot(
            ax            = self.ax,
            point_normal  = target_contact,
            label         = f"{data_type} (surface)",
            color         = color,
            point_size    = self.point_size*1.2,
            normal_length = self.normal_length*2,
            point_alpha   = self.point_alpha,
            normal_alpha  = self.normal_alpha,
        )


    def plot_whole_object(self, target, color : str):
        axis_point_normal_plot(
            ax            = self.ax,
            point_normal  = target,
            label         = "target (whole)",
            color         = color,
            point_size    = self.point_size*0.3,
            normal_length = self.normal_length*0.3,
            point_alpha   = self.point_alpha,
            normal_alpha  = self.normal_alpha,
        )

    def plot_origin_point(self, color : str):
        axis_point_normal_plot(
            ax            = self.ax,
            point_normal  = PointNormalUnitPairs(points=np.zeros([1, 3]), normals=np.zeros([1, 3])),
            label         = "origin",
            color         = color,
            point_size    = self.point_size*3,
            normal_length = self.normal_length*0,
            point_alpha   = self.point_alpha,
            normal_alpha  = self.normal_alpha,
        )


    def set_parameters(self, title: str = None):
        self.ax.set_xlabel('X', fontsize=self.label_fontsize)
        self.ax.set_ylabel('Y', fontsize=self.label_fontsize)
        self.ax.set_zlabel('Z', fontsize=self.label_fontsize)
        # ---
        plt.title(title, fontsize=self.label_fontsize)
        # ---
        set_aspect_equal_3d(self.ax)
        self.ax.view_init(elev=self.elev, azim=self.azim)


    def show_or_save(self):
        if self.mode == 0:
            plt.show()
            plt.close()
        elif self.mode == 1:
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=500, bbox_inches='tight')
            plt.close()

    def visualize(self,
            source         : PointNormalIndexUnitPairs,
            target_whole   : PointNormalUnitPairs,
            target_contact : PointNormalUnitPairs,
            n_z            : np.ndarray,
            # ---
            call_level : int,
            title      : str = None,
            origin_point : bool = True,
        ):
        if self.visualize_call == -1              : return
        if not (self.visualize_call <= call_level): return
        # ---- make plot object ----
        fig              = plt.figure(figsize=self.figsize)
        self.ax : Axes3D = fig.add_subplot(111, projection='3d')
        # -------- whole_object ---------
        self.plot_whole_object(target_whole, color='darkgray')
        self.plot_object_contact_surface(target_contact, data_type="target", color='blue')
        # ---
        self.plot_source_surface(source, data_type="source", color='red')
        if origin_point:
            self.plot_origin_point(color="darkorange")
        # --------- n_app ---------
        self.plot_n_z(n_z)
        self.plot_n_app()
        # --------- save -----------
        self.set_parameters(title=title)
        self.show_or_save()

