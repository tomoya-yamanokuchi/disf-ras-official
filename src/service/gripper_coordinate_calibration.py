"""
グリッパー座標系のキャリブレーション・検証ツール

このスクリプトは、MuJoCoシミュレーション（sim）と実UR（real）の間の
グリッパー座標系変換を実験的に検証・改善するためのツールです。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import json
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class GraspPose:
    """把握姿勢の記録"""
    name: str  # 把握位置の名前
    rotvec_deg: List[float]  # 回転ベクトル [deg]
    translation_m: List[float]  # 並進 [m]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)


@dataclass
class CalibrationData:
    """キャリブレーションデータセット"""
    timestamp: str
    grasp_poses: List[GraspPose]
    ur_measured_poses: List[Dict]  # UR RTDEから取得した実測値
    transformation_matrix: List[List[float]]  # 4x4 行列
    residual_error: float  # 最小二乗誤差
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'grasp_poses': [gp.to_dict() for gp in self.grasp_poses],
            'ur_measured_poses': self.ur_measured_poses,
            'transformation_matrix': self.transformation_matrix,
            'residual_error': self.residual_error,
        }
    
    @classmethod
    def from_dict(cls, d: Dict):
        grasp_poses = [GraspPose.from_dict(gp) for gp in d['grasp_poses']]
        return cls(
            timestamp=d['timestamp'],
            grasp_poses=grasp_poses,
            ur_measured_poses=d['ur_measured_poses'],
            transformation_matrix=d['transformation_matrix'],
            residual_error=d['residual_error'],
        )


class GripperCoordinateCalibrator:
    """
    グリッパー座標系キャリブレーション
    
    Usage:
        calibrator = GripperCoordinateCalibrator()
        
        # 既知のgrasp poseをいくつか設定
        calibrator.add_grasp_pose(
            GraspPose("position_1", [0, 0, 0], [0.1, 0.05, -0.02]),
            ur_tcp_pose=[x, y, z, rx, ry, rz]
        )
        
        # キャリブレーション実行
        T_matrix, residual = calibrator.calibrate()
        
        # 検証用に新しいpose で変換を試す
        predicted_ur_pose = calibrator.transform_pose(rotvec_deg, translation_m)
    """
    
    def __init__(self):
        self.grasp_poses: List[GraspPose] = []
        self.ur_measured_poses: List[np.ndarray] = []  # 4x4 SE(3) matrices
        self.T_sim2real: np.ndarray = None
        self.residual_error: float = None
    
    def add_grasp_pose(
        self,
        grasp_pose: GraspPose,
        ur_tcp_pose: Tuple[float, float, float, float, float, float],
    ) -> None:
        """
        キャリブレーションデータを追加
        
        Args:
            grasp_pose: MuJoCo での grasp 姿勢
            ur_tcp_pose: UR RTDE で測定された TCP 姿勢 [x, y, z, rx, ry, rz]
        """
        self.grasp_poses.append(grasp_pose)
        
        # UR pose を SE(3) 行列に変換
        ur_pose = np.asarray(ur_tcp_pose)
        T_ur = self._pose_to_se3(ur_pose)
        self.ur_measured_poses.append(T_ur)
    
    @staticmethod
    def _pose_to_se3(pose: np.ndarray) -> np.ndarray:
        """
        UR pose (6D) を SE(3) 行列 (4x4) に変換
        
        Args:
            pose: [x, y, z, rx, ry, rz] (rx, ry, rz は axis-angle [rad])
        
        Returns:
            4x4 SE(3) 行列
        """
        p = pose[:3]
        rotvec = pose[3:]
        R_mat = R.from_rotvec(rotvec).as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = p
        return T
    
    @staticmethod
    def _se3_to_pose(T: np.ndarray) -> np.ndarray:
        """
        SE(3) 行列 (4x4) を UR pose (6D) に変換
        """
        p = T[:3, 3]
        R_mat = T[:3, :3]
        rotvec = R.from_matrix(R_mat).as_rotvec()
        return np.concatenate([p, rotvec])
    
    def _sim_pose_to_se3(self, rotvec_deg: np.ndarray, translation_m: np.ndarray) -> np.ndarray:
        """
        MuJoCo pose を SE(3) 行列に変換
        """
        rotvec_rad = np.deg2rad(rotvec_deg)
        R_mat = R.from_rotvec(rotvec_rad).as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = translation_m
        return T
    
    def _residual_fn(self, T_flat: np.ndarray) -> np.ndarray:
        """
        最小二乗法の残差関数
        
        T_flat: 12D (SE(3) 行列の上3行をフラット化)
        """
        # 12D ベクトルを 4x4 行列に復元
        T = np.eye(4)
        T[:3, :] = T_flat.reshape(3, 4)
        
        residuals = []
        for grasp_pose, ur_measured in zip(self.grasp_poses, self.ur_measured_poses):
            # sim pose を SE(3) に
            T_sim = self._sim_pose_to_se3(
                np.array(grasp_pose.rotvec_deg),
                np.array(grasp_pose.translation_m),
            )
            
            # sim → real 変換を予測
            T_predicted = T @ T_sim  # T_sim2real @ T_sim
            
            # 残差を計算（回転と並進の両方）
            R_diff = T_predicted[:3, :3] @ ur_measured[:3, :3].T
            # 回転行列の差分を軸角ベクトルで表現
            rotvec_diff = R.from_matrix(R_diff).as_rotvec()
            residual_R = rotvec_diff  # 3D
            
            # 並進の差分
            residual_t = T_predicted[:3, 3] - ur_measured[:3, 3]  # 3D
            
            # 結合（回転と並進の誤差を同じスケールで扱う）
            residuals.extend(residual_R)
            residuals.extend(residual_t)
        
        return np.array(residuals)
    
    def calibrate(self, method: str = 'lm') -> Tuple[np.ndarray, float]:
        """
        キャリブレーション実行
        
        Args:
            method: 最適化手法 ('lm': Levenberg-Marquardt など)
        
        Returns:
            T_sim2real: 4x4 SE(3) 行列
            residual_error: 残差の二乗和
        """
        if len(self.grasp_poses) < 3:
            raise ValueError("キャリブレーションには最低3つ以上のサンプルが必要です")
        
        # 初期値: 単位行列
        T_init = np.eye(4)
        x0 = T_init[:3, :].flatten()
        
        # 最適化
        result = least_squares(
            self._residual_fn,
            x0,
            method=method,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=10000,
        )
        
        # 結果を 4x4 行列に復元
        T_sim2real = np.eye(4)
        T_sim2real[:3, :] = result.x.reshape(3, 4)
        
        # 残差
        residual_error = np.sum(result.fun ** 2)
        
        self.T_sim2real = T_sim2real
        self.residual_error = residual_error
        
        print(f"✓ キャリブレーション完了")
        print(f"  残差: {residual_error:.6f}")
        print(f"  推定行列:")
        print(T_sim2real)
        
        return T_sim2real, residual_error
    
    def transform_pose(
        self,
        rotvec_deg: np.ndarray,
        translation_m: np.ndarray,
    ) -> np.ndarray:
        """
        MuJoCo の grasp pose を UR pose に変換
        
        Args:
            rotvec_deg: [rx_deg, ry_deg, rz_deg]
            translation_m: [x, y, z]
        
        Returns:
            UR pose [x, y, z, rx, ry, rz]
        """
        if self.T_sim2real is None:
            raise ValueError("先に calibrate() を実行してください")
        
        T_sim = self._sim_pose_to_se3(
            np.asarray(rotvec_deg),
            np.asarray(translation_m),
        )
        
        T_ur = self.T_sim2real @ T_sim
        
        ur_pose = self._se3_to_pose(T_ur)
        return ur_pose
    
    def validate(self) -> Dict[str, float]:
        """
        キャリブレーションの精度を検証
        
        Returns:
            誤差統計の辞書
        """
        if self.T_sim2real is None:
            raise ValueError("先に calibrate() を実行してください")
        
        errors_r = []  # 回転誤差 [rad]
        errors_t = []  # 並進誤差 [m]
        
        for grasp_pose, ur_measured in zip(self.grasp_poses, self.ur_measured_poses):
            T_sim = self._sim_pose_to_se3(
                np.array(grasp_pose.rotvec_deg),
                np.array(grasp_pose.translation_m),
            )
            
            T_predicted = self.T_sim2real @ T_sim
            
            # 回転誤差
            R_diff = T_predicted[:3, :3] @ ur_measured[:3, :3].T
            rotvec_diff = R.from_matrix(R_diff).as_rotvec()
            error_r = np.linalg.norm(rotvec_diff)
            errors_r.append(error_r)
            
            # 並進誤差
            error_t = np.linalg.norm(T_predicted[:3, 3] - ur_measured[:3, 3])
            errors_t.append(error_t)
        
        return {
            'rotation_error_mean_rad': float(np.mean(errors_r)),
            'rotation_error_std_rad': float(np.std(errors_r)),
            'rotation_error_max_rad': float(np.max(errors_r)),
            'translation_error_mean_m': float(np.mean(errors_t)),
            'translation_error_std_m': float(np.std(errors_t)),
            'translation_error_max_m': float(np.max(errors_t)),
        }
    
    def save_calibration(self, filepath: str) -> None:
        """キャリブレーション結果をJSONで保存"""
        if self.T_sim2real is None:
            raise ValueError("先に calibrate() を実行してください")
        
        cal_data = CalibrationData(
            timestamp=datetime.now().isoformat(),
            grasp_poses=self.grasp_poses,
            ur_measured_poses=[m.tolist() for m in self.ur_measured_poses],
            transformation_matrix=self.T_sim2real.tolist(),
            residual_error=float(self.residual_error),
        )
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(cal_data.to_dict(), f, indent=2)
        
        print(f"✓ キャリブレーション結果を保存しました: {filepath}")
    
    def load_calibration(self, filepath: str) -> None:
        """保存したキャリブレーション結果を読み込み"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        cal_data = CalibrationData.from_dict(data)
        self.grasp_poses = cal_data.grasp_poses
        self.ur_measured_poses = [np.array(m) for m in cal_data.ur_measured_poses]
        self.T_sim2real = np.array(cal_data.transformation_matrix)
        self.residual_error = cal_data.residual_error
        
        print(f"✓ キャリブレーション結果を読み込みました: {filepath}")


class CoordinateFrameAnalyzer:
    """座標系の詳細な分析・可視化"""
    
    @staticmethod
    def analyze_current_transformation() -> Dict:
        """現在の trans_convert_ISF2UR と box_pose_to_ur_pose を分析"""
        
        # 現在の変換行列を抽出
        from ur3e_grasp.geometry import (
            DEFAULT_TRANS_ROT_UR,
            DEFAULT_TRANS_OFFSET_UR,
            DEFAULT_BASE_ROT_RV_UR,
        )
        
        R_trans = np.array(DEFAULT_TRANS_ROT_UR)
        t_offset = np.array(DEFAULT_TRANS_OFFSET_UR)
        rotvec_base = np.array(DEFAULT_BASE_ROT_RV_UR)
        R_base = R.from_rotvec(rotvec_base).as_matrix()
        
        return {
            'R_trans': R_trans.tolist(),
            't_offset': t_offset.tolist(),
            'R_base': R_base.tolist(),
            'description': {
                'R_trans': 'Z軸中心 -90° + Z軸180° の組み合わせ',
                't_offset': '並進オフセット（グリッパマウント位置を補正）',
                'R_base': '基準姿勢の回転行列',
            }
        }
    
    @staticmethod
    def print_transformation_summary():
        """変換の概要を人間が読みやすい形で出力"""
        import json
        analysis = CoordinateFrameAnalyzer.analyze_current_transformation()
        
        print("\n" + "="*60)
        print("グリッパー座標系変換の分析")
        print("="*60)
        print("\n[現在の実装]")
        print("\n1. trans_convert_ISF2UR (並進のみ):")
        print("   UR_x = ISF_y")
        print("   UR_y = -ISF_x")
        print("   UR_z = ISF_z")
        print("   → Z軸中心 -90° 回転")
        
        print("\n2. box_pose_to_ur_pose (回転と並進):")
        print("   回転: R_tcp = R_box @ R_base")
        print("   並進: p_tcp = R_trans @ p_sim + t_offset")
        
        print("\n[ハードコード値]")
        print(f"   DEFAULT_BASE_ROT_RV_UR: {analysis['description']['R_base']}")
        print(f"   DEFAULT_TRANS_ROT_UR: {analysis['description']['R_trans']}")
        print(f"   DEFAULT_TRANS_OFFSET_UR: {analysis['description']['t_offset']}")
        print("\n" + "="*60 + "\n")


# 使用例
if __name__ == "__main__":
    # 分析の出力
    CoordinateFrameAnalyzer.print_transformation_summary()
    
    # キャリブレータの初期化（デモンストレーション）
    print("キャリブレータの使用例:")
    print("-" * 60)
    
    calibrator = GripperCoordinateCalibrator()
    
    # ダミーデータで動作確認（実際には UR で計測したデータを使用）
    calibrator.add_grasp_pose(
        GraspPose("demo_1", [0.0, 0.0, 0.0], [0.1, 0.05, -0.02]),
        ur_tcp_pose=[0.5, 0.2, 0.3, 0.01, 0.02, 0.03],
    )
    calibrator.add_grasp_pose(
        GraspPose("demo_2", [10.0, 5.0, 0.0], [0.12, 0.06, -0.025]),
        ur_tcp_pose=[0.52, 0.22, 0.29, 0.02, 0.03, 0.04],
    )
    calibrator.add_grasp_pose(
        GraspPose("demo_3", [-10.0, 5.0, 0.0], [0.08, 0.04, -0.015]),
        ur_tcp_pose=[0.48, 0.18, 0.31, -0.01, 0.01, 0.02],
    )
    
    print("✓ ダミーデータを追加しました（3サンプル）")
    
    # キャリブレーション実行
    T_sim2real, residual = calibrator.calibrate()
    
    # 検証
    print("\n検証結果:")
    stats = calibrator.validate()
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
    
    # 新しいpose の変換テスト
    print("\n新しいpose の変換例:")
    rotvec_deg_test = np.array([5.0, 2.5, 0.0])
    translation_m_test = np.array([0.11, 0.055, -0.022])
    ur_pose_test = calibrator.transform_pose(rotvec_deg_test, translation_m_test)
    print(f"  入力 (sim): rotvec={rotvec_deg_test}, trans={translation_m_test}")
    print(f"  出力 (UR): {ur_pose_test}")
    
    # 結果を保存（オプション）
    # calibrator.save_calibration("calibration_result.json")
