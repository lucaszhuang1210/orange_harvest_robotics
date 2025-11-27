"""Vision Pipeline for Citrus Detection and Target Pose Generation

This module implements the vision processing pipeline:
1. Segmentation (RGB → mask)
2. Depth projection (mask + depth → 3D points in camera frame)
3. Coordinate transformation (camera frame → robot base frame)
4. Target pose generation (3D points → 6-DOF target)
"""

import torch
import numpy as np
from typing import Tuple, Optional


class VisionPipeline:
    """Vision processing pipeline for citrus detection."""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        # TODO: Initialize segmentation model here
        # self.segmentation_model = load_sam_model() or custom model
        
    def process_fixed_camera(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_quat_w: torch.Tensor,
        robot_base_pos_w: torch.Tensor,
        robot_base_quat_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Process fixed camera images to find target pose in robot base frame.
        
        Args:
            rgb_image: RGB image [H, W, 3], values in [0, 1]
            depth_image: Depth image [H, W], values in meters
            camera_intrinsics: Camera K matrix [3, 3]
            camera_pos_w: Camera position in world frame [3]
            camera_quat_w: Camera orientation in world frame [4] (w, x, y, z)
            robot_base_pos_w: Robot base position in world frame [3]
            robot_base_quat_w: Robot base orientation in world frame [4]
            
        Returns:
            target_pos_b: Target position in robot base frame [3]
            target_quat_b: Target orientation in robot base frame [4]
            success: Whether detection was successful
        """
        # Step 1: Run segmentation
        orange_mask = self._segment_orange(rgb_image)
        
        if orange_mask.sum() == 0:
            # No orange detected
            return torch.zeros(3), torch.tensor([1, 0, 0, 0]), False
        
        # Step 2: Project mask to 3D points in camera frame
        points_3d_cam = self._project_mask_to_3d(
            orange_mask, depth_image, camera_intrinsics
        )
        
        if points_3d_cam.shape[0] == 0:
            return torch.zeros(3), torch.tensor([1, 0, 0, 0]), False
        
        # Step 3: Get target point (top of orange)
        target_point_cam = self._get_orange_top(points_3d_cam)
        
        # Step 4: Transform from camera frame to world frame
        target_point_w = self._transform_camera_to_world(
            target_point_cam, camera_pos_w, camera_quat_w
        )
        
        # Step 5: Transform from world frame to robot base frame
        target_pos_b = self._transform_world_to_base(
            target_point_w, robot_base_pos_w, robot_base_quat_w
        )
        
        # Step 6: Generate target orientation (gripper facing down)
        target_quat_b = self._generate_cutting_orientation(target_pos_b)
        
        return target_pos_b, target_quat_b, True
    
    def _segment_orange(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Segment orange from RGB image.
        
        TODO: Replace with actual segmentation model (SAM, Grounded-SAM, custom CNN)
        
        Args:
            rgb_image: [H, W, 3]
            
        Returns:
            mask: Binary mask [H, W] where True = orange
        """
        # PLACEHOLDER: Simple color-based segmentation
        # Orange color: high red, medium green, low blue
        r, g, b = rgb_image[..., 0], rgb_image[..., 1], rgb_image[..., 2]
        
        # Thresholds for orange color
        mask = (r > 0.7) & (g > 0.3) & (g < 0.7) & (b < 0.3)
        
        return mask
    
    def _project_mask_to_3d(
        self,
        mask: torch.Tensor,
        depth_image: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        """Project 2D mask + depth to 3D points in camera frame.
        
        Args:
            mask: Binary mask [H, W]
            depth_image: Depth values [H, W]
            K: Camera intrinsic matrix [3, 3]
            
        Returns:
            points_3d: 3D points [N, 3] in camera frame
        """
        # Get pixel coordinates where mask is True
        v_coords, u_coords = torch.where(mask)  # v = row (y), u = col (x)
        
        # Get corresponding depth values
        depth_values = depth_image[v_coords, u_coords]
        
        # Remove invalid depths (zero or too far)
        valid = (depth_values > 0.01) & (depth_values < 10.0)
        u_coords = u_coords[valid]
        v_coords = v_coords[valid]
        depth_values = depth_values[valid]
        
        if len(depth_values) == 0:
            return torch.empty((0, 3), device=mask.device)
        
        # Unproject to 3D
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        x = (u_coords.float() - cx) * depth_values / fx
        y = (v_coords.float() - cy) * depth_values / fy
        z = depth_values
        
        points_3d = torch.stack([x, y, z], dim=-1)
        
        return points_3d
    
    def _get_orange_top(self, points_3d: torch.Tensor) -> torch.Tensor:
        """Extract target point from orange point cloud.
        
        Args:
            points_3d: Point cloud [N, 3]
            
        Returns:
            target_point: Single target point [3]
        """
        # Method 1: Highest point (top of orange)
        highest_idx = torch.argmax(points_3d[:, 2])
        target_point = points_3d[highest_idx]
        
        # Method 2: Centroid (alternative)
        # target_point = points_3d.mean(dim=0)
        
        # Method 3: RANSAC sphere fitting + top point (most robust)
        # target_point = fit_sphere_and_get_top(points_3d)
        
        return target_point
    
    def _transform_camera_to_world(
        self,
        point_cam: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Transform point from camera frame to world frame.
        
        Transformation: p_world = R_cam * p_cam + t_cam
        
        Args:
            point_cam: Point in camera frame [3]
            camera_pos_w: Camera position in world [3]
            camera_quat_w: Camera orientation in world [4] (w, x, y, z)
            
        Returns:
            point_w: Point in world frame [3]
        """
        from isaaclab.utils.math import quat_apply
        
        # Rotate point by camera orientation
        point_w = quat_apply(camera_quat_w, point_cam)
        
        # Translate by camera position
        point_w = point_w + camera_pos_w
        
        return point_w
    
    def _transform_world_to_base(
        self,
        point_w: torch.Tensor,
        base_pos_w: torch.Tensor,
        base_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Transform point from world frame to robot base frame.
        
        Transformation: p_base = R_base^T * (p_world - t_base)
        
        Args:
            point_w: Point in world frame [3]
            base_pos_w: Robot base position in world [3]
            base_quat_w: Robot base orientation in world [4]
            
        Returns:
            point_b: Point in robot base frame [3]
        """
        from isaaclab.utils.math import quat_apply, quat_inv
        
        # Translate to base frame origin
        point_translated = point_w - base_pos_w
        
        # Rotate by inverse of base orientation
        base_quat_inv = quat_inv(base_quat_w)
        point_b = quat_apply(base_quat_inv, point_translated)
        
        return point_b
    
    def _generate_cutting_orientation(self, target_pos_b: torch.Tensor) -> torch.Tensor:
        """Generate target orientation for cutting.
        
        For now: Simple downward-facing gripper.
        TODO: Compute from stem direction for perpendicular cutting.
        
        Args:
            target_pos_b: Target position in base frame [3]
            
        Returns:
            target_quat_b: Target orientation in base frame [4] (w, x, y, z)
        """
        from isaaclab.utils.math import quat_from_euler_xyz
        
        # Gripper facing downward: -90 degrees around Y axis
        target_quat_b = quat_from_euler_xyz(
            torch.tensor([0.0], device=target_pos_b.device),
            torch.tensor([-np.pi / 2], device=target_pos_b.device),
            torch.tensor([0.0], device=target_pos_b.device),
        )
        
        return target_quat_b.squeeze()


class AdvancedVisionPipeline(VisionPipeline):
    """Extended vision pipeline with stem detection and fine-tuning."""
    
    def detect_stem_in_hand_camera(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Detect stem in hand camera for fine-tuning.
        
        Args:
            rgb_image: Hand camera RGB [H, W, 3]
            depth_image: Hand camera depth [H, W]
            
        Returns:
            stem_center_pixel: Stem center in image [2] (u, v)
            stem_depth: Average depth of stem [1]
            detected: Whether stem was found
        """
        # Segment stem (green, thin)
        stem_mask = self._segment_stem(rgb_image)
        
        if stem_mask.sum() == 0:
            return torch.zeros(2), torch.zeros(1), False
        
        # Get centroid
        v_coords, u_coords = torch.where(stem_mask)
        stem_center_u = u_coords.float().mean()
        stem_center_v = v_coords.float().mean()
        stem_center = torch.tensor([stem_center_u, stem_center_v])
        
        # Get average depth
        stem_depth = depth_image[stem_mask].mean()
        
        return stem_center, stem_depth, True
    
    def _segment_stem(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Segment stem from RGB image.
        
        Args:
            rgb_image: [H, W, 3]
            
        Returns:
            mask: Binary mask [H, W]
        """
        # PLACEHOLDER: Color-based segmentation
        r, g, b = rgb_image[..., 0], rgb_image[..., 1], rgb_image[..., 2]
        
        # Green stem: low red, high green, low blue
        mask = (r < 0.3) & (g > 0.3) & (g < 0.7) & (b < 0.3)
        
        return mask
    
    def compute_visual_servoing_correction(
        self,
        stem_center_pixel: torch.Tensor,
        stem_depth: torch.Tensor,
        image_size: Tuple[int, int],
        K: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pose correction to center stem in image.
        
        Args:
            stem_center_pixel: Stem center [2] (u, v)
            stem_depth: Stem depth [1]
            image_size: (width, height)
            K: Camera intrinsics [3, 3]
            
        Returns:
            correction: Pose correction [6] (dx, dy, dz, droll, dpitch, dyaw)
        """
        width, height = image_size
        image_center = torch.tensor([width / 2, height / 2], device=stem_center_pixel.device)
        
        # Pixel error
        error_u = stem_center_pixel[0] - image_center[0]
        error_v = stem_center_pixel[1] - image_center[1]
        
        # Convert to metric error using depth
        fx = K[0, 0]
        fy = K[1, 1]
        
        # Position correction (in camera frame)
        error_x = error_u * stem_depth / fx
        error_y = error_v * stem_depth / fy
        
        # Depth correction (target: 10cm from gripper)
        target_depth = 0.10
        error_z = stem_depth - target_depth
        
        # Proportional control gains
        k_xy = 0.5
        k_z = 0.3
        
        correction = torch.zeros(6, device=stem_center_pixel.device)
        correction[0] = error_x * k_xy
        correction[1] = error_y * k_xy
        correction[2] = error_z * k_z
        # No orientation correction for now
        
        return correction


# Example usage
def example_usage():
    """Example of using the vision pipeline."""
    import torch
    
    device = "cuda:0"
    pipeline = VisionPipeline(device=device)
    
    # Dummy data
    rgb_image = torch.rand(480, 640, 3, device=device)
    depth_image = torch.ones(480, 640, device=device) * 0.8  # 80cm depth
    
    # Camera intrinsics (computed from config)
    K = torch.tensor([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0]
    ], device=device)
    
    # Camera pose (world frame)
    camera_pos_w = torch.tensor([0.3, 0.0, 1.5], device=device)
    camera_quat_w = torch.tensor([0.7071, 0.0, 0.7071, 0.0], device=device)  # Looking down
    
    # Robot base pose (world frame)
    robot_base_pos_w = torch.tensor([0.0, 0.0, 0.0], device=device)
    robot_base_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # Identity
    
    # Process
    target_pos_b, target_quat_b, success = pipeline.process_fixed_camera(
        rgb_image,
        depth_image,
        K,
        camera_pos_w,
        camera_quat_w,
        robot_base_pos_w,
        robot_base_quat_w,
    )
    
    if success:
        print(f"Target position (base frame): {target_pos_b}")
        print(f"Target orientation (base frame): {target_quat_b}")
    else:
        print("Detection failed")


if __name__ == "__main__":
    example_usage()