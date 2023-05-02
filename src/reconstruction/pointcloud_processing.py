import open3d as o3d
import numpy as np

class PointCloudProcessor:
    """Class for processing and refining point clouds."""
    
    @staticmethod
    def downsample(point_cloud, voxel_size=0.01):
        """
        Downsample a point cloud using voxel grid filtering.
        
        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud.
            voxel_size (float): Voxel size for downsampling.
            
        Returns:
            open3d.geometry.PointCloud: Downsampled point cloud.
        """
        return point_cloud.voxel_down_sample(voxel_size)

    @staticmethod
    def remove_statistical_outliers(point_cloud, nb_neighbors=20, std_ratio=2.0):
        """
        Remove statistical outliers from a point cloud.
        
        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud.
            nb_neighbors (int): Number of neighbors to analyze for each point.
            std_ratio (float): Standard deviation ratio for outlier removal.
            
        Returns:
            tuple: Filtered point cloud, index of inlier points.
        """
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                         std_ratio=std_ratio)
        return cl, ind

    @staticmethod
    def remove_radius_outliers(point_cloud, nb_points=16, radius=0.05):
        """
        Remove radius outliers from a point cloud.
        
        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud.
            nb_points (int): Minimum number of points within the radius.
            radius (float): Radius for neighbor search.
            
        Returns:
            tuple: Filtered point cloud, index of inlier points.
        """
        cl, ind = point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return cl, ind

    @staticmethod
    def crop_point_cloud(point_cloud, min_bound, max_bound):
        """
        Crop a point cloud to an axis-aligned bounding box.
        
        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud.
            min_bound (list or numpy.ndarray): Minimum coordinates [x_min, y_min, z_min].
            max_bound (list or numpy.ndarray): Maximum coordinates [x_max, y_max, z_max].
            
        Returns:
            open3d.geometry.PointCloud: Cropped point cloud.
        """
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        return point_cloud.crop(aabb)

    @staticmethod
    def segment_plane(point_cloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        """
        Segment the largest plane (e.g., floor, table) from a point cloud using RANSAC.
        
        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud.
            distance_threshold (float): Maximum distance a point can be from the plane model.
            ransac_n (int): Number of points to sample for RANSAC.
            num_iterations (int): Number of RANSAC iterations.
            
        Returns:
            tuple: Inlier point cloud (plane), outlier point cloud (rest), plane model [a, b, c, d].
        """
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        inlier_cloud = point_cloud.select_by_index(inliers)
        outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
        return inlier_cloud, outlier_cloud, plane_model

    @staticmethod
    def cluster_dbscan(point_cloud, eps=0.02, min_points=10):
        """
        Cluster a point cloud using DBSCAN.
        
        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud.
            eps (float): Density parameter for DBSCAN.
            min_points (int): Minimum number of points required to form a cluster.
            
        Returns:
            tuple: Point cloud with labels, number of clusters.
        """
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        max_label = labels.max()
        print(f"Point cloud has {max_label + 1} clusters")
        point_cloud.colors = o3d.utility.Vector3dVector(np.zeros((len(labels), 3)))
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0 # Black for noise
        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        return point_cloud, max_label + 1

# Note: The cluster_dbscan method uses matplotlib for colormap, which might need to be imported.
# Add `import matplotlib.pyplot as plt` at the top if you plan to use it.