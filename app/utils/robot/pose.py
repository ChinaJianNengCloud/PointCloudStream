import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

class Pose:
    def __init__(self, position, quaternion):
        self.p = np.array(position)  # Position: 3D vector.
        self.q = np.array(quaternion)  # Quaternion: [w, x, y, z].
    @property
    def r(self):
        return R.from_quat(self.q, scalar_first=True)

    def inv(self):
        """Return the inverse of this pose."""
        rotation = R.from_quat(self.q, scalar_first=True)
        inv_rotation = rotation.inv()
        inv_position = -inv_rotation.apply(self.p)
        inv_quat = inv_rotation.as_quat(scalar_first=True)
        return Pose(inv_position, inv_quat)

    def __mul__(self, other):
        """Pose multiplication (composition of transformations)."""
        if not isinstance(other, Pose):
            raise TypeError("Multiplication is only supported between Pose objects.")
        # Compose rotations.
        new_rotation = self.r * other.r
        new_position = self.p + self.r.apply(other.p)
        return Pose(new_position, new_rotation.as_quat(scalar_first=True))

    def set_p(self, new_position):
        """Override the position of the pose."""
        self.p = np.array(new_position)

    def __repr__(self):
        return f"Pose(position={self.p}, quaternion={self.q})"

    def apply_delta_pose(self, delta_pose, on="base"):
        """
        Apply a delta transformation (given as a Pose) to the current pose.
        
        Parameters:
            delta_pose (Pose): The delta transformation.
            on (str): The frame in which the delta is applied.
                      Options:
                        - "base": Pre-multiply (delta is in the base frame).
                        - "ee": Post-multiply (delta is in the end-effector frame).
                        - "align": Use the rotation from (delta_pose * self) but override translation with self.p + delta_pose.p.
                        - "align2": Center the transformation using a translation-only pose.
                        
        Returns:
            Pose: The new pose after applying the delta.
        """
        if not isinstance(delta_pose, Pose):
            raise TypeError("Delta pose must be a Pose object.")
        on = on.lower()
        if on == "base":
            return delta_pose * self
        elif on == "ee":
            return self * delta_pose
        elif on == "align":
            new_pose = delta_pose * self
            new_pose.set_p(self.p + delta_pose.p)
            return new_pose
        elif on == "align3":
            new_pose = self * delta_pose
            new_pose.set_p(self.p + delta_pose.p)
            return new_pose
        elif on == "align2":
            translation_only = Pose(position=self.p, quaternion=[1, 0, 0, 0])
            return (translation_only * delta_pose * translation_only.inv()) * self
        else:
            raise ValueError("Invalid 'on' parameter. Must be one of 'base', 'ee', 'align', or 'align2'.")

    def cal_delta_pose(self, target_pose, on="base"):
        """
        Calculate the delta transformation between this pose (self) and a target pose.
        
        The delta is computed according to the frame specified by the 'on' parameter:
          - "base": Delta = target_pose * (self.inv())
          - "ee":   Delta = (self.inv()) * target_pose
          - "align": The rotation is as in "base", but the translation is simply target_pose.p - self.p.
          - "align2": The rotation is as in "ee", but the translation is simply target_pose.p - self.p.
        
        Parameters:
            target_pose (Pose): The target (or "next") pose.
            on (str): The frame in which the delta is computed.
        
        Returns:
            Pose: A Pose representing the delta transformation.
        """
        if not isinstance(target_pose, Pose):
            raise TypeError("target_pose must be a Pose instance.")
        on = on.lower()
        if on == "base":
            return target_pose * self.inv()
        elif on == "ee":
            return self.inv() * target_pose
        elif on == "align":
            delta = target_pose * self.inv()
            delta.set_p(target_pose.p - self.p)
            return delta
        elif on == "align2":
            delta = self.inv() * target_pose
            delta.set_p(target_pose.p - self.p)
            return delta
        else:
            raise ValueError("Invalid 'on' parameter. Must be one of 'base', 'ee', 'align', or 'align2'.")

    @staticmethod
    def from_1d_array(vector, vector_type="rotvec", degrees=False):
        """
        Create a Pose from a vector representation.
        
        Parameters:
            vector (array-like): For 'rotvec' or 'euler' types, expects 6 elements [x, y, z, r1, r2, r3].
                                 For 'quat', expects 7 elements [x, y, z, w, x, y, z].
            vector_type (str): The rotation representation in the vector: "rotvec", "quat", or "euler".
            degrees (bool): Whether Euler angles are in degrees (only used for 'euler').
        
        Returns:
            Pose: The resulting Pose object.
        """
        vector = np.array(vector)
        vt = vector_type.lower()
        if vt == "rotvec":
            if vector.size != 6:
                raise ValueError("For 'rotvec', the vector must have 6 elements.")
            position = vector[:3]
            rotvec = vector[3:6]
            quaternion = R.from_rotvec(rotvec).as_quat(scalar_first=True)
            return Pose(position, quaternion)
        elif vt == "quat":
            if vector.size != 7:
                raise ValueError("For 'quat', the vector must have 7 elements.")
            position = vector[:3]
            quaternion = vector[3:7]
            return Pose(position, quaternion)
        elif vt == "euler":
            if vector.size != 6:
                raise ValueError("For 'euler', the vector must have 6 elements.")
            position = vector[:3]
            euler_angles = vector[3:6]
            quaternion = R.from_euler("xyz", euler_angles, degrees=degrees).as_quat(scalar_first=True)
            return Pose(position, quaternion)
        else:
            raise ValueError("Unsupported vector_type. Use 'rotvec', 'quat', or 'euler'.")

    # @staticmethod
    def to_1d_array(self, vector_type="rotvec", degrees=False) -> np.ndarray:
        """
        Convert a Pose to a vector representation.
        
        Parameters:
            pose (Pose): The pose to convert.
            vector_type (str): The desired rotation representation ("rotvec", "quat", or "euler").
            degrees (bool): Whether Euler angles should be in degrees (only for 'euler').
        
        Returns:
            np.ndarray: The pose represented as a vector.
                      6 elements for 'rotvec' or 'euler', 7 elements for 'quat'.
        """
        vt = vector_type.lower()
        if vt == "rotvec":
            r = R.from_quat(self.q, scalar_first=True)
            rotvec = r.as_rotvec()
            return np.hstack((self.p, rotvec))
        elif vt == "quat":
            return np.hstack((self.p, self.q))
        elif vt == "euler":
            r = R.from_quat(self.q, scalar_first=True)
            euler_angles = r.as_euler("xyz", degrees=degrees)
            return np.hstack((self.p, euler_angles))
        else:
            raise ValueError("Unsupported vector_type. Use 'rotvec', 'quat', or 'euler'.")

def interpolate_poses_equal_distance(
    poses, 
    target_length=15, 
    method='quadratic', 
    force_original_points=False
):
    """
    Interpolate a list of robot poses [x, y, z, rx, ry, rz] so that:
      - (x, y, z) are sampled evenly in 3D distance.
      - Orientation is interpolated via spherical interpolation (SLERP).
      - 'method' is used for (x, y, z) position interpolation only
        (passed to interp1d). If you want to change rotation interpolation,
        you would manually switch from Slerp to e.g. RotationSpline.
      - If force_original_points is True, original distances are included
        exactly (except duplicates, which are removed).

    Args:
        poses (list of [float]): 
            Original poses, each = [x, y, z, rx, ry, rz].
            We assume rx, ry, rz are Euler angles in radians (xyz order).
        target_length (int): 
            Number of output poses desired (may differ if force_original_points).
        method (str): 
            Interpolation kind for position, e.g. 'linear', 'quadratic', 'cubic'.
        force_original_points (bool):
            If True, ensures all original pose distances are included 
            exactly in the final result (except duplicates).

    Returns:
        list of list of float:
            Interpolated poses in same format [x, y, z, rx, ry, rz].
    """
    n = len(poses)
    if n < 2:
        # If only one (or zero) poses, no interpolation is meaningful.
        return poses

    # Convert to NumPy arrays
    arr = np.array(poses, dtype=float)  # shape (n, 6)
    xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
    r_euler = arr[:, 3:6]  # [rx, ry, rz] as Euler angles

    # -- 1) Compute cumulative 3D distance for positions (x,y,z)
    distances = [0.0]
    for i in range(n - 1):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        dz = zs[i+1] - zs[i]
        segment_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        distances.append(distances[-1] + segment_dist)
    distances = np.array(distances)
    total_dist = distances[-1]

    if total_dist == 0.0:
        return [poses[0] for _ in range(target_length)]

    def remove_duplicates(dist_array, *value_arrays):
        """
        Keep only entries where dist_array[i] != dist_array[i-1].
        Preserves order, returns arrays with duplicates removed.
        """
        new_dists = [dist_array[0]]
        new_values = [[v[0]] for v in value_arrays]
        for i in range(1, len(dist_array)):
            if not np.isclose(dist_array[i], new_dists[-1]):
                new_dists.append(dist_array[i])
                for arr_idx, v in enumerate(value_arrays):
                    new_values[arr_idx].append(v[i])
        # Convert final lists back to np.array
        new_dists = np.array(new_dists)
        new_values = [np.array(vals) for vals in new_values]
        return (new_dists, *new_values)

    distances, xs, ys, zs, r_euler = remove_duplicates(
        distances, xs, ys, zs, r_euler
    )

    # If everything collapsed to 1 unique point:
    if len(distances) < 2:
        # Just replicate the single pose
        single_pose = [xs[0], ys[0], zs[0], r_euler[0,0], r_euler[0,1], r_euler[0,2]]
        return [single_pose for _ in range(target_length)]
    
    total_dist = distances[-1]  # re-check after removing duplicates

    # -- 3) Set up interpolation for position
    f_x = interp1d(distances, xs, kind=method)
    f_y = interp1d(distances, ys, kind=method)
    f_z = interp1d(distances, zs, kind=method)

    # -- 4) Set up interpolation for orientation
    #       We'll do a spherical interpolation (SLERP) on rotations
    #       First, convert Euler angles -> Rotation objects
    rotations = R.from_euler('xyz', r_euler, degrees=False)

    # Create the slerp object
    #   Slerp requires strictly increasing times => 'distances' is our "time"
    slerp = Slerp(distances, rotations)

    # -- 5) Determine sample distances
    if force_original_points:
        # Merge original unique distances with a uniform sample
        num_new = max(target_length - len(distances), 1)
        uniform_distances = np.linspace(0, total_dist, num_new)
        combined = np.concatenate([distances, uniform_distances])
        sample_distances = np.unique(np.sort(combined))
    else:
        # Simply take a uniform sampling from 0 to total_dist
        sample_distances = np.linspace(0, total_dist, target_length)

    # -- 6) Evaluate each interpolation at sample distances
    xnew  = f_x(sample_distances)
    ynew  = f_y(sample_distances)
    znew  = f_z(sample_distances)

    # Slerp orientation
    Rnew  = slerp(sample_distances)  # Rotation objects
    # Convert back to Euler angles (xyz order)
    euler_new = Rnew.as_euler('xyz', degrees=False)

    # -- 7) Combine results
    result = []
    for i in range(len(sample_distances)):
        result.append([
            xnew[i],
            ynew[i],
            znew[i],
            euler_new[i, 0],
            euler_new[i, 1],
            euler_new[i, 2]
        ])
    
    return result

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    # Define a current pose.
    quat = R.from_euler("xyz", [0.1, 0.2, 0.3], degrees=False).as_quat(scalar_first=True)
    current_pose = Pose(position=[1, 2, 3], quaternion=quat)
    
    # Define a set of delta poses for testing (as translation and rotation vector).
    delta_poses = [
        (np.array([0.1, 0, 0]), np.array([0, 0, np.pi/6])),  # Translate X, rotate about Z.
        (np.array([0, 0.2, 0]), np.array([np.pi/4, 0, 0])),    # Translate Y, rotate about X.
        (np.array([0, 0, 0.3]), np.array([0, np.pi/3, 0])),    # Translate Z, rotate about Y.
    ]
    
    # Define frame types for testing.
    frame_types = ["base", "ee", "align", "align2"]
    
    for i, (delta_pos, delta_rot) in enumerate(delta_poses):
        print(f"\nTest {i+1}:")
        print(f"Delta Position: {delta_pos}\nDelta R (rotvec): {delta_rot}")
        
        # Create a delta_pose from the delta_pos and delta_rot using a 6-element vector (rotvec).
        delta_vector = np.hstack((delta_pos, delta_rot))
        delta_pose = Pose.from_1d_array(delta_vector, vector_type="rotvec")
        
        # For each frame type, first compute the next pose using apply_delta_pose,
        # then recover the delta transformation using cal_delta_pose.
        for frame_type in frame_types:
            print(f"\nFrame Type: {frame_type}")
            next_pose = current_pose.apply_delta_pose(delta_pose, on=frame_type)
            
            # Compute the delta as a Pose.
            recovered_delta_pose = current_pose.cal_delta_pose(next_pose, on=frame_type)
            # Convert the recovered delta to a vector (using rotvec) for easy comparison.
            recovered_delta_vector = Pose.to_1d_array(recovered_delta_pose, vector_type="rotvec")
            
            print(f"Next Pose: {next_pose}")
            print(f"Recovered Delta Pose: {recovered_delta_pose}")
            print(f"Recovered Delta Vector (rotvec): {recovered_delta_vector}")
            print(f"Original Delta Vector (rotvec): {delta_vector}")
            print(f"Match: {np.allclose(recovered_delta_vector, delta_vector, atol=1e-6)}")