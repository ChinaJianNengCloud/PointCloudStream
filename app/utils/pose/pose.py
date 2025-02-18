import numpy as np
from scipy.spatial.transform import Rotation as R
from vtkmodules.vtkCommonMath import vtkMatrix4x4

class Pose:
    def __init__(self, position, quaternion):
        self.p = np.array(position)  # Position: 3D vector.
        self.q = np.array(quaternion)  # Quaternion: [w, x, y, z].
    @property
    def r(self):
        return R.from_quat(self.q, scalar_first=True)
    
    @staticmethod
    def from_matrix(matrix: np.ndarray) -> "Pose":
        assert matrix.shape == (4, 4)
        return Pose(matrix[:3, 3], R.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True))
        
    @property
    def matrix(self):
        """Return the 4x4 homogeneous transformation matrix."""
        rot_matrix = self.r.as_matrix()
        homo_matrix = np.eye(4)
        homo_matrix[:3, :3] = rot_matrix
        homo_matrix[:3, 3] = self.p
        return homo_matrix
    
    @property
    def vtk_matrix(self) -> vtkMatrix4x4:
        vtk_matrix = vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, self.matrix[i, j])
        return vtk_matrix
    
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


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    # Define a current pose.
    quat = R.from_euler("xyz", [0.1, 0.2, 0.3], degrees=False).as_quat(scalar_first=True)
    current_pose = Pose(position=[1, 2, 3], quaternion=quat)
    print(current_pose)
    print(Pose.from_1d_array(np.array([1, 2, 3, 0.1, 0.2, 0.3]), vector_type="euler"))
    # print(current_pose.matrix)
    # print(current_pose.vtk_matrix)
    # # Define a set of delta poses for testing (as translation and rotation vector).
    # delta_poses = [
    #     (np.array([0.1, 0, 0]), np.array([0, 0, np.pi/6])),  # Translate X, rotate about Z.
    #     (np.array([0, 0.2, 0]), np.array([np.pi/4, 0, 0])),    # Translate Y, rotate about X.
    #     (np.array([0, 0, 0.3]), np.array([0, np.pi/3, 0])),    # Translate Z, rotate about Y.
    # ]
    
    # # Define frame types for testing.
    # frame_types = ["base", "ee", "align", "align2"]
    
    # for i, (delta_pos, delta_rot) in enumerate(delta_poses):
    #     print(f"\nTest {i+1}:")
    #     print(f"Delta Position: {delta_pos}\nDelta R (rotvec): {delta_rot}")
        
    #     # Create a delta_pose from the delta_pos and delta_rot using a 6-element vector (rotvec).
    #     delta_vector = np.hstack((delta_pos, delta_rot))
    #     delta_pose = Pose.from_1d_array(delta_vector, vector_type="rotvec")
        
    #     # For each frame type, first compute the next pose using apply_delta_pose,
    #     # then recover the delta transformation using cal_delta_pose.
    #     for frame_type in frame_types:
    #         print(f"\nFrame Type: {frame_type}")
    #         next_pose = current_pose.apply_delta_pose(delta_pose, on=frame_type)
            
    #         # Compute the delta as a Pose.
    #         recovered_delta_pose = current_pose.cal_delta_pose(next_pose, on=frame_type)
    #         # Convert the recovered delta to a vector (using rotvec) for easy comparison.
    #         recovered_delta_vector = Pose.to_1d_array(recovered_delta_pose, vector_type="rotvec")
            
    #         print(f"Next Pose: {next_pose}")
    #         print(f"Recovered Delta Pose: {recovered_delta_pose}")
    #         print(f"Recovered Delta Vector (rotvec): {recovered_delta_vector}")
    #         print(f"Original Delta Vector (rotvec): {delta_vector}")
    #         print(f"Match: {np.allclose(recovered_delta_vector, delta_vector, atol=1e-6)}")
