import numpy as np
from scipy.spatial.transform import Rotation as R

# Make sure numerical printouts use 4 decimal places and suppress scientific notation
np.set_printoptions(precision=4, suppress=True)

def pose_to_matrix(pose):
    """
    Converts a 6D pose (x, y, z, rx, ry, rz) to a 4x4 homogeneous transformation matrix.
    'pose' can be either a list or a 1D numpy array of length 6.
    """
    pose = np.asarray(pose, dtype=float)   # Convert to np.array for consistent handling
    if pose.shape != (6,):
        raise ValueError(f"pose must be a 6D vector, but got shape {pose.shape}")
    
    x, y, z, rx, ry, rz = pose
    rotation = R.from_euler('xyz', [rx, ry, rz], degrees=False).as_matrix()

    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rotation
    mat[:3, 3]  = [x, y, z]
    return mat

def matrix_to_pose(mat, return_as="list"):
    """
    Converts a 4x4 homogeneous transformation matrix to a 6D pose [x, y, z, rx, ry, rz].
    'return_as' can be "list" or "array" to control the output type.
    """
    if mat.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, but got shape {mat.shape}")

    x, y, z = mat[:3, 3]
    rotation = R.from_matrix(mat[:3, :3])
    rx, ry, rz = rotation.as_euler('xyz', degrees=False)

    pose = [x, y, z, rx, ry, rz]
    if return_as == "array":
        return np.array(pose, dtype=float)
    return pose

def delta_pose_matrix(matA, matB):
    """
    Computes the relative transform (delta) that takes matA -> matB, i.e. inv(A)*B.
    Both matA and matB should be 4x4.
    """
    return np.linalg.inv(matA) @ matB

def pose_to_next_pose(pose, delta_pose, return_as="list"):
    """
    Applies a delta_pose to an original pose.
      next_pose_matrix = pose_matrix * delta_pose_matrix
    Returns the resulting pose in the specified format ("list" or "array").
    """
    mat_pose  = pose_to_matrix(pose)
    mat_delta = pose_to_matrix(delta_pose)
    mat_next  = mat_pose @ mat_delta
    return matrix_to_pose(mat_next, return_as=return_as)

def print_pose(label, pose):
    """
    Helper for printing a 6D pose with NumPy precision settings.
    """
    arr = np.array(pose, dtype=float)
    print(f"{label} {arr}")

def print_matrix(label, mat):
    """
    Helper for printing a matrix with NumPy precision settings.
    """
    print(f"{label}\n{mat}\n")

if __name__ == "__main__":
    # 1) Define two poses A & B
    pose_a = [0.5, 0.0, 0.2, 0.1, 0.2, 0.3]  # list
    pose_b = np.array([0.7, 0.3, 0.4, 0.2, 0.1, 0.4])  # array

    print_pose("Pose A:", pose_a)
    print_pose("Pose B:", pose_b)

    # 2) Calculate the delta pose (A -> B) in the OLD (original) coordinate system
    matrix_a = pose_to_matrix(pose_a)
    matrix_b = pose_to_matrix(pose_b)
    delta_mat_old = delta_pose_matrix(matrix_a, matrix_b)
    delta_pose_old = matrix_to_pose(delta_mat_old, return_as="array")

    print_pose("\nDelta Pose (A -> B) in OLD coordinate system:", delta_pose_old)

    # 3) Apply the old delta pose to A to see if we recover B
    calculated_pose_b = pose_to_next_pose(pose_a, delta_pose_old, return_as="array")
    print_pose("Calculated Pose B' (using delta from old coords):", calculated_pose_b)

    pose_diff = np.allclose(pose_b, calculated_pose_b, atol=1e-5)
    if pose_diff:
        print("✅ Delta pose in old coords is correct: B' matches B")
    else:
        print("❌ Delta pose in old coords is incorrect: B' does NOT match B")

    # 4) Define a coordinate transform T (for the new frame):
    #    e.g., shift x=0.5 + rotate about Z by 0.5 rad
    transform_pose = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5] 
    T_matrix = pose_to_matrix(transform_pose)

    # 5) Transform A & B into the NEW coordinate frame => A' & B'
    matrix_a_new = T_matrix @ matrix_a
    matrix_b_new = T_matrix @ matrix_b
    pose_a_new = matrix_to_pose(matrix_a_new, return_as="array")
    pose_b_new = matrix_to_pose(matrix_b_new, return_as="array")

    print_pose("\nTransformed Pose A' (in new coords):", pose_a_new)
    print_pose("Transformed Pose B' (in new coords):", pose_b_new)

    # 6) Compute the NEW delta pose: (A' -> B') in new coords
    delta_mat_new = delta_pose_matrix(matrix_a_new, matrix_b_new)
    delta_pose_new = matrix_to_pose(delta_mat_new, return_as="array")
    print_pose("\nDelta Pose (A' -> B') in NEW coordinate system:", delta_pose_new)

    # 7) Apply the new delta to A' => check if we recover B'
    calculated_pose_b_new = pose_to_next_pose(pose_a_new, delta_pose_new, return_as="array")
    print_pose("Calculated B' from A' + delta_pose_new:", calculated_pose_b_new)

    pose_diff_new = np.allclose(pose_b_new, calculated_pose_b_new, atol=1e-5)
    if pose_diff_new:
        print("✅ Delta pose in new coords is correct: B' matches B' (transformed)")
    else:
        print("❌ Delta pose in new coords is incorrect: B' does NOT match B' (transformed)")

    # 8) Compare old vs new deltas
    print("\nCompare OLD delta vs NEW delta:")
    print_pose("  Old delta:", delta_pose_old)
    print_pose("  New delta:", delta_pose_new)
    if np.allclose(delta_pose_old, delta_pose_new, atol=1e-5):
        print("   → They are (unexpectedly) the SAME (only if T is identity or commutes).")
    else:
        print("   → They are DIFFERENT, as expected for different frames.")
