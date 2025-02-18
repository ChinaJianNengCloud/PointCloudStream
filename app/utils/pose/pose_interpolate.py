import numpy as np
from scipy.interpolate import interp1d

def interpolate_joint_positions_equal_distance(
    joint_positions, 
    target_length=15, 
    method='cubic', 
    force_original_points=False
):
    """
    Interpolate a list of joint positions [joint_1, joint_2, ..., joint_6] so that:
      - The joint positions are sampled evenly in terms of distance between joint configurations.
      - The interpolation method can be 'linear', 'quadratic', 'cubic', or any
        valid 'kind' option in scipy.interpolate.interp1d'.
      - If force_original_points is True, we ensure the original joint distances
        are included exactly in the sampled set (though duplicates are still removed
        to avoid interpolation errors).

    Args:
        joint_positions (list of list of float): 
            Original joint positions, each = [joint_1, joint_2, ..., joint_6].
        target_length (int): 
            Number of output joint configurations desired.
        method (str): 
            Interpolation kind, e.g. 'linear', 'quadratic', 'cubic'.
            (Passed to `interp1d(kind=...)`.)
        force_original_points (bool):
            If True, ensures all original joint positions' distances are included 
            exactly in the final result (except duplicates, which must be removed).

    Returns:
        list of list of float:
            A new list of interpolated joint positions, length = target_length 
            unless force_original_points is used (which may alter it).
    """
    n = len(joint_positions)
    if n < 2:
        # If only one (or zero) joint positions, no interpolation needed
        return joint_positions
    
    # Convert joint positions to NumPy arrays
    arr = np.array(joint_positions, dtype=float)
    joints = arr.T

    distances = [0.0]
    for i in range(n - 1):
        dist = np.linalg.norm(joints[:, i+1] - joints[:, i])  # Euclidean distance between joint configurations
        distances.append(distances[-1] + dist)
    distances = np.array(distances)
    total_dist = distances[-1]
    if total_dist == 0.0:
        return [joint_positions[0] for _ in range(target_length)]
    
    def remove_duplicates(dist_array, *value_arrays):
        new_dist = [dist_array[0]]
        new_values = [[v[0]] for v in value_arrays]

        for i in range(1, len(dist_array)):
            if not np.isclose(dist_array[i], new_dist[-1]):
                new_dist.append(dist_array[i])
                for arr_idx, v in enumerate(value_arrays):
                    new_values[arr_idx].append(v[i])

        new_dist = np.array(new_dist)
        new_values = [np.array(vals) for vals in new_values]
        return (new_dist, *new_values)

    distances, *joints = remove_duplicates(distances, *joints)

    if len(distances) < 2:
        return [joints[0].tolist() for _ in range(target_length)]
    
    total_dist = distances[-1]
    f_joints = [interp1d(distances, joint, kind=method) for joint in joints]
    if force_original_points:
        num_new = max(target_length - len(distances), 1)
        uniform_distances = np.linspace(0, total_dist, num_new)
        combined = np.concatenate([distances, uniform_distances])
        sample_distances = np.unique(np.sort(combined))
    else:
        sample_distances = np.linspace(0, total_dist, target_length)

    interpolated_joints = [f(sample_distances) for f in f_joints]

    result = []
    for i in range(len(sample_distances)):
        result.append([interpolated_joint[i] for interpolated_joint in interpolated_joints])
    
    return result
