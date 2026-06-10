import torch
import math
from torch.nn import functional as F
import numpy as np
from loguru import logger as loguru
from typing import Tuple


def compute_bbox(points, dim=0):
    """
    Compute the bounding box of a point cloud.
    
    Args:
        points: Point cloud tensor of shape (..., N, D) where N is number of points
               and D is dimension (typically 3 for 3D points)
        dim: Dimension along which to compute the bounding box (default=0)
    
    Returns:
        Min and max coordinates of the bounding box with shape (..., 2, D)
    """
    min_coords, _ = torch.min(points, dim=dim, keepdim=True)
    max_coords, _ = torch.max(points, dim=dim, keepdim=True)
    return torch.cat([min_coords, max_coords], dim=dim)

def compute_centroid(points, dim=0):
    """
    Compute the centroid (geometric center) of a point cloud.
    
    Args:
        points: Point cloud tensor of shape (..., N, D) where N is number of points
               and D is dimension (typically 3 for 3D points)
        dim: Dimension along which to compute the mean (default=0)
    
    Returns:
        Centroid of the point cloud with shape (..., D)
    """
    return torch.mean(points, dim=dim)


def bounding_sphere_and_radius(points) -> tuple[np.ndarray, float]:
    """
    Return bounding sphere center and radius of a point cloud shape (N, 3)
    Uses Welzl's algorithm for minimum enclosing sphere (simplified version)
    """
    points = np.asarray(points)
    if points.shape[0] <= 4:
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        center = (bbox_min + bbox_max) / 2.0
        distances = np.linalg.norm(points - center, axis=1)
        radius = np.max(distances)
        return center, radius

    center = np.mean(points, axis=0)
    for _ in range(10):
        distances = np.linalg.norm(points - center, axis=1)
        max_dist_idx = np.argmax(distances)
        max_dist = distances[max_dist_idx]

        farthest_point = points[max_dist_idx]
        distances_from_farthest = np.linalg.norm(points - farthest_point, axis=1)
        second_farthest_idx = np.argmax(distances_from_farthest)
        second_farthest_point = points[second_farthest_idx]

        new_center = (farthest_point + second_farthest_point) / 2.0
        new_distances = np.linalg.norm(points - new_center, axis=1)
        new_max_dist = np.max(new_distances)

        if new_max_dist < max_dist:
            center = new_center
        else:
            break

    distances = np.linalg.norm(points - center, axis=1)
    radius = np.max(distances)
    return center, radius

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        Rotation matrices as tensor of shape (3, 3).
    """
    quat = axis_angle_to_quaternion(axis_angle.unsqueeze(0))
    return quaternion_to_matrix(quat).squeeze(0)


def rmat_to_cont_6d(matrix):
    """
    :param matrix (*, 3, 3)
    :returns 6d vector (*, 6)
    """
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def cont_6d_to_rmat(cont_6d):
    """
    :param 6d vector (*, 6)
    :returns matrix (*, 3, 3)
    """
    x1 = cont_6d[..., 0:3]
    y1 = cont_6d[..., 3:6]

    x = F.normalize(x1, dim=-1)
    y = F.normalize(y1 - (y1 * x).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)

def normalize_verts(verts : torch.Tensor) -> torch.Tensor:
    """
    Normalize the vertices of a mesh to be in the range [-1, 1].
    Args:
        verts: Tensor of shape (N, 3) representing the vertices of the mesh.
    Returns:
        Normalized vertices tensor of shape (N, 3).
    """
    center = torch.mean(verts, dim=0)
    vc = verts - center
    normalized_verts = vc / torch.max(torch.abs(vc))
    return normalized_verts

def homo_mat_inverse(trans_12):
    r"""Function that inverts a 4x4 homogeneous transformation
    :math:`T_1^{2} = \begin{bmatrix} R_1 & t_1 \\ \mathbf{0} & 1 \end{bmatrix}`

    The inverse transformation is computed as follows:

    .. math::

        T_2^{1} = (T_1^{2})^{-1} = \begin{bmatrix} R_1^T & -R_1^T t_1 \\
        \mathbf{0} & 1\end{bmatrix}

    Args:
        trans_12 (torch.Tensor): transformation tensor of shape
          :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: tensor with inverted transformations.

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`

    Example:
        >>> trans_12 = torch.rand(1, 4, 4)  # Nx4x4
        >>> trans_21 = tgm.homo_mat_inverse(trans_12)  # Nx4x4
    """
    if not torch.is_tensor(trans_12):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(trans_12)))
    if not trans_12.dim() in (2, 3) and trans_12.shape[-2:] == (4, 4):
        raise ValueError("Input size must be a Nx4x4 or 4x4. Got {}"
                         .format(trans_12.shape))
    # unpack input tensor
    rmat_12: torch.Tensor = trans_12[..., :3, 0:3]  # Nx3x3
    tvec_12: torch.Tensor = trans_12[..., :3, 3:4]  # Nx3x1

    # compute the actual inverse
    rmat_21: torch.Tensor = torch.transpose(rmat_12, -1, -2)
    tvec_21: torch.Tensor = torch.matmul(-rmat_21, tvec_12)

    # pack to output tensor
    trans_21: torch.Tensor = torch.zeros_like(trans_12)
    trans_21[..., :3, 0:3] += rmat_21
    trans_21[..., :3, -1:] += tvec_21
    trans_21[..., -1, -1:] += 1.0
    return trans_21


def dq_mul(dq1, dq2):
    """
    Multiply dual quaternion dq1 with dq2.
    Expects two equally-sized tensors of shape [*, 8], where * denotes any number of dimensions.
    Returns dq1*dq2 as a tensor of shape [*, 8].
    """
    assert dq1.shape[-1] == 8
    assert dq2.shape[-1] == 8

    dq1_r, dq1_d = torch.split(dq1, [4, 4], dim=-1)
    dq2_r, dq2_d = torch.split(dq2, [4, 4], dim=-1)

    dq_prod_r = q_mul(dq1_r, dq2_r)
    dq_prod_d = q_mul(dq1_r, dq2_d) + q_mul(dq1_d, dq2_r)
    dq_prod = torch.cat([dq_prod_r, dq_prod_d], dim=-1)
    return dq_prod


def dq_translation(dq):
    """
    Returns the translation component of the input dual quaternion tensor of shape [*, 8].
    Translation is returned as tensor of shape [*, 3].
    """
    assert dq.shape[-1] == 8

    dq_r, dq_d = torch.split(dq, [4, 4], dim=-1)
    mult = q_mul((2.0 * dq_d), q_conjugate(dq_r))
    return mult[..., 1:]


def dq_normalize(dq):
    """
    Normalize the coefficients of a given dual quaternion tensor of shape [*, 8].
    """
    assert dq.shape[-1] == 8

    dq_r = dq[..., :4]
    norm = torch.sqrt(torch.sum(torch.square(dq_r), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=dq.device)))  # check for singularities
    return torch.div(dq, norm[:, None])  # dq_norm = dq / ||q|| = dq_r / ||dq_r|| + dq_d / ||dq_r||


def dq_quaternion_conjugate(dq):
    """
    Returns the quaternion conjugate of the input dual quaternion tensor of shape [*, 8].
    The quaternion conjugate is composed of the complex conjugates of the real and the dual quaternion.
    """

    assert dq.shape[-1] == 8

    conj = torch.tensor([1, -1, -1, -1, 1, -1, -1, -1], device=dq.device)  # multiplication coefficients per element
    return dq * conj.expand_as(dq)


def q_mul(q1, q2):
    """
    Multiply quaternion q1 with q2.
    Expects two equally-sized tensors of shape [*, 4], where * denotes any number of dimensions.
    Returns q1*q2 as a tensor of shape [*, 4].
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4
    original_shape = q1.shape

    # Compute outer product
    terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    q1q2 = torch.stack((w, x, y, z), dim=1).view(original_shape)
    return q1q2


def wrap_angle(theta):
    """
    Helper method: Wrap the angles of the input tensor to lie between -pi and pi.
    Odd multiples of pi are wrapped to +pi (as opposed to -pi).
    """
    pi_tensor = torch.ones_like(theta, device=theta.device) * math.pi
    result = ((theta + pi_tensor) % (2 * pi_tensor)) - pi_tensor
    result[result.eq(-pi_tensor)] = math.pi

    return result


def q_angle(q):
    """
    Determine the rotation angle of given quaternion tensors of shape [*, 4].
    Return as tensor of shape [*, 1]
    """
    assert q.shape[-1] == 4

    q = q_normalize(q)
    q_re, q_im = torch.split(q, [1, 3], dim=-1)
    norm = torch.linalg.norm(q_im, dim=-1).unsqueeze(dim=-1)
    angle = 2.0 * torch.atan2(norm, q_re)

    return angle  # wrap_angle(angle) !!! very careful


def q_normalize(q):
    """
    Normalize the coefficients of a given quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
    return torch.div(q, norm[:, None])  # q_norm = q / ||q||


def q_conjugate(q):
    """
    Returns the complex conjugate of the input quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    conj = torch.tensor([1, -1, -1, -1], device=q.device)  # multiplication coefficients per element
    return q * conj.expand_as(q)


def transform_to_dq(T):
    q_r = matrix_to_quaternion(T[:, :3, :3])
    q_d = 0.5 * q_mul(torch.cat((torch.zeros((T.shape[0], 1), dtype=T.dtype, device=T.device), T[:, :3, 3]), dim=1),
                      q_r)
    dq = torch.cat([q_r, q_d], dim=1)
    return dq


def dq_to_screw(dq, eps=1e-6):
    """
    Return the screw parameters that describe the rigid transformation encoded in the input dual quaternion.
    Input shape: [*, 8]
    Output:
     - Plucker coordinates (l, m) for the roto-translation axis (both of shape [*, 3])
     - Amount of rotation and translation around/along the axis (both of shape [*])
    """
    assert dq.shape[-1] == 8

    dq_r, dq_d = torch.split(dq, [4, 4], dim=-1)
    theta = q_angle(dq_r)  # shape: [b, 1]
    theta_sq = theta.squeeze(dim=-1)
    no_rot = torch.logical_or(theta_sq.abs() < eps, (theta_sq - math.pi).abs() < eps)
    with_rot = ~no_rot
    dq_t = dq_translation(dq)

    l = torch.zeros(*dq.shape[:-1], 3, device=dq.device)
    d = torch.zeros(*dq.shape[:-1], device=dq.device)

    l[with_rot] = dq_r[with_rot, 1:] / torch.sin(theta[with_rot] / 2)
    d[no_rot] = torch.linalg.norm(dq_t[no_rot], dim=-1)
    l[no_rot] = dq_t[no_rot] / (d[no_rot, None] + 1e-10)

    # align axis
    up_axis = torch.tensor([1, 1, 1], dtype=l.dtype, device=l.device).unsqueeze(dim=0)
    cos = (l * up_axis).sum(dim=-1, keepdim=True)
    theta = torch.where(cos >= 0, theta, -theta)
    l = torch.where(cos >= 0, l, -l)
    d[no_rot] = torch.where(cos[no_rot, 0] >= 0, d[no_rot], -d[no_rot])
    d[with_rot] = (dq_t[with_rot] * l[with_rot]).sum(dim=-1)  # batched dot product

    no_trans = torch.isclose(d, torch.zeros_like(d, device=dq.device))
    unit_transform = torch.logical_and(no_rot, no_trans)
    l[unit_transform, 0] = 1
    if unit_transform.sum() > 0:
        unit_frames = torch.nonzero(unit_transform, as_tuple=False)
        loguru.critical(f"Warning: input contains identity matrix, "
                        f"the screw axis for frame {unit_frames} is not determined")

    theta[no_rot] = eps
    t_l_cross = torch.cross(dq_t, l, dim=-1)
    m = 0.5 * (t_l_cross + torch.cross(l, t_l_cross / torch.tan(theta / 2), dim=-1))
    return l, m, theta.squeeze(-1), d


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.
    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.
    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`
    Raises:
        ValueError if `v` is of incorrect shape.
    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def _so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
            fac1[:, None, None] * skews
            + fac2[:, None, None] * skews_square
            + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles ** 2))[:, None, None]
        + (
            log_rotation_hat_square
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles ** 3))[
                :, None, None
            ]
        )
    )

    return V


def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.
    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.
    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.
    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ hat(log_rotation) 0 ]
                         [   log_translation 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].
    Note that for any `log_transform` with `0 <= ||log_rotation|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_log_map(se3_exponential_map(log_transform)) == log_transform
    ```
    The conversion has a singularity around `||log(transform)|| = 0`
    which is handled by clamping controlled with the `eps` argument.
    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid unstable gradients in the singular case.
    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.
    Raises:
        ValueError if `log_transform` is of incorrect shape.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., :3]
    log_rotation = log_transform[..., 3:]

    # rotation is an exponential map of log_rotation
    (
        R,
        rotation_angles,
        log_rotation_hat,
        log_rotation_hat_square,
    ) = _so3_exp_map(log_rotation, eps=eps)

    # translation is V @ T
    V = _se3_V_matrix(
        log_rotation,
        log_rotation_hat,
        log_rotation_hat_square,
        rotation_angles,
        eps=eps,
    )
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    return transform.permute(0, 2, 1)


def screw_param_to_exponential_coordinates(l, m, theta, d):
    # l: (B, 3), rotation axis
    # m: (B, 3), moment
    # theta: (B, )
    # d: (B, )
    eps = 1e-6
    no_rot = torch.logical_or(theta.abs() < eps, (theta - math.pi).abs() < eps)
    with_rot = ~no_rot
    q = torch.zeros(l.shape[0], 3, device=l.device)
    q[with_rot] = torch.cross(l[with_rot], m[with_rot], dim=-1)
    w = torch.zeros(l.shape[0], 3, device=l.device)
    v = torch.zeros(l.shape[0], 3, device=l.device)
    w[with_rot] = l[with_rot]
    h = d[with_rot] / theta[with_rot]
    v[with_rot] = torch.cross(q[with_rot], l[with_rot], dim=-1) + h[:, None] * l[with_rot]
    v[no_rot] = l[no_rot]
    screw_axis = torch.cat((w, v), dim=1)
    return screw_axis * theta[:, None]


def transform_from_exponential_coordinates(log_transform):
    log_transform_input = torch.cat((log_transform[:, 3:], log_transform[:, :3]), dim=1)
    transform_matrix = se3_exp_map(log_transform_input)
    return transform_matrix.permute(0, 2, 1)
