import torch


UBODY_IDX = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LBODY_IDX = [0, 1, 2, 4, 5, 7, 8, 10, 11]

UBODY_nfeats = 1 + 2 + 1 + (len(UBODY_IDX) - 1) * 3 + (len(UBODY_IDX) - 1) * 6 + len(UBODY_IDX) * 3 # 163
LBODY_nfeats = 1 + 2 + 1 + (len(LBODY_IDX) - 1) * 3 + (len(LBODY_IDX) - 1) * 6 + len(LBODY_IDX) * 3 + 4 # 107


IPOSE_PATH = '/export/home_c/shkoh/MotionGPT/datasets/i_pose/new_joint_vecs/i_pose_new.npy'



def div_body_feats(body_feats):
    '''Divide the body features into upper and lower body.'''
    
    bs, T, _ = body_feats.shape

    ubody_idx = torch.tensor(UBODY_IDX).to(body_feats.device)
    lbody_idx = torch.tensor(LBODY_IDX).to(body_feats.device)

    root_angular_vel = body_feats[:, :, 0:1]
    root_linear_vel = body_feats[:, :, 1:3]
    root_y_pos = body_feats[:, :, 3:4]
    local_joint_pos = body_feats[:, :, 4:67].reshape(-1, T, 21, 3)
    local_joint_rot = body_feats[:, :, 67:193].reshape(-1, T, 21, 6)
    local_joint_vel = body_feats[:, :, 193:259].reshape(-1, T, 22, 3)
    foot_contact = body_feats[:, :, 259:263]

    local_ujoint_pos = local_joint_pos[:, :, (ubody_idx-1)[1:]].reshape(bs, T, -1)
    local_ljoint_pos = local_joint_pos[:, :, (lbody_idx-1)[1:]].reshape(bs, T, -1)

    local_ujoint_rot = local_joint_rot[:, :, (ubody_idx-1)[1:]].reshape(bs, T, -1)
    local_ljoint_rot = local_joint_rot[:, :, (lbody_idx-1)[1:]].reshape(bs, T, -1)

    local_ujoint_vel = local_joint_vel[:, :, ubody_idx].reshape(bs, T, -1)
    local_ljoint_vel = local_joint_vel[:, :, lbody_idx].reshape(bs, T, -1)

    ubody_x = torch.cat([root_angular_vel, root_linear_vel, root_y_pos, local_ujoint_pos, local_ujoint_rot, local_ujoint_vel], dim=-1)
    lbody_x = torch.cat([root_angular_vel, root_linear_vel, root_y_pos, local_ljoint_pos, local_ljoint_rot, local_ljoint_vel, foot_contact], dim=-1)

    return ubody_x, lbody_x


def merge_body_feats(ubody_feats, lbody_feats):

    bs, T, _ = ubody_feats.shape

    ubody_idx = torch.tensor(UBODY_IDX).to(ubody_feats.device)
    lbody_idx = torch.tensor(LBODY_IDX).to(ubody_feats.device)

    root_angular_vel = ubody_feats[:, :, 0:1]
    root_linear_vel = ubody_feats[:, :, 1:3]
    root_y_pos = ubody_feats[:, :, 3:4]
    local_ujoint_pos = ubody_feats[:, :, 4:163].reshape(-1, T, 14, 3)
    local_ujoint_rot = ubody_feats[:, :, 163:475].reshape(-1, T, 14, 6)
    local_ujoint_vel = ubody_feats[:, :, 475:539].reshape(-1, T, 15, 3)

    root_angular_vel = lbody_feats[:, :, 0:1]
    root_linear_vel = lbody_feats[:, :, 1:3]
    root_y_pos = lbody_feats[:, :, 3:4]
    local_ljoint_pos = lbody_feats[:, :, 4:111].reshape(-1, T, 9, 3)
    local_ljoint_rot = lbody_feats[:, :, 111:259].reshape(-1, T, 9, 6)
    local_ljoint_vel = lbody_feats[:, :, 259:323].reshape(-1, T, 10, 3)
    foot_contact = lbody_feats[:, :, 323:327]

    local_joint_pos = torch.zeros(bs, T, 21, 3).to(ubody_feats.device)
    local_joint_rot = torch.zeros(bs, T, 21, 6).to(ubody_feats.device)
    local_joint_vel = torch.zeros(bs, T, 22, 3).to(ubody_feats.device)

    local_joint_pos[:, :, (ubody_idx-1)[1:]] = local_ujoint_pos
    local_joint_rot[:, :, (ubody_idx-1)[1:]] = local_ujoint_rot
    local_joint_vel[:, :, ubody_idx] = local_ujoint_vel

    local_joint_pos[:, :, (lbody_idx-1)[1:]] = local_ljoint_pos
    local_joint_rot[:, :, (lbody_idx-1)[1:]] = local_ljoint_rot

    local_joint_vel[:, :, lbody_idx] = local_ljoint_vel

    body_feats = torch.cat([root_angular_vel, root_linear_vel, root_y_pos, local_joint_pos.reshape(bs, T, -1), local_joint_rot.reshape(bs, T, -1), local_joint_vel.reshape(bs, T, -1), foot_contact], dim=-1)

    return body_feats
