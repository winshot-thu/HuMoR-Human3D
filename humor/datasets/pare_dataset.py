import sys, os
sys.path.append('.')
from torch.utils.data import Dataset
import joblib
import math
import torch
import numpy as np
from humor.body_model.utils import KEYPT_VERTS
from torch.utils.data import Dataset, DataLoader
import json
import os.path as osp
DEFAULT_GROUND = [0.0, -1.0, 0.0, -0.5]

def batch_rot2aa(Rs):
    cos = 0.5 * (torch.stack([torch.trace(x) for x in Rs]) - 1)
    cos = torch.clamp(cos, -1, 1)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)

def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def read_keypoints(keypoint_fn):
    OP_NUM_JOINTS = 25
    '''
    Only reads body keypoint data of first person.
    '''
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data['people']) == 0:
        print('WARNING: Found no keypoints in %s! Returning zeros!' % (keypoint_fn))
        return np.zeros((OP_NUM_JOINTS, 3), dtype=np.float)

    person_data = data['people'][0]
    body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                dtype=np.float)
    body_keypoints = body_keypoints.reshape([-1, 3])

    return body_keypoints

class PAREDataset(Dataset):
    def __init__(self, pare_results_path, img_folder, cat_mat=None, seq_len=None, overlap_len=None, vid_name=None):
        self.data = None
        self.seq_len = seq_len
        self.overlap_len = overlap_len
        self.cat_mat = cat_mat 
        self.vid_name = vid_name
        self.use_3d = True
        self.img_path = img_folder

        self.data_dict, self.seq_intervals = self.load_data(pare_results_path)
        self.data_len = len(self.data_dict['joints2d']) 
        print('RGB dataset contains %d sub-sequences...' % (self.data_len))
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        obs_data = dict()
        gt_data = dict()

        if self.use_3d:
            # obs_data['joints3d'] = torch.Tensor(self.data_dict['joints3d'][idx])
            obs_data['verts3d'] = torch.Tensor(self.data_dict['verts3d'][idx])
            obs_data['verts3d_orig'] = torch.Tensor(self.data_dict['verts3d_orig'][idx])
        
        obs_data['joints2d'] = torch.Tensor(self.data_dict['joints2d'][idx])
        obs_data['seq_interval'] = torch.Tensor(list(self.seq_intervals[idx])).to(torch.int)
        obs_data['floor_plane'] = self.data_dict['floor_plane'][idx]  
        obs_data['img_paths'] = self.data_dict['img_paths'][idx]
        obs_data['poses'] = self.data_dict['poses'][idx]

        gt_data['name'] = self.data_dict['names'][idx]
        gt_data['cam_matx'] = torch.Tensor(self.data_dict['cam_matx'][idx])


        return obs_data, gt_data
        
    def load_data(self, pare_results_path):
        data = joblib.load(pare_results_path)
        data = joblib.load("../humor-revised/out/xiangyang1/rgb_preprocess/data.pt")
        for k,v in data.items():
            data[k] = v[1:]
        # print("Loading results from out/xiangyang1/rgb_preprocess/data.pt")
        # with open("/apdcephfs/share_1290939/shaolihuang/wenshuochen/humor-revised/pare_output.pkl", "rb") as f:
        #     data = joblib.load(f)[1]

       #  data = joblib.load("/apdcephfs/share_1290939/shaolihuang/wenshuochen/humor-revised/yongkang_res_719.pt")
        data = joblib.load("/apdcephfs/share_1290939/shaolihuang/wenshuochen/humor-revised/out/xiangyang1/rgb_preprocess/data.pt")
        num_frames = data['pose'].shape[0] # pose 93, 24, 3,3 
        print('Found video with %d frames...' % (num_frames))

        # preprocess pose matrix2vec

        pose = data['pose'] # 93 24 3 3 
        pose = torch.from_numpy(pose)
        pose = batch_rot2aa(pose.reshape(pose.shape[0]*pose.shape[1], 3, 3)).reshape(pose.shape[0], pose.shape[1], 3).numpy() # 93, 24, 3

        joints2d = data['smpl_joints2d']
    
        verts = data['verts']
        joints3d = data['joints3d']
        joints3d = joints3d[:,:22,]

        seq_intervals = []

        if self.seq_len is not None and self.overlap_len is not None: # False
            num_seqs = math.ceil((num_frames - self.overlap_len) / (self.seq_len - self.overlap_len))
            r = self.seq_len*num_seqs - self.overlap_len*(num_seqs-1) - num_frames # number of extra frames we cover
            extra_o = r // (num_seqs - 1) # we increase the overlap to avoid these as much as possible
            self.overlap_len = self.overlap_len + extra_o

            new_cov = self.seq_len*num_seqs - self.overlap_len*(num_seqs-1) # now compute how many frames are still left to account for
            r = new_cov - num_frames

            # create intervals
            cur_s = 0
            cur_e = cur_s + self.seq_len
            for int_idx in range(num_seqs):
                seq_intervals.append((cur_s, cur_e))
                cur_overlap = self.overlap_len
                if int_idx < r:
                    cur_overlap += 1 # update to account for final remainder
                cur_s += (self.seq_len - cur_overlap)
                cur_e = cur_s + self.seq_len

            print('Splitting into subsequences of length %d frames overlapping by %d...' % (self.seq_len, self.overlap_len))
        else:
            print('Not splitting the video...')
            num_seqs = 1
            self.seq_len = num_frames
            seq_intervals = [(0, self.seq_len)] # 如果不切割，seq_intervals就是0-视频长度
        
        # intrinsics
        cam_mat = self.cat_mat

        img_paths = None
        if self.img_path is not None:
            img_paths = [osp.join(self.img_path, img_fn)
                            for img_fn in os.listdir(self.img_path)
                            if img_fn.endswith('.png') or
                            img_fn.endswith('.jpg') and
                            not img_fn.startswith('.')]
            img_paths = sorted(img_paths)

        if self.use_3d:
            data_out = {
                'img_paths': [],
                'cam_matx' : [],
                'joints2d': [], # 25个2d关键点
                'names': [],
                'floor_plane': [],
                'joints3d': [], # 22个关节点
                'verts3d': [], # 43个顶点
                'verts3d_orig':[],
                'poses':[]
            }
  
        else:
            data_out = {
                'img_paths': [],
                'cam_matx' : [],
                'joints2d': [], # 25个2d关键点
                'names': [],
                'floor_plane': [],
                'poses':[]
            }

        floor_plane = np.array(DEFAULT_GROUND)
        for seq_idx in range(num_seqs):
            sidx, eidx = seq_intervals[seq_idx]
            data_out['cam_matx'].append(cam_mat)
            data_out['joints2d'].append(joints2d[sidx:eidx, :25, :])
            data_out['names'].append(self.vid_name + '_' + '%04d' % (seq_idx))
            data_out['floor_plane'].append(floor_plane)
            data_out['poses'].append(pose[sidx:eidx,:22])

            if img_paths is not None:
                data_out['img_paths'].append(img_paths[sidx:eidx])

            if self.use_3d:
                data_out['joints3d'].append(joints3d[sidx:eidx])
                data_out['verts3d'].append(verts[sidx:eidx, KEYPT_VERTS])
                data_out['verts3d_orig'].append(verts[sidx:eidx])

        return data_out, seq_intervals # data_out ('img_paths', (1, 93)) ('mask_paths', (1, 93)) ('cam_matx', (1, 3, 3)) ('joints2d', (1, 93, 25, 3)) ('floor_plane', (1, 4)) ('names', (1,))

