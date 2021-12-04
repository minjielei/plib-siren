import h5py  as h5
import numpy as np
from scipy import ndimage
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PhotonLibrary(object):
    def __init__(self, plib_file='plib.h5', pmt_file='pmt_loc.csv', lut_file=None):
        if not os.path.isfile(plib_file):
            print('Downloading photon library file... (>300MByte, may take minutes')
            os.system('curl -O https://www.nevis.columbia.edu/~kazuhiro/plib.h5 ./')
        if not os.path.isfile(plib_file):
            print('Error: failed to download the photon library file...')
            raise Exception

        with h5.File(plib_file,'r') as f:
            self._vis  = np.array(f['vis'], dtype=np.float32)
            self._min  = np.array(f['min'])
            self._max  = np.array(f['max'])
            self.shape = np.array(f['numvox'])
            
        pmt_data = np.loadtxt(pmt_file,skiprows=1,delimiter=',')
        if not (pmt_data[:,0].astype(np.int32) == np.arange(pmt_data.shape[0])).all():
            raise Exception('pmt_loc.csv contains optical channel data not in order of channel numbers')
        self._pmt_pos = pmt_data[:,1:4]
        self._pmt_dir = pmt_data[:,4:7]
        if not self._pmt_pos.shape[0] == self._vis.shape[1]:
            raise Exception('Optical detector count mismatch: photon library %d v.s. pmt_loc.csv %d' % (self._vis.shape[1],
                                                                                                        self._pmt_pos.shape[0])
                           )
        if ((self._pmt_pos < self._min) | (self._max < self._pmt_pos)).any():
            raise Exception('Some PMT positions are out of the volume bounds')
            
        # correct for errors in plib file
        self._max[0] += 10
        self._min[0] += 10

        # Load weighting look up table if provided
        if lut_file:
            with h5.File(lut_file,'r') as f:
                self.lut = np.array(f['lut'])
                self.bins = np.array(f['bins'])
                self.pmt_groups = np.concatenate((np.zeros(90), np.ones(90)))

    def LoadData(self, transform=True, eps=1e-5, extend=False):
        '''
        Load photon library visibility data. Apply scale transform if specified
        '''
        data = self._vis
        if transform:
            data = self.DataTransform(data, eps)
            
        if extend:
            data = np.expand_dims(data, -1)
            self.lut = np.expand_dims(self.lut, -1)

        return data

    def DataTransform(self, data, eps=1e-5):
        '''
        Transform vis data to log scale for training
        '''
        v0 = np.log10(eps)
        v1 = np.log10(1.+eps)
        return (np.log10(data+eps) - v0) / (v1 - v0)

    def DataTransformInv(self, data, eps=1e-5):
        '''
        Inverse log scale transform
        '''
        v0 = np.log10(eps)
        v1 = np.log10(1.+eps)
        return np.power(10, data * (v1 - v0) + v0) - eps
        
    def LoadCoord(self, normalize=True, extend=False):
        '''
        Load input coord for training/evaluation
        '''
        vox_ids = np.arange(self._vis.shape[0])
        
        return self.CoordFromVoxID(vox_ids, normalize=normalize, extend=extend)

    def CoordFromVoxID(self, idx, normalize=True, extend=False):
        '''
        Load input coord from vox id 
        '''
        if np.isscalar(idx):
            idx = np.array([idx])
        
        pos_coord = self.VoxID2Coord(idx).astype(np.float32)
        pmt_coord = ((self._pmt_pos - self._min) / (self._max - self._min)).astype(np.float32)
        
        if normalize:
            pos_coord = 2 * (pos_coord - 0.5)
            pmt_coord = 2 * (pmt_coord - 0.5)
        
        if extend:
            pos_coord = self.ExtendCoord(pos_coord, pmt_coord)
        
        return pos_coord.squeeze()
    
    def ExtendCoord(self, pos_coord, pmt_coord):
        '''
        Extend pos coord with pmt_coord for 6D input
        '''
        pos_coord = np.broadcast_to(np.expand_dims(pos_coord, 1), (pos_coord.shape[0], pmt_coord.shape[0], pos_coord.shape[1]))
        pmt_coord = np.broadcast_to(pmt_coord, (pos_coord.shape[0], pmt_coord.shape[0], pmt_coord.shape[1]))
        return np.concatenate((pos_coord, pmt_coord), -1)

    def UniformSample(self,num_points=32,use_numpy=True,use_world_coordinate=False):
        '''
        Samples visibility for a specified number of points uniformly sampled within the voxelized volume
        INPUT
          num_points - number of points to be sampled
          use_numpy - if True, the return is in numpy array. If False, the return is in torch Tensor
          use_world_coordinate - if True, returns absolute (x,y,z) position. Else fractional position is returned.
        RETURN
          An array of position, shape (num_points,3)
          An array of visibility, shape (num_points,180)
        '''
        
        array_ctor = np.array if use_numpy else torch.Tensor
        
        pos = np.random.uniform(size=num_points*3).reshape(num_points,3)
        axis_id = (pos[:] * self.shape).astype(np.int32)
        
        if use_world_coordinate:
            pos = array_ctor(self.AxisID2Position(axis_id))
        else:
            pos = array_ctor(pos)
            
        vis = array_ctor(self.VisibilityFromAxisID(axis_id))

        return pos,vis

    def VisibilityFromAxisID(self, axis_id, ch=None):
        return self.Visibility(self.AxisID2VoxID(axis_id),ch)

    def VisibilityFromXYZ(self, pos, ch=None):
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, device=device)
        return self.Visibility(self.Position2VoxID(pos), ch)

    def Visibility(self, vids, ch=None):
        '''
        Returns a probability for a detector to observe a photon.
        If ch (=detector ID) is unspecified, returns an array of probability for all detectors
        INPUT
          vids - Tensor of integer voxel IDs
          ch  - Integer (valid range 0 to N-1) to specify an optical detector (optional)
        RETURN
          Probability(ies) in FP32 to observe a photon at a specified location for each vid
        '''
        if ch is None:
            return self._vis[vids]
        return self._vis[vids][ch]

    def AxisID2VoxID(self, axis_id):
        '''
        Takes an integer ID for voxels along xyz axis (ix, iy, iz) and converts to a voxel ID
        INPUT
          axis_id - Length 3 integer array noting the position in discretized index along xyz axis
        RETURN
          The voxel ID (single integer)          
        '''
        return axis_id[:, 0] + axis_id[:, 1]*self.shape[0] + axis_id[:, 2]*(self.shape[0] * self.shape[1])

    def AxisID2Position(self, axis_id):
        '''
        Takes a axis ID (discretized location along xyz axis) and converts to a xyz position (x,y,z)
        INPUT
          axis_id - The axis ID in an integer array (ix,iy,iz)
        RETURN
          Length 3 floating point array noting the position along xyz axis
        '''    
        return self._min + (self._max - self._min) / self.shape * (axis_id + 0.5)

    def Position2VoxID(self, pos):
        '''
        Takes a tensor of xyz position (x,y,z) and converts to a tensor of voxel IDs
        INPUT
          pos - Tensor of length 3 floating point array noting the position along xyz axis
        RETURN
          Tensor of sigle integer voxel IDs       
        '''
        axis_ids = ((pos - self._min) / (self._max - self._min) * self.shape).int()

        return (axis_ids[:, 0] + axis_ids[:, 1] * self.shape[0] +  axis_ids[:, 2]*(self.shape[0] * self.shape[1])).long()
    
    def VoxID2AxisID(self, vid):
        '''
        Takes a voxel ID and converts to discretized index along xyz axis
        INPUT
          vid - The voxel ID (single integer)          
        RETURN
          Length 3 integer array noting the position in discretized index along xyz axis
        '''
        xid = vid.astype(int) % self.shape[0]
        yid = ((vid - xid) / self.shape[0]).astype(int) % self.shape[1]
        zid = ((vid - xid - (yid * self.shape[0])) / (self.shape[0] * self.shape[1])).astype(int) % self.shape[2]
        
        return np.reshape(np.stack([xid,yid,zid], -1), (-1, 3)).astype(np.float32) 

    def VoxID2Coord(self, vid):
        '''
        Takes a voxel ID and converts to normalized coordniate
        INPUT
          vid - The voxel ID (single integer)          
        RETURN
          Length 3 normalized coordinate array
        '''
        axis_id = self.VoxID2AxisID(vid)
        
        return (axis_id + 0.5) / self.shape

    def SmearData(self, bias):
        '''
        Apply Gaussian smearing to visibility values for inverse solving studies
        '''
        var = abs(np.random.normal(1.0, bias, len(self._vis)))
        self._vis = self._vis * np.expand_dims(var, -1)
        
    def SmearDataByDistance(self, data):
        indeces = np.arange(len(data))
        dist = self.DistanceToPMT(indeces)
        
        return data * np.exp(-dist / 300.)
        
    def DistanceToPMT(self, idx):
        '''
        Compute distance to nearest PMT plane along drift axis given idx
        '''
        pos_x = self._min[0] + (self._max[0] - self._min[0]) / self.shape[0] * (idx.astype(int) % self.shape[0] + 0.5)
        pmt_x1, pmt_x2 = -362.5, -57.5
        dist1, dist2 = np.abs(pos_x - pmt_x1), np.abs(pos_x - pmt_x2)
        
        return np.concatenate((np.transpose([dist1]*90), np.transpose([dist2]*90)), -1)

    def LocalVariation(self, xslice, pmt, window = 5, eps=1e-9, full=False):
        '''
        Compute local visibility value fluctuation at vid
        '''
        img_slice = np.reshape(self._vis, (394, 77, 74, -1))[:, :, xslice, pmt].swapaxes(0, 1)
        local_std = ndimage.generic_filter(img_slice, np.std, size=window)
        local_mean = ndimage.generic_filter(img_slice, np.mean, size=window)

        if not full:
            pos = ((self._pmt_pos[pmt] - self._min) / (self._max - self._min) * self.shape).astype(int)
            return (local_std / (local_mean + eps) * 100)[pos[1], pos[2]]

        return local_std / (local_mean + eps) * 100

    def WeightFromIdx(self, vid):
        '''
        Weighting factor for data at vid based on the provided weight lut file
        '''
        vis = self.DataTransform(self.Visibility(vid))
        xid = np.expand_dims(vid % 74, -1)
        bin_id = np.digitize(vis, self.bins) - 1
        batch_idx = (bin_id + self.pmt_groups * \
                len(self.bins) + xid * 2 * len(self.bins)).astype(int)

        return self.lut[batch_idx].astype(np.float32)

    def WeightFromIdxSingle(self, vid):
        '''
        Weighting factor for data at vid based on single version of weight lut file
        '''
        vis = self.DataTransform(self.Visibility(vid))
        bin_id = np.digitize(vis, self.bins) - 1

        return self.lut[bin_id].astype(np.float32)