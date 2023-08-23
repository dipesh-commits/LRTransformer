
import os
import h5py
import torch
import torch.nn as nn
from loguru import logger
from lib.pointops.functions import pointops

def loadFromH5(filename, load_labels=True):
	f = h5py.File(filename,'r')
	all_points = f['points'][:]
	count_room = f['count_room'][:]
	tmp_points = []
	idp = 0
	for i in range(len(count_room)):
		tmp_points.append(all_points[idp:idp+count_room[i], :])
		idp += count_room[i]
	f.close()
	room = []
	labels = []
	class_labels = []
	if load_labels:
		for i in range(len(tmp_points)):
			room.append(tmp_points[i][:,:-2])
			labels.append(tmp_points[i][:,-2].astype(int))
			class_labels.append(tmp_points[i][:,-1].astype(int))
		return room, labels, class_labels
	else:
		return tmp_points

def savePCD(filename,points):
	if len(points)==0:
		return
	f = open(filename,"w")
	l = len(points)
	header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH %d
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d
DATA ascii
""" % (l,l)
	f.write(header)
	for p in points:
		rgb = (int(p[3]) << 16) | (int(p[4]) << 8) | int(p[5])
		f.write("%f %f %f %d\n"%(p[0],p[1],p[2],rgb))
	f.close()
	print('Saved %d points to %s' % (l,filename))


def savePLY(filename, points):
	f = open(filename,'w')
	f.write("""ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""" % len(points))
	for p in points:
		f.write("%f %f %f %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
former_inlier_plane = None
former_neighbor_plane = None
       
class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super(PointTransformerLayer, self).__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes, device=DEVICE)
        self.linear_k = nn.Linear(in_planes, mid_planes, device=DEVICE)
        self.linear_v = nn.Linear(in_planes, out_planes,device=DEVICE)
        self.linear_p = nn.Sequential(nn.Linear(3, 3,device=DEVICE), nn.BatchNorm1d(3,device=DEVICE), nn.ReLU(inplace=True), nn.Linear(3, out_planes,device=DEVICE))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes, device=DEVICE), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes, device=DEVICE),
                                    nn.BatchNorm1d(mid_planes // share_planes, device=DEVICE), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes, device=DEVICE))
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x = pxo  # (n, 3), (n,c)
        x = x.permute(0,2,1).contiguous()
        p = p.contiguous()
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        

        #mycode

        # used pointops because it also uses batch size to query and group the points. It returens 4D tensor (b,n,nsample,c)
        x_k_obj = pointops.QueryAndGroup(nsample=16,use_xyz=True)
        x_v_obj = pointops.QueryAndGroup(nsample=16, use_xyz=False)
        x_k = x_k_obj(p,p,x_k.permute(0,2,1))
        x_v = x_v_obj(p,p,x_v.permute(0,2,1))
        
        
        # x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        # x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        x_k = x_k.permute(0,2,3,1)
        x_v = x_v.permute(0,2,3,1)
        p_r, x_k = x_k[:, :, :, 0:3], x_k[:, :, :, 3:]
        

        for i, layer in enumerate(self.linear_p):
            if i==1:
                p_r = p_r.permute(0,3,1,2)
                p_r = p_r.reshape(p_r.shape[0], p_r.shape[1], p_r.shape[2]*p_r.shape[3]).contiguous()
                
                # p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                p_r = layer(p_r).reshape(p_r.shape[0], p_r.shape[1], 512, 16).permute(0,2,3,1).contiguous()   # changed here
                
            else:
                p_r = layer(p_r)    # (n, nsample, c)    # for mine (b,n,nsample,c)


        # w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        w = x_k - x_q.unsqueeze(2) + p_r.view(p_r.shape[0], p_r.shape[1], p_r.shape[2], self.out_planes//self.mid_planes, self.mid_planes).sum(3)  # (b, n, nsample, c)
        
        for i, layer in enumerate(self.linear_w):
            if i%3 == 0:
                 w = w.permute(0,3,1,2)
                 w = w.reshape(w.shape[0], w.shape[1], w.shape[2]*w.shape[3]).contiguous()
                 w = layer(w).reshape(w.shape[0], w.shape[1], 512 ,16).permute(0,2,3,1).contiguous()   # changed here
                # w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            else: 
                w = layer(w)
        
        w = self.softmax(w)  # (n, nsample, c)
        
        b, n, nsample, c = x_v.shape; s = self.share_planes
        
        x = ((x_v + p_r).view(b, n, nsample, s, c // s) * w.unsqueeze(3)).sum(2).view(b, n, c)
        
        return x
        
class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):   # planes = features, 
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False, device=DEVICE)
        self.bn1 = nn.BatchNorm1d(planes, device=DEVICE)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes, device=DEVICE)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False, device=DEVICE)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion, device=DEVICE)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x = pxo  # (n, 3), (n, c), (b)
               
        x = self.relu(self.bn1(self.linear1(x.float()).permute(0,2,1).float()))
        
        identity = x
        x = self.relu(self.bn2(self.transformer2([p, x]).permute(0,2,1).float()))
        x = x.permute(0,2,1)
        x = self.bn3(self.linear3(x).permute(0,2,1))
        x = x.permute(0,2,1)        
        
        x += identity.permute(0,2,1)
        x = self.relu(x)
        return [p, x]
    
class PointTransformerSeg(nn.Module):
    
    def __init__(self, block, blocks, planes, c=13, k=0, inlier_first=False, neighbor_first=False, inlier=False, neighbor=False, last_block=False):
        super(PointTransformerSeg, self).__init__()
        self.c = c
        self.k = k   # k is the number of class labels
        self.inlier_first = inlier_first
        self.neighbor_first = neighbor_first
        self.inlier, self.neighbor = inlier, neighbor
        self.last_block= last_block
        self.in_planes, planes = c, planes   # planes mean total number of features in each block [32,64,64,128,512], c means the total number of input features
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 32
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]   # stride means the number of times to reduce the dimensions. Currently, i don't need to reduce the dimension. so taking 1 for each blocks
        
        self.encoders = nn.ModuleList()
        for i in range(len(blocks)):
            self.encoders.append(self._make_enc(block, planes[i], blocks[i], share_planes, stride=stride[0], nsample=nsample[1]))  # N/1
        
        if k == 2:
            # self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
            
            # self.cls = nn.Sequential(nn.Linear(planes[-1], planes[0]//2, device=DEVICE), nn.BatchNorm1d(planes[0]//2, device=DEVICE), nn.ReLU(inplace=True), nn.Linear(planes[0]//2, k, device=DEVICE))
            self.cls_lin1 = nn.Linear(planes[-1], planes[0]//2, device=DEVICE)
            self.cls_batch1 = nn.BatchNorm1d(planes[0]//2, device=DEVICE)
            self.cls_relu1 = nn.ReLU(inplace=True)
            self.cls = nn.Linear(planes[0]//2, k, device=DEVICE)
            # self.cls = nn.Sequential(nn.Linear(planes[-1], planes[0]//2, device=DEVICE), nn.BatchNorm1d(planes[0]//2, device=DEVICE), nn.ReLU(inplace=True), nn.Linear(planes[0]//2, k, device=DEVICE))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        global former_inlier_plane
        global former_neighbor_plane
        layers = nn.ModuleList()                
        for _ in range(blocks):
            # self.in_planes = planes * block.expansion
            if self.inlier_first or self.neighbor_first:
                 self.in_planes = self.c
                 self.inlier_first = False
                 self.neighbor_first = False
            else:
                if self.inlier:
                    self.in_planes = former_inlier_plane
                
                if self.neighbor:
                    self.in_planes = former_neighbor_plane

                if self.last_block:
                    # it is hard coded. need to make it dynamic
                    self.in_planes = 2176    # for last block we have a total of features from (inlier branch from second block+neighbor branch from second block+output from first block) as input after pooling and concat
                    self.last_block = False                
            
            if self.inlier:
                former_inlier_plane = planes
            if self.neighbor:
                former_neighbor_plane = planes
            layers.append(block(self.in_planes, planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)


    def forward(self, pxo):
        
        x = pxo  # (n, 3), (n, c), (b)
        p = x[:,:,:3]

        # x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)

        for _, enc in enumerate(self.encoders):
            p,x = enc([p,x])      # p means xyz, x means features, o means offset   (b,n,c)

        if self.k == 2:
            x = self.cls_relu1(self.cls_batch1(self.cls_lin1(x).permute(0,2,1)))
            x = x.permute(0,2,1)
            x = self.cls(x)
            # x = self.cls(x)
            return x
        else:
            return [p,x]
        

class LRTransformer(nn.Module):
    def __init__(self, batch, seq_len, num_inlier, num_neighbor, feature_size):
        super(LRTransformer, self).__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.num_inlier = num_inlier
        self.num_neighbor = num_neighbor
        self.feature_size = feature_size

        self.block1_inliner = PointTransformerSeg(PointTransformerBlock, [1,1], [128, 128], inlier_first=True, inlier=True)
        self.block1_neighbor = PointTransformerSeg(PointTransformerBlock, [1,1], [128, 128], neighbor_first=True, neighbor=True)

        self.block2_inliner = PointTransformerSeg(PointTransformerBlock, [1,1,1,1], [128,256,512,1024], inlier=True)
        self.block2_neighbor = PointTransformerSeg(PointTransformerBlock, [1,1,1,1], [128,256,512,1024], neighbor=True)

        self.block3_inliner = PointTransformerSeg(PointTransformerBlock, [1,1,1], [512,256,128],  k=2, inlier=True, last_block=True)
        self.block3_neighbor = PointTransformerSeg(PointTransformerBlock, [1,1,1], [512, 256,128], k=2, neighbor=True, last_block=True)

    def forward(self, x):
        inlier_feat, neighbor_feat = x

        _, to_remove_feat_b1 = self.block1_inliner(inlier_feat)
        _, to_add_feat_b1 = self.block1_neighbor(neighbor_feat)
        

        _, to_remove_feat_b2 = self.block2_inliner(to_remove_feat_b1)
        _, to_add_feat_b2 = self.block2_neighbor(to_add_feat_b1)


        # add linear layer instead of max pooling
        # to_remove_feat_b2 = nn.MaxPool1d(2)(to_remove_feat_b2.permute(0,2,1))
        # to_remove_feat_b2 = to_remove_feat_b2.permute(0,2,1)
        # to_remove_feat_b2 = nn.Linear(to_remove_feat_b2.shape[2], to_remove_feat_b2.shape[2], bias=False, device= DEVICE)(to_remove_feat_b2)
        # to_remove_feat_b2 = nn.ReLU(inplace=True)(to_remove_feat_b2)
        # to_remove_feat_b2 = nn.MaxPool1d(to_remove_feat_b2.shape[1])(to_remove_feat_b2.permute(0,2,1)).permute(0,2,1)

        # to_add_feat_b2 = nn.MaxPool1d(2)(to_add_feat_b2.permute(0,2,1))
        # to_add_feat_b2 = to_add_feat_b2.permute(0,2,1)
        # to_add_feat_b2 = nn.Linear(to_add_feat_b2.shape[2], to_add_feat_b2.shape[2], bias=False, device= DEVICE)(to_add_feat_b2)
        # to_add_feat_b2 = nn.ReLU(inplace=True)(to_add_feat_b2)
        # to_add_feat_b2 = nn.MaxPool1d(to_add_feat_b2.shape[1])(to_add_feat_b2.permute(0,2,1)).permute(0,2,1)

        # # exit()

        # feature_pool = torch.cat((to_remove_feat_b2, to_add_feat_b2), dim=-1)

        # logger.warning(torch.amax(to_remove_feat_b2,dim=1,keepdim=True))
        # logger.error(torch.mean(to_remove_feat_b2, dim=1, keepdim=True))
        

        feature_pool = torch.cat((
            torch.mean(to_remove_feat_b2, dim=1, keepdim=True),
            torch.mean(to_add_feat_b2, dim=1, keepdim=True)
        ),dim=-1)      # max pooling to one point

        # logger.error(feature_pool.shape)
        # exit()

        feature_pool = torch.broadcast_to(feature_pool, (feature_pool.shape[0], self.num_inlier, feature_pool.shape[2]))   # broadcast to N number of points

        to_remove_feat_b1 = torch.cat((feature_pool, to_remove_feat_b1),dim=-1)     # concatenate position encoding features
        to_add_feat_b1 = torch.cat((feature_pool, to_add_feat_b1), axis=-1)

        
        
        to_remove = self.block3_inliner(to_remove_feat_b1)
        to_add = self.block3_neighbor(to_add_feat_b1)       
        return (to_remove, to_add)

if __name__ == '__main__':
     pass