import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from lrtransformer import LRTransformer
from loguru import logger
from torch.utils.data import DataLoader
from util.s3dis import S3DIS
from util.data_util import collate_fn
from util.focal_loss import FocalLoss
import sys, time

import torch
import numpy as np

import h5py
import torch.nn as nn

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


BATCH_SIZE = 48
NUM_INLINER_POINTS = 512
NUM_NEIGHBORS_POINTS = 512
LITE = None
TOTAL_EPOCHS = 1
#s3dis
TRAIN_AREA = ['1','2','3','4','6', 'scannet']
VAL_AREA = ['5']

# for scannet
# TRAIN_AREA = ['1']
# VAL_AREA = None

MULTISEED = 20
VAL_STEP = 5

FEATURE_SIZE = 13 # XYZ(3), normalized_xyz(3), rgb(3), normal(3), curvature(1)

initialized = False
cross_domain = False
np.random.seed(0)
np.set_printoptions(2,linewidth=100,suppress=True,sign=' ')

for i in range(len(sys.argv)):
	if sys.argv[i]=='--train-area':
		TRAIN_AREA = sys.argv[i+1].split(',')
	if sys.argv[i]=='--val-area':
		VAL_AREA = sys.argv[i+1].split(',')
	if sys.argv[i]=='--cross-domain':
		cross_domain = True
	if sys.argv[i]=='--multiseed':
		MULTISEED = int(sys.argv[i+1])
	if sys.argv[i]=='--lite':
		LITE = int(sys.argv[i+1])

if cross_domain:
	MODEL_PATH = 'models/cross_domain/lrgnet_%s.pth'%TRAIN_AREA[0]
elif FEATURE_SIZE==6:
	MODEL_PATH = 'models/lrgnet_model%s_xyz.pth'%VAL_AREA[0]
elif FEATURE_SIZE==9:
	MODEL_PATH = 'models/lrgnet_model%s_xyzrgb.pth'%VAL_AREA[0]
elif FEATURE_SIZE==12:
	MODEL_PATH = 'models/lrgnet_model%s_xyzrgbn.pth'%VAL_AREA[0]
else:
	# use full set of features
	if NUM_INLINER_POINTS!=512 or NUM_NEIGHBORS_POINTS!=512:
		MODEL_PATH = 'models/lrgnet_model%s_i_%d_j_%d.pth'%(VAL_AREA[0], NUM_INLINER_POINTS, NUM_NEIGHBORS_POINTS)
	elif LITE is not None:
		MODEL_PATH = 'models/lrgnet_model%s_lite_%d.pth'%(VAL_AREA[0], LITE)
	else:
		# MODEL_PATH = 'models/latest/lrgnet_model10%s.pth'%VAL_AREA[0]
		MODEL_PATH = 'models/largescale/lrgnet_model.pth'  # for scannet


#
def train(model, criterion, optimizer, trainloader, testloader):
	idx = np.arange(len(train_inlier_points))
	np.random.shuffle(idx)

	inlier_points = torch.from_numpy(np.zeros((BATCH_SIZE, NUM_INLINER_POINTS, FEATURE_SIZE)))
	neighbor_points = torch.from_numpy(np.zeros((BATCH_SIZE, NUM_NEIGHBORS_POINTS, FEATURE_SIZE)))
	input_add = torch.from_numpy(np.zeros((BATCH_SIZE, NUM_NEIGHBORS_POINTS), dtype=np.int32))
	input_remove = torch.from_numpy(np.zeros((BATCH_SIZE, NUM_INLINER_POINTS), dtype=np.int32))

	loss_arr, add_acc_arr, add_prc_arr, add_rcl_arr, rmv_acc_arr, rmv_prc_arr, rmv_rcl_arr = [], [], [], [], [],[], []
	start_time = time.time()
	model.train()
	for i, (inlier_pl, inlier_lab, inlier_co, inlier_off, neighbor_pl, neighbor_lab, neighbor_co, neighbor_off) in enumerate(trainloader):
		for j in range(len(inlier_co)):
				
			if inlier_co[j] >= NUM_INLINER_POINTS:
				subset = np.random.choice(inlier_co[j], NUM_INLINER_POINTS, replace=False)
			else:
				subset = list(range(inlier_co[j])) + list(np.random.choice(inlier_co[j].item(), (NUM_INLINER_POINTS-inlier_co[j]).item(), replace=True))		
			inlier_points[j,:,:] = inlier_pl[j][subset, :]
			input_remove[j,:] = inlier_lab[j][subset]			
		
			if neighbor_co[j] >= NUM_NEIGHBORS_POINTS:
				subset = np.random.choice(neighbor_co[j], NUM_NEIGHBORS_POINTS, replace=False)	
			else:
				subset = list(range(neighbor_co[j])) + list(np.random.choice(neighbor_co[j].item(), (NUM_NEIGHBORS_POINTS-neighbor_co[j]).item(), replace=True))
			neighbor_points[j,:,:] = neighbor_pl[j][subset, :]
			input_add[j,:] = neighbor_lab[j][subset]
		
		inlier_points = inlier_points.double().to(DEVICE)
		neighbor_points = neighbor_points.double().to(DEVICE)
		inlier_target = input_remove.long().to(DEVICE)
		neighbor_target =input_add.long().to(DEVICE)  #(b,n)

		outputs = model([inlier_points, neighbor_points])
		to_remove = outputs[0]   #(b,n,c)
		to_add = outputs[1]		#(b,n,c)

		add_loss = torch.mean(criterion(to_add.permute(0,2,1), neighbor_target))


		add_acc = torch.mean(torch.eq(torch.argmax(to_add,dim=-1), neighbor_target.type(torch.long)).type(torch.float))
		TP = torch.sum(torch.logical_and(torch.eq(torch.argmax(to_add, dim=-1),1), torch.eq(neighbor_target,1)).type(torch.float))
		add_prc = TP/ (torch.sum(torch.argmax(to_add, dim=-1)).type(torch.float)+1)
		add_rcl = TP/ (torch.sum(neighbor_target).type(torch.float)+1)
		

		pos_mask = torch.nonzero(inlier_target.type(torch.bool), as_tuple=True)
		neg_mask = torch.nonzero((1- inlier_target).type(torch.bool),as_tuple=True)

		pos_loss = torch.mean(criterion(to_remove[pos_mask], inlier_target[pos_mask])).double()
		neg_loss = torch.mean(criterion(to_remove[neg_mask], inlier_target[neg_mask])).double()
		pos_loss = torch.where(torch.isnan(pos_loss), 0.0, pos_loss)
		neg_loss = torch.where(torch.isnan(neg_loss), 0.0, neg_loss)

		remove_loss = pos_loss + neg_loss
		

		remove_acc = torch.mean(torch.eq(torch.argmax(to_remove,dim=-1), inlier_target.type(torch.long)).type(torch.float))
		remove_mask = nn.Softmax(dim=-1)(to_remove)[:,:,1]>0.5
		TP = torch.sum(torch.logical_and(remove_mask, torch.eq(inlier_target,1)).type(torch.float))
		remove_prc = TP/(torch.sum(remove_mask.type(torch.float))+1)
		remove_rcl = TP/(torch.sum(inlier_target).type(torch.float)+1)

		loss = add_loss + remove_loss
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		
		loss_arr.append(loss.item())
		add_acc_arr.append(add_acc.item())
		add_prc_arr.append(add_prc.item())
		add_rcl_arr.append(add_rcl.item())
		rmv_acc_arr.append(remove_acc.item())
		rmv_prc_arr.append(remove_prc.item())
		rmv_rcl_arr.append(remove_rcl.item())
	
	epoch_time.append(time.time() - start_time)
	print("Epoch %d loss %.2f add %.2f/%.2f rmv %.2f/%.2f"%(epoch+1,np.mean(loss_arr),np.mean(add_prc_arr),np.mean(add_rcl_arr),np.mean(rmv_prc_arr), np.mean(rmv_rcl_arr)))

	if VAL_AREA is not None and epoch%VAL_STEP == VAL_STEP-1:
		print("Running validation.......")
		model.eval()
		with torch.no_grad():
			inlier_points = torch.from_numpy(np.zeros((BATCH_SIZE, NUM_INLINER_POINTS, FEATURE_SIZE)))
			neighbor_points = torch.from_numpy(np.zeros((BATCH_SIZE, NUM_NEIGHBORS_POINTS, FEATURE_SIZE)))
			input_add = torch.from_numpy(np.zeros((BATCH_SIZE, NUM_NEIGHBORS_POINTS), dtype=np.int32))
			input_remove = torch.from_numpy(np.zeros((BATCH_SIZE, NUM_INLINER_POINTS), dtype=np.int32))

			loss_arr, add_prc_arr, add_rcl_arr, rmv_prc_arr, rmv_rcl_arr = [], [], [], [], []
			for i, (inlier_pl, inlier_lab, inlier_co, inlier_off, neighbor_pl, neighbor_lab, neighbor_co, neighbor_off) in enumerate(testloader):
			
				
				for j in range(len(inlier_co)):
						
					if inlier_co[j] >= NUM_INLINER_POINTS:
						subset = np.random.choice(inlier_co[j], NUM_INLINER_POINTS, replace=False)
					else:
						subset = list(range(inlier_co[j])) + list(np.random.choice(inlier_co[j].item(), (NUM_INLINER_POINTS-inlier_co[j]).item(), replace=True))		
					inlier_points[j,:,:] = inlier_pl[j][subset, :]
					input_remove[j,:] = inlier_lab[j][subset]			
					# inlier_off[j] = NUM_INLINER_POINTS if j==0 else NUM_INLINER_POINTS+inlier_off[j-1]
				
					if neighbor_co[j] >= NUM_NEIGHBORS_POINTS:
						subset = np.random.choice(neighbor_co[j], NUM_NEIGHBORS_POINTS, replace=False)	
					else:
						subset = list(range(neighbor_co[j])) + list(np.random.choice(neighbor_co[j].item(), (NUM_NEIGHBORS_POINTS-neighbor_co[j]).item(), replace=True))
					neighbor_points[j,:,:] = neighbor_pl[j][subset, :]
					input_add[j,:] = neighbor_lab[j][subset]
					# neighbor_off[j] = NUM_NEIGHBORS_POINTS if j==0 else NUM_NEIGHBORS_POINTS+neighbor_off[j-1]

				
					inlier_points = inlier_points.double().to(DEVICE)
					neighbor_points = neighbor_points.double().to(DEVICE)
					inlier_target = input_remove.long().to(DEVICE)
					neighbor_target =input_add.long().to(DEVICE)  #(b,n)
					# inlier_off = inlier_off.double().to(DEVICE)
					# neighbor_off = neighbor_off.double().to(DEVICE)

				
				outputs = model([inlier_points, neighbor_points])
				to_remove = outputs[0]   #(b,n,c)
				to_add = outputs[1]		#(b,c,n)				

				add_loss = torch.mean(criterion(to_add.permute(0,2,1), neighbor_target))

				add_acc = torch.mean(torch.eq(torch.argmax(to_add,dim=-1), neighbor_target.type(torch.long)).type(torch.float))
				TP = torch.sum(torch.logical_and(torch.eq(torch.argmax(to_add, dim=-1),1), torch.eq(neighbor_target,1)).type(torch.float))
				add_prc = TP/ (torch.sum(torch.argmax(to_add, dim=-1)).type(torch.float)+1)
				add_rcl = TP/ (torch.sum(neighbor_target).type(torch.float)+1)
				

				pos_mask = torch.nonzero(inlier_target.type(torch.bool), as_tuple=True)
				neg_mask = torch.nonzero((1- inlier_target).type(torch.bool),as_tuple=True)

				pos_loss = torch.mean(criterion(to_remove[pos_mask], inlier_target[pos_mask])).double()
				neg_loss = torch.mean(criterion(to_remove[neg_mask], inlier_target[neg_mask])).double()
				pos_loss = torch.where(torch.isnan(pos_loss), 0.0, pos_loss)
				neg_loss = torch.where(torch.isnan(neg_loss), 0.0, neg_loss)

				remove_loss = pos_loss + neg_loss
				

				remove_acc = torch.mean(torch.eq(torch.argmax(to_remove,dim=-1), inlier_target.type(torch.long)).type(torch.float))
				remove_mask = nn.Softmax(dim=-1)(to_remove)[:,:,1]>0.5
				TP = torch.sum(torch.logical_and(remove_mask, torch.eq(inlier_target,1)).type(torch.float))
				remove_prc = TP/(torch.sum(remove_mask.type(torch.float))+1)
				remove_rcl = TP/(torch.sum(inlier_target).type(torch.float)+1)

				loss = add_loss + remove_loss

				loss_arr.append(loss.item())
				add_acc_arr.append(add_acc.item())
				add_prc_arr.append(add_prc.item())
				add_rcl_arr.append(add_rcl.item())
				rmv_acc_arr.append(remove_acc.item())
				rmv_prc_arr.append(remove_prc.item())
				rmv_rcl_arr.append(remove_rcl.item())
				
	print("Validation %d loss %.2f add %.2f/%.2f rmv %.2f/%.2f"%(epoch+1,np.mean(loss_arr),np.mean(add_prc_arr),np.mean(add_rcl_arr),np.mean(rmv_prc_arr), np.mean(rmv_rcl_arr)))
	
	return model

	

if __name__ == "__main__":
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	model = LRTransformer(BATCH_SIZE, 1, NUM_INLINER_POINTS, NUM_NEIGHBORS_POINTS, FEATURE_SIZE)
	# model = nn.DataParallel(model, device_ids=[1])
	print(DEVICE)
	model = model.to(DEVICE)

	
	epoch_time = []
	criterion = nn.CrossEntropyLoss() 
	#criterion = FocalLoss()
	optimizer = torch.optim.Adam(params=list(model.parameters()), lr=0.001)
	EPOCHS = 120

	for epoch in range(EPOCHS):
		logger.info(f"Epochs: {epoch+1}")

		if not initialized or MULTISEED > 1:
			initialized = True
			val_inlier_count, val_inlier_points, val_remove, val_add, val_neighbor_count, val_neighbor_points = [], [], [], [], [], []
			train_inlier_count, train_inlier_points, train_remove, train_neighbor_count, train_add, train_neighbor_points = [], [], [], [], [], []
			if VAL_AREA is not None and (MULTISEED==0 and epoch%VAL_STEP==0 or MULTISEED>0 and epoch%VAL_STEP==VAL_STEP-1):
				AREA_LIST = TRAIN_AREA + VAL_AREA
			else:
				AREA_LIST= TRAIN_AREA

			for i, AREA in enumerate(AREA_LIST):
				if isinstance(AREA, str) and AREA.startswith('synthetic'):
					f = h5py.File('data/staged_%s.h5' % AREA, 'r')
				elif MULTISEED > 0 and AREA in TRAIN_AREA:
					SEED = epoch % MULTISEED
					try:
						f = h5py.File('data/multiseed/largescale/seed%d_area%s.h5'%(SEED,AREA),'r')    # for large scale 
						# f = h5py.File('data/multiseed/s3dis/seed%d_area%s.h5'%(SEED,AREA),'r')       # for s3dis
						# f = h5py.File('data/multiseed/scannet/seed%d_areascannet.h5'%(SEED),'r')   # for scannet
					except OSError:
						continue
				else:
					f = h5py.File('data/staged/staged_area%s.h5'%(AREA),'r')
				
				if VAL_AREA is not None and AREA in VAL_AREA:
					logger.info(f"Loading validation set....{f.filename}")
					count = f['count'][:]
					val_inlier_count.extend(count)
					points = f['points'][:]
					remove = f['remove'][:]
					idp = 0
					for i in range(len(count)):
						val_inlier_points.append(points[idp:idp+count[i],:FEATURE_SIZE])
						val_remove.append(remove[idp:idp+count[i]])
						idp+= count[i]
					neighbor_count = f['neighbor_count'][:]
					val_neighbor_count.extend(neighbor_count)
					neighbor_points = f['neighbor_points'][:]
					add = f['add'][:]
					
					idp = 0
					for i in range(len(neighbor_count)):
						val_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :FEATURE_SIZE])
						val_add.append(add[idp:idp+neighbor_count[i]])
						idp += neighbor_count[i]
				if AREA in TRAIN_AREA:
					logger.info(f"Loading training set....{f.filename}")
					count = f['count'][:]
					train_inlier_count.extend(count)
					points = f['points'][:]
					remove = f['remove'][:]

					idp = 0
					for i in range(len(count)):
						train_inlier_points.append(points[idp:idp+count[i],:FEATURE_SIZE])
						train_remove.append(remove[idp:idp+count[i]])
						idp+= count[i]
					neighbor_count = f['neighbor_count'][:]
					train_neighbor_count.extend(neighbor_count)
					neighbor_points = f['neighbor_points'][:]
					add = f['add'][:]
					idp = 0
					for i in range(len(neighbor_count)):
						train_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i],:FEATURE_SIZE])
						train_add.append(add[idp:idp+neighbor_count[i]])
						idp+= neighbor_count[i]
				if FEATURE_SIZE is None:
					FEATURE_SIZE = points.shape[1]
				f.close()

			#filter out instances where the neighbor array is empty
			train_inlier_points = [train_inlier_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
			train_inlier_count = [train_inlier_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]

			train_neighbor_points = [train_neighbor_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
			train_neighbor_count = [train_neighbor_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]

			train_add = [train_add[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
			train_remove = [train_remove[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]

			val_inlier_points = [val_inlier_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
			val_inlier_count = [val_inlier_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]

			val_neighbor_points = [val_neighbor_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
			val_neighbor_count = [val_neighbor_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]

			val_add = [val_add[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
			val_remove = [val_remove[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]


			print('train',len(train_inlier_points),train_inlier_points[0].shape, len(train_neighbor_points), len(train_inlier_count))
			print('val',len(val_inlier_points), len(val_neighbor_points))

		training_data = S3DIS(inlier_data=train_inlier_points, inlier_label=train_remove, inlier_co = train_inlier_count, neighbor_data=train_neighbor_points, neighbor_label=train_add, neighbor_co = train_neighbor_count)
		train_loader = DataLoader(training_data, BATCH_SIZE, shuffle=True, collate_fn = collate_fn)
		validation_data = S3DIS(inlier_data=val_inlier_points, inlier_label=val_remove, inlier_co = val_inlier_count, neighbor_data=val_neighbor_points, neighbor_label=val_add, neighbor_co = val_neighbor_count)
		test_loader = DataLoader(validation_data, BATCH_SIZE, shuffle=False, collate_fn = collate_fn)


		# MODEL_PATH = 'models/lrgnet_model%s.pth'%(epoch+1)
		net = train(model=model, criterion=criterion, optimizer=optimizer, trainloader=train_loader, testloader=test_loader)
		if VAL_AREA is not None and epoch%VAL_STEP == VAL_STEP-1:
			torch.save(net.state_dict(), MODEL_PATH)
			print(f"Model saved for epoch {epoch+1}")
		else:
			torch.save(net.state_dict(), MODEL_PATH)
			print(f"Model saved for epoch {epoch+1}")
	print("Avg Epoch Time: %.3f" % np.mean(epoch_time))
	   






