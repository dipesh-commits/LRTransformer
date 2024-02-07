# Region-Transformer: Self-Attention Region Based Class-Agnostic Point Cloud Segmentation

![architecture](figure/networkarchitecture.png?raw=true)

## Environment Setup and Dependencies

This environment setup is designed for Pytorch version >=1.11.0, Cuda version > 11.0. 
I used Linux operating system along with the following major requirements.
- cuda 11.7.0 with Pytorch 1.13.1 with NVIDIA A6000 series GPU
- cuda 11.3.0 with Pytorch 1.11.0 with NVIDIA A5500 series GPU

Make sure you have properly installed NVIDIA GPU drivers for above cuda versions

Run the following commands
```
conda create -n myenv python=3.8
conda activate myenv

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch (on A5500)
 OR 
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia (on A6000)
```
	
Clone the repository and run the below commands
```
cd lib/pointops
python setup.py install
cd ../..
pip install -r requirements.txt
```


## Data Staging

Run the following script to download the necessary point cloud files in H5 format to the *data* folder.

```
bash download_data.sh
```

## Data Visualization

To check the data shape/size contained in each H5 file:

```
python examine_h5.py data/scannet.h5
```

```
<HDF5 dataset "count_room": shape (312,), type "<i4">
min 5451.00 mean 25397.58 max 68432.00
<HDF5 dataset "points": shape (7924044, 8), type "<f4">
min -6.17 mean 1.80 max 107.00

```

```
Total training points for S3DIS: 844345552
Total training instances for S3DIS: 3519615

Total training points for Scannet: 741306913
Total training instances for Scannet: 5026079
```


To convert the H5 data file into individual point cloud files (PLY) in format, run the script as follows.
PLY files can be opened using the [CloudCompare](https://www.danielgm.net/cc/) program

```bash
#Render the point clouds in original RGB color
python h5_to_ply.py data/s3dis_area3.h5 --rgb
#Render the point clouds colored according to segmentation ID
python h5_to_ply.py data/s3dis_area3.h5 --seg
```

```
...
Saved to data/viz/22.ply: (18464 points)
Saved to data/viz/23.ply: (20749 points)
```

To plot the instance color legend for a target room:

```bash
#Plot color legend for room #100 in ScanNet
python h5_to_ply.py data/scannet.h5 --target 100
```


## Learn-Region Transformer (LRTransformer)

Run simulations to stage ground truth data for LRTransformer.

```bash
python stage_data.py
#To apply data augmentation, run stage_data with different random seeds
for i in 0 1 2 3 4 5 6 7
do
	for j in s3dis scannet
	do
		python stage_data.py --seed $i --area $j
	done
done
```

Train Region-Transformer for each area of the S3DIS dataset.

```bash
python train_lr_transformer.py --train-area 1,2,3,4,6 --val-area 5
```

Test Region-Transformer and measure the accuracy metrics.

```bash
python test_lr_transformer.py --area 5 --save
python test_lr_transformer.py --area scannet --save
```

Credits to 
--------

	@ARTICLE{chen2021ral,
		author={J. {Chen} and Z. {Kira} and Y. K. {Cho}},
		journal={IEEE Robotics and Automation Letters}, 
		title={LRGNet: Learnable Region Growing for Class-Agnostic Point Cloud Segmentation}, 
		year={2021},
		volume={6},
		number={2},
		pages={2799-2806},
		doi={10.1109/LRA.2021.3062607},
	}


	@misc{zhao2021point,
      title={Point Transformer}, 
      author={Hengshuang Zhao and Li Jiang and Jiaya Jia and Philip Torr and Vladlen Koltun},
      year={2021},
      eprint={2012.09164},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
