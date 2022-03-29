
### This code mainly comes from [Spherical_Regression](https://github.com/leoshine/Spherical_Regression)

<br>

## Dataset

- Download datasets

```bash
mkdir dataset && cd dataset

# download dataset 
wget http://isis-data.science.uva.nl/shuai/datasets/ModelNet10-SO3.tar.gz

# unzip and overwrite ModelNet10-SO3 folder
tar xzvf ModelNet10-SO3.tar.gz

# You should find following 3 lmdb database folders extracted:
#  (1) train_100V.Rawjpg.lmdb : 
#        training set with 100 random sampled views per CAD model. 
#  (2) train_20V.Rawjpg.lmdb
#        training set with  20 random sampled views per CAD model. 
#  (3) test_20V.Rawjpg.lmdb  
#        test set with 20 random sampled views per CAD model. 

cd ..
```
Alternatively, you can download from [this googe drive](https://drive.google.com/file/d/17GLZbNTDq8B_MOgrV1TiJPoqcm_oQ_mK/view?usp=sharing)

<br>

## Train and Test
- train/test
```bash
cd S3.3D_Rotation

#For more parsers and options, please see config.py
python train.py --exp_name 9D_RPMG_sofa --category sofa --mode 9D_RPMG 
python test.py --exp_name 9D_RPMG_sofa --category sofa --mode 9D_RPMG 
cd ..
```

- Use tensorboard to view the logs of training and testing.
```bash
cd exps
tensorboard --logdir .
```


## Citation

```
@inproceedings{liao2019spherical,
  title={Spherical regression: Learning viewpoints, surface normals and 3d rotations on n-spheres},
  author={Liao, Shuai and Gavves, Efstratios and Snoek, Cees GM},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9759--9767},
  year={2019}
}
```
