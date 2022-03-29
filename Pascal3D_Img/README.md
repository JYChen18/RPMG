
### This code mainly comes from [Spherical_Regression](https://github.com/leoshine/Spherical_Regression)

<br>

## Dataset
- Download dataset
```bash
mkdir dataset && cd dataset

# download and unzip Pascal3D+ dataset
wget ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip
unzip PASCAL3D+_release1.1.zip

# download and untar the synthetic data from "Render for CNN".
wget https://shapenet.cs.stanford.edu/media/syn_images_cropped_bkg_overlaid.tar
tar xvf syn_images_cropped_bkg_overlaid.tar
cd ..
```

- Preprocess data
```bash
cd S3.3D_Rotation
python dataset.py
cd ..
```
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

@inproceedings{su2015render,
  title={Render for cnn: Viewpoint estimation in images using cnns trained with rendered 3d model views},
  author={Su, Hao and Qi, Charles R and Li, Yangyan and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2686--2694},
  year={2015}
}
```
