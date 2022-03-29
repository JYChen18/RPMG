### This code mainly comes from [poselstm](https://github.com/hazirbas/poselstm-pytorch)

<br>

## Dataset
- Download a Cambridge Landscape dataset (e.g. [KingsCollege](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset)) under dataset/ folder.
- Compute the mean image
```bash
python util/compute_image_mean.py --dataroot dataset/KingsCollege --height 256 --width 455 --save_resized_imgs
```



## PoseLSTM train/test

- If you would like to initialize the network with the pretrained weights, download the places-googlenet.pickle file under the *pretrained_models/* folder:
``` bash
mkdir pretrained_models & cd pretrained_models
wget https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/places-googlenet.pickle
cd ..
```

- Train a model:
```bash
python train.py --model poselstm --dataroot ./dataset/KingsCollege --name KingsCollege_9D_RPMG --mode 9D_RPMG --niter 1200 --beta 500 --gpu 0
```
Checkpoints are saved under `./checkpoints/KingsCollege_9D_RPMG`.
- Test the model:
```bash
python test.py --model poselstm  --dataroot ./dataset/KingsCollege --name KingsCollege_9D_RPMG --mode 9D_RPMG --gpu 0
```
The test errors will be saved to a text file under `./results/KingsCollege_9D_RPMG`.


### Optimization scheme and loss weights
* We use the training scheme defined in [PoseLSTM](https://arxiv.org/abs/1611.07890)
* Note that mean subtraction **is not used** in PoseLSTM models
* Results can be improved with a hyper-parameter search. We only care about the rotation error here.

| Dataset       | beta | iters|
| ------------- |:----:| :----:|
| King's College  | 500  | 1200 |
| Old Hospital   | 1500 | 1200 |
| Shop Façade    | 100  | 1200 |
| St Mary's Church | 250  | 1200 |

## Citation
```
@inproceedings{PoseNet15,
  title={PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization},
  author={Alex Kendall, Matthew Grimes and Roberto Cipolla },
  journal={ICCV},
  year={2015}
}
@inproceedings{PoseLSTM17,
  author = {Florian Walch and Caner Hazirbas and Laura Leal-Taixé and Torsten Sattler and Sebastian Hilsenbeck and Daniel Cremers},
  title = {Image-based localization using LSTMs for structured feature correlation},
  month = {October},
  year = {2017},
  booktitle = {ICCV},
  eprint = {1611.07890},
  url = {https://github.com/NavVisResearch/NavVis-Indoor-Dataset},
}
```
