# SuperJunction
This is the official PyTorch implementation of SuperJunction for retinal image registration. 

*For training, please refer to 'main.py' and change the corresponding configs under the folder 'configs'.

*Regarding the [pretrained models](https://drive.google.com/drive/folders/149c0oxaZ2qNV0X2aqDFcTL6x2j7cpIUN?usp=sharing), please download them here, including the weights of both Superjunction and SuperGlue.

*To evaluate SuperJunction, please use the following codes:
 ```bash
 python export_SJ.py export_descriptor configs/magicpoint_retina_repeatability_heatmap.yaml Your_export_folder(superpoint_retina) #This is for exporting the keypoints
 python evaluation_Sj.py + Your_export_path(logs/superpoint_retina/predictions) #This is for visualizing the keypoints
 python calc_precision_recall.py #This is for calculating the precision and recall
 ```
