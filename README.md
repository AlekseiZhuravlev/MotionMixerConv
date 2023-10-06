
# Convolutional Mixer

<div align="center">
  <b>Aleksei Zhuravlev and Valentin von Bornhaupt</b><br>
  <b>University of Bonn</b>
</div>

## Visualization

<table>
  <tr>
    <td><img src="conv_mixer/visualization/best_model_h36m/walking_1_10.gif" width="320" height="240"><br />Human3.6m, 10 frames prediction</td>
    <td><img src="conv_mixer/visualization/best_model_ais_autoregressive/2021-08-04-singlePerson_001_20_10.gif" width="320" height="240"><br />AIS dataset, 25 frames autoregressive prediction</td>
    </tr>
</table>

## Project Structure

* **Datasets**
  - `conv_mixer/dataset_ais_xyz.py` - dataset class for the AIS lab dataset
  - `h36m/datasets/dataset_h36m.py` - dataset class for the Human3.6m dataset, xyz format
  - `h36m/datasets/dataset_h36m_ang.py` - dataset class for the Human3.6m dataset, axis-angle format
* **Encoding**
  - `conv_mixer/encoding/positional_encoding.py` - encoding of the pose vector, with an option to add sinusoidal encoding
* **Model**
  - `h36m/conv_mixer_model.py` - ConvMixer model
* **Training**
  - `h36m/train_mixer_ais.py` - training script for the AIS lab dataset
  - `h36m/train_mixer_h36m.py` - training script for the Human3.6m dataset
* **Hyperparameter search**
  - `optuna_search/conv_optuna_main.py` - hyperparameter search for both datasets
* **Visualization**
  - `conv_mixer/utils/visualization_helpers_ais.py` - functions for visualization of the AIS dataset
  - `conv_mixer/utils/visualization_helpers_h3m.py` - functions for visualization of the Human3.6m dataset
  - `conv_mixer/utils/visualization` - gifs and images used in this notebook

## References

- https://github.com/MotionMLP/MotionMixer
- https://optuna.org/
- https://pytorch.org/

