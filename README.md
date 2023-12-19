# DiaPer 🩲

PyTorch implementation for [DiaPer: End-to-End Neural Diarization with Perceiver-Based Attractors](https://arxiv.org/pdf/2312.04324.pdf).


## Usage

### Getting started

We recommend to create an [anaconda](https://www.anaconda.com/) environment
```bash
conda create -n DiaPer python=3.7
conda activate DiaPer
```
Clone the repository
```bash
git clone https://github.com/BUTSpeechFIT/DiaPer.git
```
Install the packages
```bash
conda install pip
pip install git+https://github.com/fnlandini/transformers
conda install numpy
conda install -c conda-forge tensorboard
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install safe_gpu
pip install yamlargparse
pip install scikit-learn
pip install decorator
pip install librosa==0.9.1
pip install setuptools==59.5.0
pip install h5py
pip install matplotlib
```
Other versions might work but these were the settings used for this work.

Run the example
```bash
./examples/run_example.sh
```
If it works, you should be set.


### Train
To run the training you can call:
```bash
    python diaper/train.py -c examples/train.yaml
```
Note that in the example you need to define the train and validation data directories as well as the output directory. The rest of the parameters are standard ones, as used in our publication.
For adaptation or fine-tuning, the process is similar:
```bash
    python diaper/train.py -c examples/finetune_adaptedmorespeakers.yaml
```
In that case, you will need to provide the path where to find the trained model that you want to adapt/fine-tune.


### Inference
To run the inference, you can call:
```bash
    python diaper/infer.py -c examples/infer.yaml
```
Note that in the example you need to define the data, model and output directories.

Or, if you want to only evaluate one file:
```
    python diaper/infer_single_file.py -c examples/infer.yaml --wav-dir <directory with wav file> --wav-name <filename without extension>
```
Note that in the example you need to define the model and output directories.

### Inference with pre-trained models
You can also run inference using the models we share. Either with the usual approach or a single file like:
```bash
python diaper/infer_single_file.py -c examples/infer_16k_10attractors.yaml --wav-dir examples --wav-name IS1009a
```
for the model trained on simulated conversations (no fine-tuning) or with fine-tuning as:
```bash
python diaper/infer_single_file.py -c examples/infer_16k_10attractors_AMIheadsetFT.yaml --wav-dir examples --wav-name IS1009a
```
You should obtain results as in `examples/IS1009a_infer_16k_10attractors.rttm` and `examples/IS1009a_infer_16k_10attractors_AMIheadsetFT.rttm` respectively.

All models trained on publicly available and free data are shared inside the folder `models`. Both families of models with 10 and 20 attractors are available. If you want to use any of them, modify the infer files above to suit your needs. You will need to change `models_path` and `epochs` (and `rttms_dir`, where the output will be generated) to use the model you want.


## Results

| | 10 attractors | 10 attractors | 20 attractors | 20 attractors | VAD+VBx+OSD |
|---|---|---|---|---|---|
| **DER** and **RTTMs** | without FT | with FT | without FT | with FT | --- |
AISHELL-4 | [48.21](results/DiaPer/10attractors/AISHELL4mix/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AISHELL4mix/withoutFT/test/rttms) | [41.43](results/DiaPer/10attractors/AISHELL4mix/withFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AISHELL4mix/withFT/test/rttms) | [47.86](results/DiaPer/20attractors/AISHELL4mix/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AISHELL4mix/withoutFT/test/rttms) | [31.30](results/DiaPer/20attractors/AISHELL4mix/withFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AISHELL4mix/withFT/test/rttms) | [15.84](results/baseline_VBx/16kHz/AISHELL4mix/test/result_collar0.0)% [📁](results/baseline_VBx/16kHz/AISHELL4mix/test/rttms) |
AliMeeting (far) | [38.67](results/DiaPer/10attractors/AliMeetingFarmix/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AliMeetingFarmix/withoutFT/test/rttms) | [32.60](results/DiaPer/10attractors/AliMeetingFarmix/withFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AliMeetingFarmix/withFT/test/rttms) | [34.35](results/DiaPer/20attractors/AliMeetingFarmix/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AliMeetingFarmix/withoutFT/test/rttms) | [26.27](results/DiaPer/20attractors/AliMeetingFarmix/withFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AliMeetingFarmix/withFT/test/rttms) | [28.84](results/baseline_VBx/16kHz/AliMeetingFarmix/test/result_collar0.0)% [📁](results/baseline_VBx/16kHz/AliMeetingFarmix/test/rttms) |
AliMeeting (near) | [28.19](results/DiaPer/10attractors/AliMeetingNearmix/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AliMeetingNearmix/withoutFT/test/rttms) | [27.82](results/DiaPer/10attractors/AliMeetingNearmix/withFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AliMeetingNearmix/withFT/test/rttms) | [23.90](results/DiaPer/20attractors/AliMeetingNearmix/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AliMeetingNearmix/withoutFT/test/rttms) | [24.44](results/DiaPer/20attractors/AliMeetingNearmix/withFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AliMeetingNearmix/withFT/test/rttms) | [22.59](results/baseline_VBx/16kHz/AliMeetingNearmix/test/result_collar0.0)% [📁](results/baseline_VBx/16kHz/AliMeetingNearmix/test/rttms) |
AMI (array) | [57.07](results/DiaPer/10attractors/AMImixarray/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AMImixarray/withoutFT/test/rttms) | [49.75](results/DiaPer/10attractors/AMImixarray/withFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AMImixarray/withFT/test/rttms) | [52.29](results/DiaPer/20attractors/AMImixarray/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AMImixarray/withoutFT/test/rttms) | [50.97](results/DiaPer/20attractors/AMImixarray/withFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AMImixarray/withFT/test/rttms) | [34.61](results/baseline_VBx/16kHz/AMImixarray/test/result_collar0.0)% [📁](results/baseline_VBx/16kHz/AMImixarray/test/rttms) |
AMI (headset) | [36.36](results/DiaPer/10attractors/AMImixheadset/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AMImixheadset/withoutFT/test/rttms) | [32.94](results/DiaPer/10attractors/AMImixheadset/withFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/AMImixheadset/withFT/test/rttms) | [35.08](results/DiaPer/20attractors/AMImixheadset/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AMImixheadset/withoutFT/test/rttms) | [30.49](results/DiaPer/20attractors/AMImixheadset/withFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/AMImixheadset/withFT/test/rttms) | [22.42](results/baseline_VBx/16kHz/AMImixheadset/test/result_collar0.0)% [📁](results/baseline_VBx/16kHz/AMImixheadset/test/rttms) |
Callhome | [14.86](results/DiaPer/telephone_10attractors/Callhome/withoutFT/part2/result_collar0.25)% [📁](results/DiaPer/telephone_10attractors/Callhome/withoutFT/part2/rttms) | [13.60](results/DiaPer/telephone_10attractors/Callhome/withFT/part2/result_collar0.25)% [📁](results/DiaPer/telephone_10attractors/Callhome/withFT/part2/rttms) | -- | -- | [13.62](results/baseline_VBx/8kHz/Callhome/part2/result_collar0.25)% [📁](results/baseline_VBx/8kHz/Callhome/part2/rttms) |
CHiME6 | [78.25](results/DiaPer/10attractors/CHiME6/withoutFT/eval/result_collar0.25)% [📁](results/DiaPer/10attractors/CHiME6/withoutFT/eval/rttms) | [70.77](results/DiaPer/10attractors/CHiME6/withFT/eval/result_collar0.25)% [📁](results/DiaPer/10attractors/CHiME6/withFT/eval/rttms) | [77.51](results/DiaPer/20attractors/CHiME6/withoutFT/eval/result_collar0.25)% [📁](results/DiaPer/20attractors/CHiME6/withoutFT/eval/rttms) | [69.94](results/DiaPer/20attractors/CHiME6/withFT/eval/result_collar0.25)% [📁](results/DiaPer/20attractors/CHiME6/withFT/eval/rttms) | [70.42](results/baseline_VBx/16kHz/CHiME6/eval/result_collar0.25)% [📁](results/baseline_VBx/16kHz/CHiME6/eval/rttms) |
DIHARD 2 | [43.75](results/DiaPer/10attractors/DIHARD2/withoutFT/eval/result_collar0.0)% [📁](results/DiaPer/10attractors/DIHARD2/withoutFT/eval/rttms) | [32.97](results/DiaPer/10attractors/DIHARD2/withFT/eval/result_collar0.0)% [📁](results/DiaPer/10attractors/DIHARD2/withFT/eval/rttms) | [44.51](results/DiaPer/20attractors/DIHARD2/withoutFT/eval/result_collar0.0)% [📁](results/DiaPer/20attractors/DIHARD2/withoutFT/eval/rttms) | [31.23](results/DiaPer/20attractors/DIHARD2/withFT/eval/result_collar0.0)% [📁](results/DiaPer/20attractors/DIHARD2/withFT/eval/rttms) | [26.67](results/baseline_VBx/16kHz/DIHARD2/eval/result_collar0.0)% [📁](results/baseline_VBx/16kHz/DIHARD2/eval/rttms) |
DIHARD 3 full | [34.21](results/DiaPer/10attractors/DIHARD3full/withoutFT/eval/result_collar0.0)% [📁](results/DiaPer/10attractors/DIHARD3full/withoutFT/eval/rttms) | [24.12](results/DiaPer/10attractors/DIHARD3full/withFT/eval/result_collar0.0)% [📁](results/DiaPer/10attractors/DIHARD3full/withFT/eval/rttms) | [34.82](results/DiaPer/20attractors/DIHARD3full/withoutFT/eval/result_collar0.0)% [📁](results/DiaPer/20attractors/DIHARD3full/withoutFT/eval/rttms) | [22.77](results/DiaPer/20attractors/DIHARD3full/withFT/eval/result_collar0.0)% [📁](results/DiaPer/20attractors/DIHARD3full/withFT/eval/rttms) | [20.28](results/baseline_VBx/16kHz/DIHARD3full/eval/result_collar0.0)% [📁](results/baseline_VBx/16kHz/DIHARD3full/eval/rttms) |
DipCo | [48.26](results/DiaPer/10attractors/DipCo/withoutFT/eval/result_collar0.25)% [📁](results/DiaPer/10attractors/DipCo/withoutFT/eval/rttms) | -- | [43.37](results/DiaPer/20attractors/DipCo/withoutFT/eval/result_collar0.25)% [📁](results/DiaPer/20attractors/DipCo/withoutFT/eval/rttms) | -- | [49.22](results/baseline_VBx/16kHz/DipCo/eval/result_collar0.25)% [📁](results/baseline_VBx/16kHz/DipCo/eval/rttms) |
Mixer6 | [21.03](results/DiaPer/10attractors/Mixer6/withoutFT/eval/result_collar0.25)% [📁](results/DiaPer/10attractors/Mixer6/withoutFT/eval/rttms) | [13.41](results/DiaPer/10attractors/Mixer6/withFT/eval/result_collar0.25)% [📁](results/DiaPer/10attractors/Mixer6/withFT/eval/rttms) | [18.51](results/DiaPer/20attractors/Mixer6/withoutFT/eval/result_collar0.25)% [📁](results/DiaPer/20attractors/Mixer6/withoutFT/eval/rttms) | [10.99](results/DiaPer/20attractors/Mixer6/withFT/eval/result_collar0.25)% [📁](results/DiaPer/20attractors/Mixer6/withFT/eval/rttms) | [35.60](results/baseline_VBx/16kHz/Mixer6/eval/result_collar0.25)% [📁](results/baseline_VBx/16kHz/Mixer6/eval/rttms) |
MSDWild | [35.69](results/DiaPer/10attractors/MSDWild/withoutFT/few.val/result_collar0.25)% [📁](results/DiaPer/10attractors/MSDWild/withoutFT/few.val/rttms) | [15.46](results/DiaPer/10attractors/MSDWild/withFT/few.val/result_collar0.25)% [📁](results/DiaPer/10attractors/MSDWild/withFT/few.val/rttms) | [25.07](results/DiaPer/20attractors/MSDWild/withoutFT/few.val/result_collar0.25)% [📁](results/DiaPer/20attractors/MSDWild/withoutFT/few.val/rttms) | [14.59](results/DiaPer/20attractors/MSDWild/withFT/few.val/result_collar0.25)% [📁](results/DiaPer/20attractors/MSDWild/withFT/few.val/rttms) | [16.86](results/baseline_VBx/16kHz/MSDWild/few.val/result_collar0.25)% [📁](results/baseline_VBx/16kHz/MSDWild/few.val/rttms) |
RAMC | [38.05](results/DiaPer/10attractors/RAMC/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/RAMC/withoutFT/test/rttms) | [21.11](results/DiaPer/10attractors/RAMC/withFT/test/result_collar0.0)% [📁](results/DiaPer/10attractors/RAMC/withFT/test/rttms) | [32.08](results/DiaPer/20attractors/RAMC/withoutFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/RAMC/withoutFT/test/rttms) | [18.69](results/DiaPer/20attractors/RAMC/withFT/test/result_collar0.0)% [📁](results/DiaPer/20attractors/RAMC/withFT/test/rttms) | [18.19](results/baseline_VBx/16kHz/RAMC/test/result_collar0.0)% [📁](results/baseline_VBx/16kHz/RAMC/test/rttms) |
VoxConverse | [23.20](results/DiaPer/10attractors/VoxConverse/withoutFT/test/result_collar0.25)% [📁](results/DiaPer/10attractors/VoxConverse/withoutFT/test/rttms) | -- | [22.10](results/DiaPer/20attractors/VoxConverse/withoutFT/test/result_collar0.25)% [📁](results/DiaPer/20attractors/VoxConverse/withoutFT/test/rttms) | -- | [6.12](results/baseline_VBx/16kHz/VoxConverse/test/result_collar0.25)% [📁](results/baseline_VBx/16kHz/VoxConverse/test/rttms) |


## Citation
In case of using the software, referencing results or finding the repository useful in any way please cite:
```
@article{landini2023diaper,
  title={DiaPer: End-to-End Neural Diarization with Perceiver-Based Attractors},
  author={Landini, Federico and Diez, Mireia and Stafylakis, Themos and Burget, Luk{\'a}{\v{s}}},
  journal={arXiv preprint arXiv:2312.04324},
  year={2023}
}
```
If you did not use it for a publication but still found it useful, also let me know by email, I would love to know too :)


## Contact
If you have comments or questions, please contact me at landini@fit.vutbr.cz
