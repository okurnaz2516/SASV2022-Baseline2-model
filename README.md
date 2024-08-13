# SASV_PR

## [A Probabilistic Fusion Framework for Spoofing Aware Speaker Verification](https://www.isca-speech.org/archive/odyssey_2022/zhang22b_odyssey.html)
[[PDF](https://www.isca-speech.org/archive/pdfs/odyssey_2022/zhang22b_odyssey.pdf)][[Video](https://www.youtube.com/watch?v=98p-KLH3cKc)][[Slides](https://labsites.rochester.edu/air/publications/Zhang22Odyssey.pdf)]

### Data preparation
Get the pre-trained ASV and CM embeddings from [here](https://drive.google.com/drive/folders/1kn_slob4BGHPmk_O8HaAiJE5P6qqBFV7?usp=sharing).

Our embedding extractions process is the same as in the [baseline repository](https://github.com/sasv-challenge/SASVC2022_Baseline) (commit 8bfbf1f3f7373). Please refer to their `save_embeddings.py`. The ASV subsystem is from [here](https://github.com/TaoRuijie/ECAPA-TDNN/tree/a2290930b910a3cba7e099d2447d02c18919b3a4) and CM subsystem from [here](https://github.com/clovaai/aasist/tree/a04c9863f63d44471dde8a6abcb3b082b07cd1d1).

### Run our code
Our code is based on PyTorch.

Before running, you need to specify where you would like to save the model by `-o`, you also need to specify which model you want to use by `-m`, the default is `pr_s_f`.

The options for `-m` are: `pr_l_i`, `pr_s_i`, `pr_c_i`, `pr_l_f`, `pr_s_f` if you would like to use our methods. Please refer to our paper for more details.

```
python3 main_train.py -o ./exp_result/ -m pr_s_f
```

### Citation
```
@inproceedings{zhang2022prob,
  author={You Zhang and Ge Zhu and Zhiyao Duan},
  title={{A Probabilistic Fusion Framework for Spoofing Aware Speaker Verification}},
  year=2022,
  booktitle={Proc. The Speaker and Language Recognition Workshop (Odyssey)},
  pages={77--84},
  doi={10.21437/Odyssey.2022-11}
}
```