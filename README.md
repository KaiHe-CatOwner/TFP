# TFP

## Setup

### 1. Create conda environment
```
conda create -n fsl-ner -y python=3.7 && conda activate fsl-ner
```
### 2. Install dependecies
Install the required packages
```
pip install -r requirements.txt
```

### 3. Run code
(1) LE setting :  python train_demo.py --ID 0 --gpu 0 --task few-nerd --mode intra --N 5 --K 1 --averaged_times 3 --batch_size 16 --test_bz 1 --lr 5e-5  --adapt_step 15  --c2c_tau 1000  

```
(1) LE setting : python train_demo.py --ID 1 --gpu 1  --task domain-transfer --mode conll  --K 1 --support_num 0 --averaged_times 3 --batch_size 13  --lr 5e-5  --adapt_step 10  --c2c_tau 10000  

```
(1) LE setting : python train_demo.py --ID 5-movie_01 --gpu 1 --task in-label-space --mode mit-movie --K 5 --averaged_times 3 --batch_size 4 --test_bz 4 --support_num 1  --lr 5e-5  --adapt_step 15  --val_interval 50 --c2c_tau 100 

```

### 4. Tips

(1) All used data can be downloaded :
https://drive.google.com/file/d/1WD1ZnoG9TIOzr0QfslDd-VfBoshy_wFJ/view?usp=sharing

(2) All scripts for runs are in dircetory commands.
