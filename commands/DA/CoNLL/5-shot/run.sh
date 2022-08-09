python train_demo.py --ID DA_conll_0 --gpu 1  --task domain-transfer --mode conll  --K 5 --support_num 0 --averaged_times 3 --batch_size 8  --lr 5e-5  --adapt_step 10  --c2c_tau 1000 
python train_demo.py --ID DA_conll_1 --gpu 1  --task domain-transfer --mode conll  --K 5 --support_num 1 --averaged_times 3 --batch_size 8  --lr 5e-5  --adapt_step 10  --c2c_tau 1000 
python train_demo.py --ID DA_conll_2 --gpu 2  --task domain-transfer --mode conll  --K 5 --support_num 2 --averaged_times 3 --batch_size 8  --lr 5e-5  --adapt_step 10  --c2c_tau 1000
python train_demo.py --ID DA_conll_3 --gpu 2  --task domain-transfer --mode conll  --K 5 --support_num 3 --averaged_times 3 --batch_size 8  --lr 5e-5  --adapt_step 10  --c2c_tau 1000 
python train_demo.py --ID DA_conll_4 --gpu 1  --task domain-transfer --mode conll  --K 5 --support_num 4 --averaged_times 3 --batch_size 8  --lr 5e-5  --adapt_step 10  --c2c_tau 1000
