python train_demo.py --ID DA_wnut_0 --gpu 0  --task domain-transfer --mode wnut  --K 5 --support_num 0 --averaged_times 3 --batch_size 8  --lr 5e-5  --adapt_step 15  --c2c_tau 1000 
python train_demo.py --ID DA_wnut_1 --gpu 0  --task domain-transfer --mode wnut  --K 5 --support_num 1 --averaged_times 3 --batch_size 4  --lr 5e-5  --adapt_step 15  --c2c_tau 1000 
python train_demo.py --ID DA_wnut_2 --gpu 1  --task domain-transfer --mode wnut  --K 5 --support_num 2 --averaged_times 3 --batch_size 8  --lr 5e-5  --adapt_step 15  --c2c_tau 1000
python train_demo.py --ID DA_wnut_3 --gpu 2  --task domain-transfer --mode wnut  --K 5 --support_num 3 --averaged_times 3 --batch_size 8  --lr 5e-5  --adapt_step 15  --c2c_tau 1000 
python train_demo.py --ID DA_wnut_4 --gpu 3  --task domain-transfer --mode wnut  --K 5 --support_num 4 --averaged_times 3 --batch_size 8  --lr 5e-5  --adapt_step 15  --c2c_tau 1000
