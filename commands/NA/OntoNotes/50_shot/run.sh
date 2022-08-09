python train_demo.py --ID 50-ontonotes_1 --gpu 0 --task in-label-space --mode ontonotes --K 50 --averaged_times 3 --batch_size 1 --test_bz 1 --support_num 1  --lr 5e-5   --adapt_step 15  --val_interval 100 --c2c_tau 1000
python train_demo.py --ID 50-ontonotes_2 --gpu 2 --task in-label-space --mode ontonotes --K 50 --averaged_times 3 --batch_size 1 --test_bz 1 --support_num 2  --lr 5e-5   --adapt_step 15  --val_interval 100 --c2c_tau 1000 
python train_demo.py --ID 50-ontonotes_3 --gpu 3 --task in-label-space --mode ontonotes --K 50 --averaged_times 3 --batch_size 1 --test_bz 1 --support_num 3  --lr 5e-5   --adapt_step 15  --val_interval 100 --c2c_tau 1000 

