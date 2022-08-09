 python train_demo.py --ID few_NERD_intra_3 --gpu 0 --task few-nerd --mode intra --N 5 --K 1 --averaged_times 3 --batch_size 16 --test_bz 1 --lr 5e-5  --adapt_step 15  --c2c_tau 1000  
 python train_demo.py --ID few_NERD_inter_3 --gpu 1 --task few-nerd --mode inter --N 5 --K 1 --averaged_times 3 --batch_size 16 --test_bz 1 --lr 5e-5  --adapt_step 15  --c2c_tau 1000  

 python train_demo.py --ID few_NERD_intra_0 --gpu 0 --task few-nerd --mode intra --N 5 --K 5 --averaged_times 3 --batch_size 16  --test_bz 1 --lr 5e-5  --adapt_step 15  --c2c_tau 10000  
 python train_demo.py --ID few_NERD_inter_0 --gpu 1 --task few-nerd --mode inter --N 5 --K 5 --averaged_times 3 --batch_size 16  --test_bz 1 --lr 5e-5  --adapt_step 15  --c2c_tau 10000  
