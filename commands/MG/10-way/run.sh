 python train_demo.py --ID few_NERD_intra_02 --gpu 2 --task few-nerd --mode intra --N 10 --K 1 --averaged_times 3 --batch_size 16  --lr 5e-5  --adapt_step 15  --c2c_tau 1000  
 python train_demo.py --ID few_NERD_inter_02 --gpu 2 --task few-nerd --mode inter --N 10 --K 1 --averaged_times 3 --batch_size 16  --lr 5e-5  --adapt_step 15  --c2c_tau 1000  

 python train_demo.py --ID few_NERD_intra_10-0 --gpu 1 --task few-nerd --mode intra --N 10 --K 5 --averaged_times 3 --batch_size 14  --lr 5e-5  --adapt_step 15  --c2c_tau 10000  
 python train_demo.py --ID few_NERD_inter_10-1 --gpu 0 --task few-nerd --mode inter --N 10 --K 5 --averaged_times 3 --batch_size 14  --lr 5e-5  --adapt_step 15  --c2c_tau 10000  


