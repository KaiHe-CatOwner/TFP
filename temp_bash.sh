


srun -p PV1003q  -w node15 python  train_demo.py --ID test_description --gpu 1 --batch_size 4  --prompt_semantic 'description' --load_ckpt  /export/home/hk52025804/workshop_NTU/FS_NER/checkpoint/test_description-in-label-space-conll-4-5-seed42.pt


srun -p PV1003q  -w node15 python  train_demo.py --ID test_description --gpu 1 --batch_size 4  --prompt_semantic 'description' --load_ckpt  /export/home/hk52025804/workshop_NTU/FS_NER/checkpoint/test_description-in-label-space-conll-4-5-seed42.pt
srun -p PV1003q  -w node15 python  train_demo.py --ID test_mismatch    --gpu 1 --batch_size 4  --prompt_semantic 'mismatch' --load_ckpt  /export/home/hk52025804/workshop_NTU/FS_NER/checkpoint/test_mismatch-in-label-space-conll-4-5-seed42.pt


srun -p PV1003q  -w node15 python  train_demo.py --ID test_random      --gpu 2 --batch_size 4  --prompt_semantic 'random' --load_ckpt /export/home/hk52025804/workshop_NTU/FS_NER/checkpoint/test_random-in-label-space-conll-4-5-seed42.pt 


