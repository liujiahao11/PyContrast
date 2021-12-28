CUDA_VISIBLE_DEVICES=0,1,2,3 python main_contrast.py --method MoCo --cosine --epochs 200 --batch_size 256 --multiprocessing-distributed --world-size 1 --rank 0
