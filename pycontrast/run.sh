CUDA_VISIBLE_DEVICES=1,2,3 python main_contrast.py --method MoCo --cosine --epochs 100 --batch_size 128 --multiprocessing-distributed --world-size 1 --rank 0
