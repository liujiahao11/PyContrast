CUDA_VISIBLE_DEVICES=0,5,6 python main_linear.py --method MoCo --ckpt ./save/MoCo_resnet50_RGB_Jig_False_moco_aug_A_linear_0.07_cosine/ckpt_epoch_100.pth --epochs 100 --batch_size 128 --multiprocessing-distributed --world-size 1 --rank 0
