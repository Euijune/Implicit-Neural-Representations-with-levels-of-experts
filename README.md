# Implicit-Neural-Representations-with-levels-of-experts
Implement INR with LoE (python)

```
!python train.py --coord_batch_size {your_image_size} --epoch 1001 --steps_til_summary 50\
--img_path {your_image_path} --save_dir {your_result_dir_path} --model_name {your_model_name}
```

example
```
!python train.py --coord_batch_size 65536 --epoch 1001 --steps_til_summary 50\
--img_path cameraman --save_dir experiment/capstone --model_name cameraman_test
```
