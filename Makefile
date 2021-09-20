define GENERATE_CSV_SCRIPT
import os\n
import pandas as pd\n

pd.DataFrame({\n
    "images": [f"images/{path_to_image}" for path_to_image in os.listdir("data/images")],\n
    "masks": [f"masks/{path_to_mask}" for path_to_mask in os.listdir("data/masks")],\n
}).to_csv("data/train.csv", index=False)\n
endef

SCRIPT_RUNNER = $(shell echo '${GENERATE_CSV_SCRIPT}' | python3)

prepare_csv:
	@echo $(SCRIPT_RUNNER)

all: exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 exp9

exp1:
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 0 --pin-memory false > num_workers-0_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 1 --pin-memory false > num_workers-1_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 1 --pin-memory true > num_workers-1_pin_mem-true.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 2 --pin-memory false > num_workers-2_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 2 --pin-memory true > num_workers-2_pin_mem-true.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 4 --pin-memory false > num_workers-4_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 4 --pin-memory true > num_workers-4_pin_mem-true.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 6 --pin-memory false > num_workers-6_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 6 --pin-memory true > num_workers-6_pin_mem-true.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 8 --pin-memory false > num_workers-8_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 8 --pin-memory true > num_workers-8_pin_mem-true.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 10 --pin-memory false > num_workers-10_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 10 --pin-memory true > num_workers-10_pin_mem-true.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 12 --pin-memory false > num_workers-12_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 12 --pin-memory true > num_workers-12_pin_mem-true.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 14 --pin-memory false > num_workers-14_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 14 --pin-memory true > num_workers-14_pin_mem-true.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 16 --pin-memory false > num_workers-16_pin_mem-false.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 16 --pin-memory true > num_workers-16_pin_mem-true.log

exp2:
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 6 --pin-memory true --detect-anomaly True > anomaly_detection-true.log

exp3:
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 6 --pin-memory true --enable-bias-decoder False > no_bias_decoder.log

exp4:
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 6 --pin-memory true --enable-bias-decoder False > regular_train_memory_usage.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 6 --pin-memory true --enable-bias-decoder False --speedup-zero-grad True > speed_up_zero_grad.log

exp5:
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 24 --num-workers 6 --pin-memory true --enable-bias-decoder False > increased_batch_size.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 25 --num-workers 6 --pin-memory true --enable-bias-decoder False --enable-gradient-checkpointing True > increase_batch_size_with_gradient_checkpointing.log

exp6:
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 6 --pin-memory true --enable-bias-decoder False --enable-cudnn-benchmark True > use_cudnn_benchmark.log

exp7:
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 6 --pin-memory true --enable-bias-decoder False --enable-cudnn-benchmark True --mixed-precision-mode O1 > cudnn_bench_mp_O1.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 16 --num-workers 6 --pin-memory true --enable-bias-decoder False --enable-cudnn-benchmark True --mixed-precision-mode O2 > cudnn_bench_mp_O2.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 32 --num-workers 12 --pin-memory true --enable-bias-decoder False --enable-cudnn-benchmark True --mixed-precision-mode O1 > bs32_cudnn_bench_mp_O1.log
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 32 --num-workers 12 --pin-memory true --enable-bias-decoder False --enable-cudnn-benchmark True --mixed-precision-mode O2 > bs32_cudnn_bench_mp_O2.log

exp8:
	docker run -it --rm --gpus device=1 -v `pwd`/data:/data --ipc=host experiment python train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 32 --num-workers 12 --pin-memory true --enable-bias-decoder False --enable-cudnn-benchmark True --mixed-precision-mode O2 --use-fused-adam True > fused_optimizer.log

exp9:
	docker run -it --rm --gpus all -v `pwd`/data:/data --ipc=host experiment python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 train.py --df-path /data/train.csv --root-path /data/ --pretrained-model true --epochs 30 --batch-size 32 --num-workers 12 --pin-memory true --enable-bias-decoder False --mixed-precision-mode O2 --use-fused-adam True > dist_training.log


