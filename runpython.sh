srun --killable --mem=400m -c2 --time=3-0 --gres=gpu:2 python3 $@