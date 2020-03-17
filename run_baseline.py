import os

runs = 5
iterations = 3

for i in range(runs):
    for j in range(iterations):
        print('Run ', i, ' iteration ', j)
        os.system("python main_baseline.py train --lr=5e-4 --num_classes=7 --test_every=500 --logs='run_" + i + "'logs_" + j + "' --batch_size=64 --model_path='run_" + i + "models_" + j + "' --unseen_index=" + j + " --inner_loops=45001 --step_size=15000 --state_dict='' --data_root='/data'")