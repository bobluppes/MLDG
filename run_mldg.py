import os

runs = 5
iterations = 3

for i in range(runs):
    for j in range(iterations):
        print('Run ', i, ' iteration ', j)
        os.system("python main_mldg.py train --lr=5e-4 --num_classes=7 --test_every=500 --logs='run_" + str(i) + "logs_mldg_" + str(j) + "' --batch_size=64 --model_path='run_" +
                  str(i) + "models_mldg_" + str(j) + "' --unseen_index=" + str(j) + " --inner_loops=45001 --step_size=15000 --state_dict='' --data_root='data' --meta_step_size=5e-1 --meta_val_beta=1.0")
