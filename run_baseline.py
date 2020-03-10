import os

runs = 5
iterations = 3

for i in range(runs):
    for j in range(iterations):
        print('Run ', i, ' iteration ', j)
        os.system('python main_baseline.py train')