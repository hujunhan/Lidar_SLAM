## Read ground truth data from csv file into a list 
## And plot the ground truth data using matplotlib

import csv
import matplotlib.pyplot as plt
import numpy as np
## csv file path
csv_path = './data/college/ground_truth/registered_poses.csv'
reader = csv.reader(open(csv_path, 'r'))

pose_list = []

## Print the first 10 lines of csv file
# for i in range(10):
#     row = next(reader)
#     print(row)

## Pass the first line
next(reader)

## Read the csv file into a list
for row in reader:
    pose_list.append(row[2:4])

pose_list=np.asarray(pose_list)
pose_list=pose_list.astype(np.float32)

plt.scatter(pose_list[:,0],pose_list[:,1],marker='.')
plt.title(f'ground truth data')
plt.axis('equal')
plt.show()
