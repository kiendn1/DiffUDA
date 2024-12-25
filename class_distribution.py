import glob
import matplotlib.pyplot as plt
import numpy as np

r = {}
sum_files = 0
for i in sorted(glob.glob('/home/user/code/DiffUDA/Office Home/Real/*')):
    name_class = i.split('/')[-1]
    count = len(glob.glob(i+'/*'))
    r[name_class] = count
    sum_files += count

for name_class in r.keys():
    r[name_class] = float(r[name_class]/sum_files)

num_images_per_class = np.array(list(r.values()))

plt.figure(figsize=(6, 8))  # Setting the figure size to match the provided image
plt.bar(range(65), num_images_per_class)
plt.title('Number of Images per Class in Office-Home (Real)')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.savefig('experiment/distribution/real_distribution.png')