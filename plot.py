import matplotlib.pyplot as  plt 
import numpy as np
import pickle as pkl 
import sys 
import os 

file = sys.argv[1]

path = file[:-26] + 'figures_' + file[-18:-4]
fontsize = 15
try:
    os.makedirs(path, exist_ok=True)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

with open(file, 'rb') as f:
	proportions_of_added_pixels_, accuracy_metrics_ = pkl.load(f)

for metric in ['OA', 'mIoU']:
	fig = plt.figure()
	plt.plot(accuracy_metrics_[metric])
	plt.ylabel(metric, fontsize=fontsize)
	plt.xlabel('Steps', fontsize=fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	fig.savefig(path + '/{}.png'.format(metric), dpi=300, bbox_inches = 'tight', pad_inches = 0.05)


color_1 = (255/255, 189/255, 0/255)
color_2 = (57/255, 0/255, 153/255)
color_3 = (158/255, 0/255, 89/255)

fig = plt.figure()
n_steps = proportions_of_added_pixels_.shape[1]
steps = [(n_steps-1)//3, 2*(n_steps-1)//3, n_steps-1]
x = np.arange(proportions_of_added_pixels_.shape[0])
y_lim = proportions_of_added_pixels_[:,steps[2]].max()+0.02
plt.bar(x, height=proportions_of_added_pixels_[:,steps[2]], color=color_3, alpha=1)
plt.bar(x, height=proportions_of_added_pixels_[:,steps[1]], color=color_2, alpha=1)
plt.bar(x, height=proportions_of_added_pixels_[:,steps[0]], color=color_1, alpha=1)
plt.ylim(0, y_lim)
plt.xticks(x, x, rotation=45, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('Proportions of added labels', fontsize=fontsize)
plt.xlabel('Classes', fontsize=fontsize)
plt.savefig(path + '/added_classes.png', dpi=300, bbox_inches = 'tight', pad_inches = 0.05)