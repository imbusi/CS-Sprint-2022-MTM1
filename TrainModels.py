import os
import sys
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from MobileNetV2 import MobileNet as MN
from ResNet import ResNet as RN


start = time()
ROOT_DIR = os.getcwd()

training_path = os.path.join(ROOT_DIR, 'Images')
vs = 0.2
epochs = 5

labelDictionary = dict()
class_names = os.listdir(training_path)
class_names.sort()
for i in range(len(class_names)):
	labelDictionary[i] = class_names[i]

print('Here is your class Dictionary:', labelDictionary)

model = input('Enter what model you want to use (MobileNet, ResNet): ')
if model == 'MobileNet':
	print('\nTraining the MobileNetV2 Model...')
	Model = MN(training_path, vs, epochs, labelDictionary)
	print('MobileNet Model Returned...\n')
	path = os.path.join(ROOT_DIR,'SavedModels', 'MobileNetV2')
elif model == 'ResNet':
	print('\nTraining the ResNet50 Model...')
	Model = RN(training_path, vs, epochs, labelDictionary)
	print('ResNet50 Model Returned...\n')
	path = os.path.join(ROOT_DIR,'SavedModels', 'ResNet')
else:
	print('Wrong input.')
	sys.exit()




h5_file = os.path.join(path, 'model.h5')
Model.save(h5_file)

print('\nSaving model to {} folder...\n'.format(path))

f = open(os.path.join(path, 'classes.txt'), 'w')
for cl in class_names:
    f.write(str(cl) + ' ')
f.write('\n')
if model == 'MobileNet':
	f.write('MobileNetV2')
else:
	f.write('ResNet')
f.write('\n')
f.write(str(epochs))
f.close()


end = time()
print('Total time of execution: {:.3f} seconds.'.format(end-start))
print('\nDone!')
