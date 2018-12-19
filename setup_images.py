import os
import shutil
import glob
import random
import cnn_vgg16

parentdir=os.getcwd()
os.chdir('cnn')

cnndir = os.getcwd()
thingdir='101_things2'
newdir = os.path.join(cnndir,thingdir)
fulldir = os.path.join(cnndir, '101_ObjectCategories')
class_list =[]

if os.path.exists(newdir):
    shutil.rmtree(newdir, ignore_errors=True)

os.mkdir(newdir, 0o777)


for ff in os.listdir(fulldir):
    if random.random() > 0.66:
    #if str(ff).strip('/') in ['dragonfly', 'ibis', 'mayfly', 'pigeon', 'rooster']:
        for subfolder in ['test', 'train', 'valid']:
            targetpath = os.path.join(newdir, subfolder)
            if not os.path.exists(targetpath):
                os.mkdir(targetpath, 0o777)
            if subfolder in 'train':
                numfiles = 20
                offset = 1
            if subfolder in 'valid':
                numfiles = 10
                offset = 21
            if subfolder in 'test':
                numfiles = 10
                offset = 31

            if ff not in class_list:
                print('adding '.format(ff))
                class_list.append(str(ff).strip('/'))

            out_thing_folder = os.path.join(targetpath, ff)
            if not os.path.exists(out_thing_folder):
                os.mkdir(out_thing_folder, 0o777)

            for ii in range(numfiles):
                iioffset = ii + offset
                if len(str(iioffset))==1:
                    imagenum = '0{}'.format(iioffset)
                else:
                    imagenum = iioffset

                trainfile = os.path.join(fulldir, os.path.basename(ff), 'image_00{}.jpg'.format(imagenum))
                if os.path.exists(trainfile) and  os.path.exists(out_thing_folder):
                    shutil.copy(trainfile,out_thing_folder)
                else:
                    print('not enough files for {}->{}'.format(trainfile, out_thing_folder))
                    exit()
    else:
        print('should be skipping {}'.format(ff))

thing_set = '101_things'
thing_classes = ['accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain',
                 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone',
                 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head',
                 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu',
                 'euphonium', 'ewer', 'Faces', 'Faces_easy', 'ferry', 'flamingo', 'flamingo_head', 'garfield',
                 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis',
                 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'Leopards', 'llama', 'lobster',
                 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'Motorbikes', 'nautilus', 'octopus',
                 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster',
                 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler',
                 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella',
                 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

train_basesize = 20
valid_basesize = 8
test_basesize = 1

os.chdir(parentdir)

cnn_vgg16.runit(thingdir, class_list, train_basesize, valid_basesize, test_basesize)