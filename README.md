to use this, download the images from: http://www.vision.caltech.edu/Image_Datasets/Caltech101/

put them in a folder in ./cnn/101_ObjectCategories

i.e. ./cnn/101_ObjectCategories/airplanes/image_0009.jpg

note: the code, as written, assumes 40 images per class/category. The data from caltech101 do not all have 40 folders. you can either add more images to reach 40 in the ones that are missing images, or simply delete those folders.

edit then run ./setup_images.py

you probably want to modify the output thingdir (line 11), and adjust the percentage of classes to include (line 23).

finally, you may need to adjust the test_baseline value on line 79. I run out of memory on my 1070 Ti when I run more then around 30 classes and more than a test_basesize of 1.

