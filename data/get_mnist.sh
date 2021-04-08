
echo Downloading...
wget https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz

echo Unpacking...
tar -xzf mnist_png.tar.gz
mv mnist_png/training mnist_background
mv mnist_png/testing mnist_evaluation

echo Processing...
python preprocess.py mnist_background
python preprocess.py mnist_evaluation

echo Cleaning up...
rm -r mnist_png mnist_png.tar.gz
