
function unpack {
    root=$PWD
    cd $1
    for lang in *
    do
        for char in $lang/*
        do
            mv $char ${char/\//_}
        done
        rm -r $lang
    done
    cd $root
}

echo Downloading...
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip

echo Unpacking...
unzip images_background.zip
unzip images_evaluation.zip
mv images_background omniglot_background
mv images_evaluation omniglot_evaluation
unpack omniglot_background
unpack omniglot_evaluation

echo Processing...
python preprocess.py omniglot_background
python preprocess.py omniglot_evaluation

echo Cleaning up...
rm images_background.zip images_evaluation.zip
