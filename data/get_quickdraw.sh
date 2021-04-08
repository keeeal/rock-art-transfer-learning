# requires gsutil (sudo snap install google-cloud-sdk)

#echo Downloading...
mkdir quickdraw_background quickdraw_evaluation
gsutil -m cp gs://quickdraw_dataset/full/simplified/*.ndjson quickdraw_background

echo Unpacking...
python rasterize.py quickdraw_background -n 1000
mv quickdraw_background/*.ndjson quickdraw_evaluation/
python rasterize.py quickdraw_evaluation -n 20

echo Processing...
python preprocess.py quickdraw_background
python preprocess.py quickdraw_evaluation

echo Cleaning up...
rm -r quickdraw_evaluation/*.ndjson
