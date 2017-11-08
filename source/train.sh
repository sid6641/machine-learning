DataSetDirName=faces
expName=test_1

mkdir -p /experiment/$DataSetDirName 

echo "---------------------------------------------"
echo "printing contents of /experiment/DataSetDirName "
ls -al /experiment/$DataSetDirName


echo "starting tar"
tar -xzf /experiment/train -C /experiment/$DataSetDirName/
echo "done with tar"

echo "---------------------------------------------"
echo "printing contents of /experiments/DataSetDirName "
ls -al /experiment/$DataSetDirName


python main.py > trainLog.txt

echo "********************** done with main.py **********************"

mv $expName/* /experiment/model/

mv trainLog.txt /experiment/model/trainLog.txt
mv weight.h5 /experiment/model/weight.h5
