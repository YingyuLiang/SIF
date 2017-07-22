DATADIR=./data
MULTINLI=$DATADIR/multinli_0.9
SNLI=$DATADIR/snli_1.0

# download multinli 
if [ ! -d $MULTINLI ]; then
wget http://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip 
unzip multinli_0.9.zip -d $DATADIR
rm multinli_0.9.zip
fi

# download snli 
if [ ! -d $SNLI ]; then
wget http://www.nyu.edu/projects/bowman/multinli/snli_1.0.zip
unzip snli_1.0.zip -d $DATADIR
rm snli_1.0.zip
fi




