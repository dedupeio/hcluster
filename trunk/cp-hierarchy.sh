#!/bin/sh

if [ -z "$1" ];
then
    echo No scipy source directory specified. EXITING!
    exit
fi

DEST=hcluster/

cp $1/scipy/cluster/hierarchy.py $DEST
cp $1/scipy/cluster/src/hierarchy_wrap.c $DEST
cp $1/scipy/cluster/src/common.h $DEST
cp $1/scipy/cluster/src/hierarchy.c $DEST
cp $1/scipy/cluster/src/hierarchy.h $DEST
cp $1/scipy/cluster/tests/test_hierarchy.py $DEST/tests/
cp $1/scipy/cluster/tests/*.txt $DEST/tests/

sed -i 's/import scipy.spatial.distance as distance/import hcluster.distance/g' $DEST/hierarchy.py
sed -i 's/from scipy.cluster.hierarchy /from hcluster.hierarchy/g/' $DEST/tests/test_hierarchy.py
sed -i 's/from scipy.spatial.distance /from hcluster.distance/g/' $DEST/tests/test_hierarchy.py

cp $1/scipy/spatial/distance.py $DEST
cp $1/scipy/spatial/src/distance_wrap.c $DEST
cp $1/scipy/spatial/src/common.h $DEST
cp $1/scipy/spatial/src/distance.c $DEST
cp $1/scipy/spatial/src/distance.h $DEST
cp $1/scipy/spatial/tests/test_distance.py $DEST/tests/
cp $1/scipy/spatial/tests/*.txt $DEST/tests/

sed -i 's/from scipy.spatial.distance /from hcluster.distance/g/' $DEST/tests/test_distance.py