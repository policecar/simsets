#!/bin/bash

find . -name *.pyc -print | xargs rm
find . -name *.pyc -print | xargs rm

sed -i 's/local/server/g' load_matrices.py
sed -i 's/nodes/svo/g' load_matrices.py

sed -i 's/local/server/g' load_matrices_ctx.py
sed -i 's/nodes/svo/g' load_matrices_ctx.py

sed -i 's/local/server/g' load_pred_matrices.py
sed -i 's/nodes/svo/g' load_pred_matrices.py

