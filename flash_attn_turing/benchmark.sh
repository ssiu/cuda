#!/bin/bash


ncu -f --target-processes all --set full \
--import-source on \
-o profile_sm75_gemm python test.py