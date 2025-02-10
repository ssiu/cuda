#!/bin/bash


ncu -f --target-processes all --set full \
--import-source on \
-o profile_flash_attn python test.py