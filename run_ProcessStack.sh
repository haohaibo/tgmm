#!/bin/bash
path=/home/hhb/work/tgmm-hhb-m/nmeth.3036-S2
parallel -j5 $path/build/nucleiChSvWshedPBC/ProcessStack {1} {2} ::: $path/data/TGMM_configFile_linux.txt ::: {0..30}
