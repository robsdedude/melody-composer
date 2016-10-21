#!/bin/bash
set -e

# generates a 0.mid in each integer folder (using 10 training melodies)
# generates a 3.mid in each integer folder (using all training melodies)

python automatic_content_generation.py --order 4 -b 4 -l 10 -o final_res/0/0.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 10 -o final_res/1/0.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 10 -o final_res/2/0.mid -d
python automatic_content_generaion.py --order 4 -b 4 -l 10 -o final_res/3/0.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 10 -o final_res/4/0.mid -d

python automatic_content_generation.py --order 4 -b 4 -m 5 -o final_res/tmp/0.mid -d
mv final_res/tmp/00.mid final_res/0/3.mid
mv final_res/tmp/01.mid final_res/1/3.mid
mv final_res/tmp/02.mid final_res/2/3.mid
mv final_res/tmp/03.mid final_res/3/3.mid
mv final_res/tmp/04.mid final_res/4/3.mid
rm -r final_res/tmp
