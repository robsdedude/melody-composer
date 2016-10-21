#!/bin/bash
set -e

# generates a 1.mid in each integer folder (using 20 training melodies)
# generates a 2.mid in each integer folder (using 50 training melodies)

python automatic_content_generation.py --order 4 -b 4 -l 20 -o final_res/0/1.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 20 -o final_res/1/1.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 20 -o final_res/2/1.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 20 -o final_res/3/1.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 20 -o final_res/4/1.mid -d

python automatic_content_generation.py --order 4 -b 4 -l 50 -o final_res/0/2.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 50 -o final_res/1/2.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 50 -o final_res/2/2.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 50 -o final_res/3/2.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 50 -o final_res/4/2.mid -d

