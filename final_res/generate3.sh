#!/bin/bash
set -e

# generates a 4.mid and a 5.mid in each integer folder (using 1 training melody)

python automatic_content_generation.py --order 4 -b 4 -l 1 -m 2 -o final_res/0/4.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 1 -m 2 -o final_res/1/4.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 1 -m 2 -o final_res/2/4.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 1 -m 2 -o final_res/3/4.mid -d
python automatic_content_generation.py --order 4 -b 4 -l 1 -m 2 -o final_res/4/4.mid -d

for n in 0 1 2 3 4; do
    mv final_res/$n/40.mid final_res/$n/4.mid
    mv final_res/$n/41.mid final_res/$n/5.mid
done

