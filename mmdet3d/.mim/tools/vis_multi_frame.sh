# python tools/vis_frame.py work_dirs/out/baseline.pkl configs/bevdet_occ_self/bevdet-occ-ds.py --frame-idx $1 --save-path out/frames/baseline --save-same-path &
# python tools/vis_frame.py work_dirs/out/fbnew.pkl configs/bevdet_occ_self/bevdet-occ-ds.py --frame-idx $1 --save-path out/frames/fb --save-same-path &
# python tools/vis_frame.py work_dirs/out/long_ema_24e.pkl configs/bevdet_occ_self/bevdet-occ-ds.py --frame-idx $1 --save-path out/frames/ours --save-same-path --vis-gt

for i in {3700..4000..20}
do
    python tools/vis_frame.py work_dirs/out/long_ema_24e.pkl configs/bevdet_occ_self/bevdet-occ-ds.py --frame-idx $i --save-path out/ours/$i --save-same-path --vis-gt
done
echo "finish"