python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/clear /workspace/mnt/cache/NMA_data/fid_dataset/obscure
6组
python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/news/clear1 /workspace/mnt/cache/NMA_data/fid_dataset/news/obscore2
python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/news/clear1_1 /workspace/mnt/cache/NMA_data/fid_dataset/news/obscore2_2
python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/news/clear1 /workspace/mnt/cache/NMA_data/fid_dataset/news/obscore2_2
python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/news/clear1_1 /workspace/mnt/cache/NMA_data/fid_dataset/news/obscore2
python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/news/clear1 /workspace/mnt/cache/NMA_data/fid_dataset/news/clear1_1
python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/news/obscore2 /workspace/mnt/cache/NMA_data/fid_dataset/news/obscore2_2
python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/news/png1 /workspace/mnt/cache/NMA_data/fid_dataset/news/png1_1
python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/news/png2 /workspace/mnt/cache/NMA_data/fid_dataset/news/png2_2
python ./fid_score.py /workspace/mnt/cache/NMA_data/fid_dataset/news/png1 /workspace/mnt/cache/NMA_data/fid_dataset/news/png3
 可以加参数gpu  --gpu '0,1,2,3,5,6,7'

 eg:
 python ./fid_score_v1.py /workspace/mnt/cache/NMA_data/e-bike/cam_20465_梅园路与当湖路西向东2_20191220070000_20191220071500_20191220070000_20191220071500-19613-0002.jpg_23.jpg \
 /workspace/mnt/cache/NMA_data/e-bike/cam_20465_梅园路与当湖路西向东2_20191220010000_20191220011500_20191220005958_20191220011500-184233-0063.jpg_9.jpg --gpu '0,1,2,3,5,6,7'
tensor([0.9310]

nohup python -u ./fid_score_v1.py /workspace/mnt/cache/NMA_data/fid_dataset/news/png1 /workspace/mnt/cache/NMA_data/fid_dataset/news/png9 --gpu '0,1,2,3,5,6,7'>log.log 2>&1 &

20200603060139