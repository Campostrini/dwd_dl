from dwd_dl import cfg
import dwd_dl.mp4 as mp4
import sys
import os

if __name__ == "__main__":
    cfg.initialize(True, True)
    assert os.path.exists(os.path.abspath(sys.argv[1]))
    assert os.path.exists(os.path.dirname(os.path.abspath(sys.argv[2])))
    assert sys.argv[2].endswith('.mp4')

    mp4.save_mp4(start_date=cfg.CFG.VIDEO_START, end_date=cfg.CFG.VIDEO_END, path_to_saved_model=sys.argv[1],
                 path_to_mp4=sys.argv[2],)
