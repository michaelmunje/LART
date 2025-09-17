import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import hydra
import joblib
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from phalp.configs.base import CACHE_DIR, FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.utils.utils import progress_bar
import yaml
import tqdm

# This script is based off the demo.py script from LART:
# https://github.com/brjathu/LART/blob/main/scripts/demo.py

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, lart_cfg) -> None:
        super().__init__(lart_cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

# create the tracker with hmr2 backend
class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2Predictor(self.cfg)

# create the tracker with action predictor
class LART(HMR2_4dhuman):
    def __init__(self, cfg):

        download_files = {
            "lart_mvit.config" : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/lart_mvit.config", os.path.join(CACHE_DIR, "phalp/ava")],
            "lart_mvit.ckpt"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/lart_mvit.ckpt", os.path.join(CACHE_DIR, "phalp/ava")],
            "mvit.yaml"        : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/mvit.yaml", os.path.join(CACHE_DIR, "phalp/ava")],
            "mvit.pyth"        : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/mvit.pyth", os.path.join(CACHE_DIR, "phalp/ava")],
        }
        self.cached_download_from_drive(download_files)
        super().__init__(cfg)

    def setup_predictor(self):
        # setup predictor model witch predicts actions from poses
        log.info("Loading Predictor model...")
        from lart.utils.wrapper_phalp import Pose_transformer
        self.pose_predictor = Pose_transformer(self.cfg, self)
        self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)


def render_lart(lart, phalp_pkl_path, only_render_lart=True):
    """
    Renders the lart object on each frame of a video and saves the rendered frames as a video.

    Args:
        lart (object): The lart object containing the postprocessor and lart_cfg.
        phalp_pkl_path (str): The path to the phalp pkl file.
        only_render_lart (bool, optional): If True, only the rendered lart object is saved as a video. 
            If False, the rendered lart object is concatenated with the original frame and saved as a video. 
            Defaults to True.
    """
    
    phalp_tracker = lart.postprocessor.phalp_tracker
    lart_cfg = lart.postprocessor.cfg
    
    video_pkl_name = phalp_pkl_path.split("/")[-1].split(".")[0]
    save_pkl_path = os.path.join(lart_cfg.video.output_dir, "results_temporal/", video_pkl_name + ".pkl")
    save_video_path = os.path.join(lart_cfg.video.output_dir, "results_temporal_videos/", video_pkl_name + "_.mp4")
    final_visuals_dic = joblib.load(save_pkl_path)
    
    video_pkl_name = save_pkl_path.split("/")[-1].split(".")[0]
    list_of_frames = list(final_visuals_dic.keys())
    
    for t_, frame_path in progress_bar(enumerate(list_of_frames), description="Rendering : " + video_pkl_name, total=len(list_of_frames), disable=False):
        
        image = phalp_tracker.io_manager.read_frame(frame_path)

        ################### Front view #########################
        lart_cfg.render.up_scale = int(lart_cfg.render.output_resolution / lart_cfg.render.res)
        phalp_tracker.visualizer.reset_render(lart_cfg.render.res*lart_cfg.render.up_scale)
        final_visuals_dic[frame_path]['frame'] = image
        panel_render, f_size = phalp_tracker.visualizer.render_video(final_visuals_dic[frame_path])      
        del final_visuals_dic[frame_path]['frame']

        # resize the image back to render resolution
        panel_rgb = cv2.resize(image, (f_size[0], f_size[1]), interpolation=cv2.INTER_AREA)

        # save the predicted actions labels
        if('label' in final_visuals_dic[frame_path]):
            labels_to_save = []
            for tid_ in final_visuals_dic[frame_path]['label']:
                ava_labels = final_visuals_dic[frame_path]['label'][tid_]
                labels_to_save.append(ava_labels)
            labels_to_save = np.array(labels_to_save)

        if only_render_lart:
            panel_1 = panel_render
        else:
            panel_1 = np.concatenate((panel_rgb, panel_render), axis=1)
        final_panel = panel_1
        

        phalp_tracker.io_manager.save_video(save_video_path, final_panel, (final_panel.shape[1], final_panel.shape[0]), t=t_)
        t_ += 1

    phalp_tracker.io_manager.close_video()

@hydra.main(version_base="1.2", config_name="config")
def main(lart_cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""

    # # Setup the tracker and track the video
    autolabel_cfg_path = lart_cfg.autolabel_cfg
    config = yaml.load(open(autolabel_cfg_path, "r"), Loader=yaml.FullLoader)
    
    lart_cfg.video.output_dir = os.path.join(config['lart_data_directory'], config['lart_raw_data_foldername'])
    
    lart_cfg.phalp.low_th_c = config['phalp_low_th_c']
    lart_cfg.phalp.small_w = config['phalp_small_w']
    lart_cfg.phalp.small_h = config['phalp_small_h']
    
    # fx, fy = config['camera_matrix']['fx'], config['camera_matrix']['fy']
    # focal = int((fx + fy) / 2)
    # lart_cfg.EXTRA.FOCAL_LENGTH = focal
    
    # make a copy of the config file
    lart_cfg = lart_cfg.copy()
    
    video_names_to_process = ['/u/mmunje/scratch/vlm-sn/scand_spot/videos/34_Spot.mp4']

    for video_name in tqdm.tqdm(video_names_to_process):
        # make a copy of the config file
        lart_cfg_bkup = lart_cfg.copy()
        lart_cfg.video.source = video_name
        vidcap = cv2.VideoCapture(lart_cfg.video.source)
        lart_cfg.video.end_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # let's also get image resolution
        lart_cfg.render.output_resolution = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vidcap.release()
        
        lart_cfg.render.enable = True
        phalp_tracker = HMR2_4dhuman(lart_cfg)
        _, pkl_path = phalp_tracker.track()
        del phalp_tracker

        # Setup the LART model and run it on the tracked video to get the action predictions
        
        lart_cfg.render.enable = True
        lart_cfg.render.colors = 'slahmr'
        lart_cfg.render.type = "GHOST_MESH"
        lart_cfg.pose_predictor.config_path = f"{CACHE_DIR}/phalp/ava/lart_mvit.config"
        lart_cfg.pose_predictor.weights_path = f"{CACHE_DIR}/phalp/ava/lart_mvit.ckpt"
        lart_cfg.post_process.save_fast_tracks = True
        lart_model = LART(lart_cfg)
        lart_model.setup_postprocessor()
        lart_model.postprocessor.run_lart(pkl_path)
        
        render_lart(lart_model, pkl_path, only_render_lart=True)

        lart_cfg = lart_cfg_bkup

if __name__ == "__main__":
    main()
