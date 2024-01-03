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

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
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
    phalp_tracker = lart.postprocessor.phalp_tracker
    cfg = lart.postprocessor.cfg
    
    video_pkl_name = phalp_pkl_path.split("/")[-1].split(".")[0]
    save_pkl_path = os.path.join(cfg.video.output_dir, "results_temporal/", video_pkl_name + ".pkl")
    save_video_path = os.path.join(cfg.video.output_dir, "results_temporal_videos/", video_pkl_name + "_.mp4")
    final_visuals_dic = joblib.load(save_pkl_path)
    
    video_pkl_name = save_pkl_path.split("/")[-1].split(".")[0]
    list_of_frames = list(final_visuals_dic.keys())
    
    for t_, frame_path in progress_bar(enumerate(list_of_frames), description="Rendering : " + video_pkl_name, total=len(list_of_frames), disable=False):
        
        image = phalp_tracker.io_manager.read_frame(frame_path)

        ################### Front view #########################
        cfg.render.up_scale = int(cfg.render.output_resolution / cfg.render.res)
        phalp_tracker.visualizer.reset_render(cfg.render.res*cfg.render.up_scale)
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
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""

    # # Setup the tracker and track the video
    cfg.phalp.low_th_c = 0.8
    # cfg.phalp.max_age_track = 90
    cfg.phalp.small_w = 25
    cfg.phalp.small_h = 50
    
    vidcap = cv2.VideoCapture(cfg.video.source)
    cfg.video.end_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # let's also get image resolution
    cfg.render.output_resolution = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidcap.release()
    
    cfg.render.enable = False
    phalp_tracker = HMR2_4dhuman(cfg)
    _, pkl_path = phalp_tracker.track()
    del phalp_tracker

    # Setup the LART model and run it on the tracked video to get the action predictions
    
    # not rendering of video, we use a custom renderer
    cfg.render.enable = False
    cfg.render.colors = 'slahmr'
    cfg.render.type = "GHOST_MESH"
    cfg.pose_predictor.config_path = f"{CACHE_DIR}/phalp/ava/lart_mvit.config"
    cfg.pose_predictor.weights_path = f"{CACHE_DIR}/phalp/ava/lart_mvit.ckpt"
    cfg.post_process.save_fast_tracks = True
    lart_model = LART(cfg)
    lart_model.setup_postprocessor()
    lart_model.postprocessor.run_lart(pkl_path)
    render_lart(lart_model, pkl_path, only_render_lart=True)

if __name__ == "__main__":
    main()
