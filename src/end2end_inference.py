import torch
import yaml
import argparse
import importlib
import subprocess
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import font_manager
from transformers import CLIPProcessor, CLIPModel

from src.utils.visualization import vis_result_v2


device = torch.device('cuda')
FRAME_INTERVAL = 8
# UCF-Crime
# ucf_cls_defs = {
#     'Normal': 'Normal behavior, views or surveillance content captured by camera that not involve any unexpected or unusual events.',
#     'Abuse': 'Intentional beating or abuse of animals like dogs.',
#     'Arrest': 'Police arresting suspects, which may involve pressing them to the ground, controlling hands or aiming with guns.',
#     'Arson': 'The deliberate setting of a fire by someone, usually characterized by flames, smoke, puring gasoline.',
#     'Assault': 'Multiple people surround and assault one person with fists and cudgels.',
#     'Burglary': 'Burglary, usually characterized by crossing the cashier, breaking doors and windows, and carry things.',
#     'Explosion': 'Explosion, often resulting in fire, smoke, and scattered debris.',
#     'Fighting': 'A group of people fighting and brawling, which can be seen in punches, kicks',
#     'RoadAccidents': 'A collision between two or more vehicles, often resulting in injury or damage.',
#     'Robbery': 'Robbing others property through violent means such as beating or holding a gun',
#     'Shooting': 'The act of firing a firearm, often with gun flame and people lying down.',
#     'Shoplifting': 'Shoplifting, sneak things into bags, clothes or under skirts in stores.',
#     'Stealing': 'Stealing property from cars or stealing motorcycles and batteries.',
#     'Vandalism': 'Damaging vehicles, overturning shelves, or smashing store door.',
# }
# XD-Violence
ucf_cls_defs = {
    'Normal': 'Normal behavior, views or surveillance content captured by camera that not involve any unexpected or unusual events.',
    'Explosion': 'Explosion, often resulting in fire, smoke, and scattered debris.',
    'Fighting': 'A group of people fighting and brawling, which can be seen in punches, kicks',
    'RoadAccidents': 'A collision between two or more vehicles, often resulting in injury or damage.',
    'Shooting': 'War scene, often involving gunfire, explosions, cannon, tanks, and blood.',
    'Riot': 'The chaotic riot scene. There are many people and special police officers who are suppressing it.',
}

def extract_frames(video_path: str, output_dir: str, interval=8):
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vf', f'select="not(mod(n\,{interval}))"',
        '-vsync', 'vfr', '-q:v', '2',
        f'"{output_dir}/%05d.jpg"'
    ]
    try:
        print(' '.join(cmd))
        subprocess.run(' '.join(cmd), shell=True, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        print(e.stderr)
        return None
    output_paths = [str(p) for p in sorted(Path(output_dir).glob('*.jpg'))]
    print(f"Extracted {len(output_paths)} frames to {output_dir}")
    return output_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./tmp')
    parser.add_argument('-vp', '--video_path', type=str, required=True)
    args = parser.parse_args()
    return args


def build_models(cfg, ckpt_path):
    # build model
    model_cls_paths = cfg['model']['class_path'].split('.')
    model_cls = getattr(
        importlib.import_module('.'.join(model_cls_paths[:-1])),
        model_cls_paths[-1]
    )
    model_cfg_paths = cfg['model']['init_args']['model_config']['class_path'].split('.')
    model_cfg_cls = getattr(
        importlib.import_module('.'.join(model_cfg_paths[:-1])),
        model_cfg_paths[-1]
    )
    model_cfg = model_cfg_cls(**cfg['model']['init_args']['model_config']['init_args'])
    training_cfg_paths = cfg['model']['init_args']['training_config']['class_path'].split('.')
    training_cfg_cls = getattr(
        importlib.import_module('.'.join(training_cfg_paths[:-1])),
        training_cfg_paths[-1]
    )
    training_cfg = training_cfg_cls(**cfg['model']['init_args']['training_config']['init_args'])
    model = model_cls(model_cfg, training_cfg).to(device)

    # load checkpoints
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)['state_dict']
    load_states = model.load_state_dict(state_dict, strict=False)
    if len(load_states[1]) > 0:
        print(f"Load may failed")
        print(f"Unexpected Keys: {load_states[1]}")
    if not all([k for k in load_states[0] if k.startswith('clip_text_model.')]):
        print(f"Load may failed")
        print(f"Missing Keys: {load_states[0]}")

    model.eval()

    # load vision encoder
    clip_processor = CLIPProcessor.from_pretrained(cfg['model']['init_args']['model_config']['init_args']['clip_name'])
    clip_model = CLIPModel.from_pretrained(cfg['model']['init_args']['model_config']['init_args']['clip_name']).to(device)
    return model, clip_processor, clip_model


@torch.no_grad()
def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        cfg['model']['init_args']['model_config']['init_args']['verbalizer_type'] = None
    cache_dir = Path(args.cache_dir, datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S'))
    cache_dir.mkdir(parents=True)

    # build models
    model, clip_processor, clip_model = build_models(cfg, args.ckpt)

    # load video
    img_paths = extract_frames(args.video_path, str(cache_dir), FRAME_INTERVAL)

    # extract features
    v_feats = []
    batch_size = 128
    for i in tqdm(range(0, len(img_paths), batch_size), desc='Extracting features'):
        images = [Image.open(p) for p in img_paths[i:i + batch_size]]
        images_ts = clip_processor(images=images, return_tensors='pt', do_center_crop=True, do_resize=True, size=(224, 224)).pixel_values.to(device)
        v_feats.append(clip_model.get_image_features(images_ts).cpu())
    v_feats = torch.cat(v_feats, dim=0)  # [T,512]

    # only for LGTAdapter
    # if v_feats.shape[0] < 512:
    #     v_feats = torch.cat([
    #         v_feats,
    #         torch.zeros(512 - v_feats.shape[0], v_feats.shape[1], dtype=v_feats.dtype)
    #     ], dim=0)
    # elif v_feats.shape[0] > 512:
    #     v_feats = v_feats[:512]

    """
    Args:
        batch:
            v_feat [B,L,E], v_feat_l[B]
        class_names: List of str, encoded with soft prompts
        query_captions: List of str, encoded w/o soft_prompts
    Returns:
        vis_feats: [B,T,E] vision features after temporal encoding
        class_feats: [C,E] CLIP encoded class features
        cls_bin_logits: [B,T] binary logits (w/ class_names)
        cls_sim_mat: [B,T,C] similarity matrix (w/ class_names)
        query_caption_feats: [S,E] CLIP encoded query caption features
        cap_bin_logits: [B,T] binary logits (w/ query_captions)
        cap_sim_mat: [B,T,S] similarity matrix (w/ query_captions)
    """
    inputs_dict = {
        'batch': {
            'v_feat': v_feats[None, ...].to(device),
            'v_feat_l': torch.tensor([len(v_feats)], device=device),
        },
        # 'query_captions': list(ucf_cls_defs.values()),
        'class_names': list(ucf_cls_defs.keys()),
        # 'class_names': list(ucf_cls_defs.values()),
    }
    outputs_dict = model(**inputs_dict)

    # uncomment the following two lines to get the scores with language input
    # bin_score = outputs_dict['cap_bin_logits'][0].sigmoid().cpu().numpy()  # [T]
    # mul_score = outputs_dict['cap_sim_mat'][0].sigmoid().cpu().T.numpy()  # [S,T]

    # the following two lines to get the scores with class input
    bin_score = outputs_dict['cls_bin_logits'][0].sigmoid().cpu().numpy()  # [T]
    mul_score = outputs_dict['cls_sim_mat'][0].sigmoid().cpu().T.numpy()  # [S,T]


    fig = vis_result_v2(
        np.concatenate([bin_score[None, :], mul_score], axis=0),
        ['binary'] + list(ucf_cls_defs.keys()),
        args.video_path,
        num_img=12,
    )
    # add title
    font = font_manager.FontProperties(fname='simhei.ttf', size='large')
    fig.suptitle(f"{Path(args.video_path).stem}", fontproperties=font)
    plt.tight_layout()
    plt.savefig('end2end_inference_result.png')
    plt.show()


if __name__ == '__main__':
    main()



