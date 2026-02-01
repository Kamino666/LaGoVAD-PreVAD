import torch
import decord
from pathlib import Path
import src.models.clip as clip
from src.utils.video_loader import VideoLoader
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np

# Dir for your input videos
video_dirs = [
    'data/train_videos',
    'data/test_videos',
]
# Define the output directory for saving the extracted features, creating it if it doesn't exist
output_dir = Path('ViT-B-16-features')
output_dir.mkdir(exist_ok=True)

vid_paths = []
for vd in video_dirs:
    vid_paths += list(Path(vd).glob('*.mp4'))
print(len(vid_paths))

device = torch.device('cuda:1')
model, _ = clip.load('ViT-B/16', jit=True, device=device)

# Create a ThreadPoolExecutor for saving features in parallel, improving efficiency
save_executor = ThreadPoolExecutor(max_workers=2)
futures = []
try:
    with Progress() as progress:
        vid_task = progress.add_task('Extracting features', total=len(vid_paths))
        for vid_path in vid_paths:
            output_path = output_dir / f'{vid_path.stem}.npy'
            if output_path.exists():
                progress.update(vid_task, advance=1)
                continue

            loader = VideoLoader(str(vid_path), batch_size=256, interval=8, augmentation='no_center_crop')
            # loader = VideoLoader(str(vid_path), batch_size=256, interval=5)
            # loader = VideoLoader(str(vid_path), batch_size=256, interval=8, output_size=336, augmentation='no_center_crop')

            batch_task = progress.add_task('Processing batches', total=len(loader))
            feats = []
            try:
                for frame_batch in loader:
                    # 1,T,C,H,W
                    with torch.no_grad():
                        frame_batch = frame_batch.to(device).squeeze(0)
                        feat = model.encode_image(frame_batch)  # T,512
                        feats.append(feat.cpu().numpy())
                    progress.update(batch_task, advance=1)
            except decord.DECORDError as e:
                print(vid_path)
                continue
            if len(feats) == 0:
                print(vid_path)
            feats = np.concatenate(feats, axis=0)

            futures.append(
                save_executor.submit(np.save, str(output_path), feats)
            )
            progress.update(batch_task, visible=False)
            progress.update(vid_task, advance=1)
finally:
    save_executor.shutdown()
    wait(futures)
