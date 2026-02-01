import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import decord
import numpy as np
from pathlib import Path
from einops import rearrange
from PIL import Image


def optimal_grid(num_images):
  """
  计算最佳的网格布局

  Args:
    num_images: 图片数量

  Returns:
    tuple: (行数, 列数)
  """

  min_rows = min(num_images, 6)
  best_diff = float('inf')
  best_rows, best_cols = 0, 0

  for rows in range(1, min_rows + 1):
    cols = num_images // rows
    if num_images % rows != 0:
      cols += 1  # 如果不能整除，列数加1

    diff = abs(rows - cols)
    if diff < best_diff:
      best_diff = diff
      best_rows, best_cols = rows, cols

  return best_rows, best_cols


def vis_result(scores, gt_scores=None, vid_path=None, fps=30, num_img=5,
               save_dir=None, save_filename='tmp', return_fig=False):
    # score: tensor[(N,) T]  gt_scores: tensor[n, 2]
    if len(scores.shape) == 1:
        scores = scores[None, :]
    # if gt_scores is not None:
    #     assert scores.shape[1] == gt_scores.shape[0]
    num_scores = len(scores)
    num_vid = 0 if vid_path is None else 1
    gs = GridSpec(nrows=num_scores + num_vid, ncols=num_img)

    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1500 * px, 1000 * px))

    # display video
    if vid_path is not None:
        vr = decord.VideoReader(str(vid_path))
        fps = vr.get_avg_fps()
        indices = np.linspace(0, len(vr) - 1, num_img).astype(int).tolist()
        imgs = vr.get_batch(indices).asnumpy()  # T,H,W,3
        for i in range(num_img):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(imgs[i])
            ax.axis('off')

    # display scores
    timestamps = np.arange(0, scores.shape[1] / fps, 1 / fps)[:scores.shape[1]]
    timestamps = [datetime.datetime.fromtimestamp(i) for i in timestamps.tolist()]
    for row_i in range(num_vid, num_scores + num_vid):
        ax = fig.add_subplot(gs[row_i, :])
        ax.plot(timestamps, scores[row_i-1])
        # ax.xaxis.set_major_locator(mdates.SecondLocator(int(scores.shape[1] / fps / 10)))
        # print(int(scores.shape[1] / fps / 10))
        # print(scores.shape[1])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        if gt_scores is not None:
            for j in range(gt_scores.shape[0]):
                start, end = (gt_scores[j] / fps).tolist()
                start = datetime.datetime.fromtimestamp(start)
                end = datetime.datetime.fromtimestamp(end)
                ax.axvspan(start, end, color='red', alpha=0.2)
        ax.grid('on')
        ax.set_ylim(-0.1, 1.1)
        # ax.set_xlim(0, None)
    # print(timestamps)
    # 显示图表
    # plt.show()
    if save_dir is not None:
        Path(save_dir).mkdir(exist_ok=True)
        save_path = Path(save_dir) / f"{save_filename}.png"
        fig.savefig(save_path)
        # save_path = Path(save_dir) / f"{save_filename}.svg"
        # fig.savefig(save_path)
    if return_fig is False:
        plt.close(fig)
        return None
    else:
        return fig


def vis_result_v2(scores, score_names, vid_path, fps=30, num_img=5, ylim=True):
    # scores: tensor[N, T]  score_names: str [N]
    num_scores = len(scores)
    row, col = optimal_grid(num_scores)
    row += 1  # for video

    gs = GridSpec(nrows=row, ncols=col)
    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1500 * px, 1000 * px))

    vr = decord.VideoReader(str(vid_path))
    fps = vr.get_avg_fps()
    indices = np.linspace(0, len(vr) - 1, num_img).astype(int).tolist()
    imgs = vr.get_batch(indices).asnumpy()  # T,H,W,3

    # show video
    scroll_img = rearrange(imgs, 't h w c -> h (t w) c')
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(scroll_img)
    ax.axis('off')

    # display scores
    timestamps = np.arange(0, scores.shape[1] / fps, 1 / fps)[:scores.shape[1]]
    timestamps = [datetime.datetime.fromtimestamp(i) for i in timestamps.tolist()]
    for i in range(num_scores):
        _row, _col = i // col, i % col
        _row += 1

        ax = fig.add_subplot(gs[_row, _col])
        ax.plot(timestamps, scores[i])
        ax.set_ylabel(score_names[i])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax.grid('on')
        if ylim:
            ax.set_ylim(-0.1, 1.1)

    return fig


if __name__ == '__main__':
    # fake_scores = np.arange(473).reshape(1, 473)
    fake_scores = np.random.randn(1, 473)
    fake_scores = 1 / (1 + np.exp(-fake_scores))
    fake_gt_scores = np.array([[[5, 25], [90, 200]]])
    vis_result(fake_scores, fake_gt_scores,
               # '/data/lzh/datasets/VIDAL-10M/samples/1.mp4',
               # save_dir='.'
               )

