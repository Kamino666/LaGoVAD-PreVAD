# ------------------------------------------------------------------------------
# Adapted from https://github.com/activitynet/ActivityNet/
# Original licence: Copyright (c) Microsoft, under the MIT License.
# Warning: some functions are modified.
# ------------------------------------------------------------------------------
import argparse
import json
import os
import ssl
import subprocess
import time
import pickle
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ssl._create_default_https_context = ssl._create_unverified_context


def download_clip_range(video_identifier,
                        output_filename,
                        start_time=None,
                        end_time=None,
                        split_chapters=False,
                        tmp_dir='/tmp/kinetics/.tmp_dir',
                        num_attempts=3,
                        url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.
    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the beginning time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    BASE_CMD = ['yt-dlp', '--quiet', '--no-warnings', '--no-check-certificate']

    proxy_cmd = ['--proxy', '"http://127.0.0.1:7890/"']
    prefer_res = "480p"
    if prefer_res == "480p":
        res_cmd = ['-S', '"ext,res:480"']
    elif prefer_res == "720p":
        res_cmd = ['-S', '"ext,res:720"']
    else:
        res_cmd = ['-S', '"ext,res:480"']
    BASE_CMD += res_cmd
    BASE_CMD += proxy_cmd

    cookie_path = "cookie.txt"
    if cookie_path is not None:
        cookie_cmd = ['--cookies', 'cookie.txt']
        BASE_CMD += cookie_cmd

    # Defensive argument checking.
    assert isinstance(video_identifier, str), 'video_identifier must be string'
    assert isinstance(output_filename, str), 'output_filename must be string'
    output_dir = Path(output_filename).parent
    # assert len(video_identifier) == 11, 'video_identifier must have length 11'

    status = False
    # Construct command line for getting the direct video link.
    # tmp_filename = os.path.join(tmp_dir, '%s.%%(ext)s' % uuid.uuid4())

    if not os.path.exists(output_filename):
        if start_time is not None:
            time_range = "*{}-{}".format(
                time.strftime("%M:%S", time.gmtime(start_time)),
                time.strftime("%M:%S", time.gmtime(end_time))
            )
            command = BASE_CMD + [
                '-o', f'{output_filename}',
                '--download-sections', time_range,
                '--write-info-json',
                f'{url_base + video_identifier}'
            ]
            command = ' '.join(command)
            print(command)
        elif split_chapters is True:
            chapter_param = 'chapter:' + str(output_dir / '[%(id)s]%(section_title)s.%(ext)s')
            command = BASE_CMD + [
                '-o', f'{output_filename}',
                '--split-chapters', '-o', f'"{chapter_param}"',
                '--write-info-json',
                f'{url_base + video_identifier}'
            ]
        else:
            command = BASE_CMD + [
                '-o', f'{output_filename}',
                '--write-info-json',
                f'{url_base + video_identifier}'
            ]
            command = ' '.join(command)
            print(command)

        attempts = 0
        while True:
            try:
                subprocess.check_output(
                    command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                attempts += 1
                if attempts == num_attempts:
                    return status, err.output
            else:
                break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    # os.remove(tmp_filename)
    return status, 'Downloaded'


def download_clip_wrapper(row, dl_dir, trim_format, tmp_dir):
    """Wrapper for parallel processing purposes."""
    basename = f"{row['video_id']}.mp4"
    if row['start_time'] is not None:
        basename = f"{row['video_id']}_{row['start_time']}_{row['end_time']}.mp4"
    output_filename = os.path.join(dl_dir, basename)
    clip_id = os.path.basename(output_filename).split('.mp4')[0]
    if os.path.exists(output_filename):
        status = tuple([clip_id, True, 'Exists'])
        return status

    downloaded, log = download_clip_range(
        row['video_id'],
        output_filename,
        row['start_time'],
        row['end_time'],
        tmp_dir=tmp_dir)
    status = tuple([clip_id, downloaded, log])
    return status


def parse_csv(input_csv):
    df = pd.read_csv(input_csv)
    download_data = []
    for item in df.itertuples():
        video_id = item[0]
        st = item[1]
        en = item[2]
        download_data.append({
            'video_id': video_id,
            'start_time': st,  # 秒 如80
            'end_time': en  # 秒 如90
        })
    return download_data


def parse_txt(input_txt):
    download_data = []
    with open(input_txt) as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':
                break
            download_data.append({
                'video_id': line,
                'start_time': None,
                'end_time': None
            })
    return download_data


def main(input_csv: str,
         output_dir,
         trim_format='%06d',
         num_jobs=24,
         tmp_dir='/tmp/kinetics',
         **kwargs):
    tmp_dir = os.path.join(tmp_dir, '.tmp_dir')

    if input_csv.endswith('.csv'):
        dataset = parse_csv(input_csv)
    elif input_csv.endswith('.txt'):
        dataset = parse_txt(input_csv)
    else:
        raise NotImplementedError

    # Download all clips.
    if num_jobs == 1:
        status_list = []
        for row in dataset:
            status_list.append(
                download_clip_wrapper(row, output_dir, trim_format, tmp_dir))
    else:
        status_list = Parallel(n_jobs=num_jobs)(
            delayed(download_clip_wrapper)(
                row, output_dir, trim_format, tmp_dir
            )
            for row in dataset[:]
        )

    with open('download_report.pkl', 'wb') as fobj:
        pickle.dump(status_list, fobj)


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        'input_csv',
        type=str,
        help=('CSV file containing the following format: '
              'YouTube Identifier,Start time,End time'))
    p.add_argument(
        'output_dir',
        type=str,
        help='Output directory where videos will be saved.')
    p.add_argument(
        '-f',
        '--trim-format',
        type=str,
        default='%06d',
        help=('This will be the format for the '
              'filename of trimmed videos: '
              'videoid_%0xd(start_time)_%0xd(end_time).mp4'))
    p.add_argument('-n', '--num-jobs', type=int, default=5)
    p.add_argument('-t', '--tmp-dir', type=str, default='/tmp/kinetics')
    p.add_argument('--split-by-chapter', action='store_true')
    # help='CSV file of the previous version of Kinetics.')
    main(**vars(p.parse_args()))
