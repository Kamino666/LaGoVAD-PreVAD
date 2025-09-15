import json
import random
import streamlink
import traceback
import requests
import pickle
from datetime import datetime
from typing import Optional
from joblib import Parallel, delayed
from pathlib import Path


def pull_stream(url, output_file, duration=30.0, proxy: Optional[str] = 'http://127.0.0.1:7890/',
                chunk_size: int = 8192):
    """
    Pull a live stream and save it to a file.

    Args:
        url (str): the URL of the stream
        output_file (str): the file path to save the stream to
        duration (float): the duration of the stream to pull (in seconds). Defaults to 30.0.
        proxy (str): the proxy address to use. Defaults to 'http://127.0.0.1:7890/'.
        chunk_size (int): the size of the chunks to read from the stream. Defaults to 8192 (1024 Byte).
    """
    session = streamlink.Streamlink()
    if proxy is not None:
        session.set_option('http-proxy', proxy)
    session.set_option('hls-duration', duration)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36 Edg/128.0.0.0',
        'Origin': 'https://weixin.hngscloud.com',
        'Referer': 'https://weixin.hngscloud.com/',
    }
    # http_query_params = {
    #
    # }
    session.set_option('http-headers', headers)
    # print(url)
    try:
        stream = session.streams(url)['best']
    except streamlink.PluginError as e:
        return str(e)

    fd = stream.open()
    fo = open(output_file, 'wb')
    err_msg = ""
    try:
        while True:
            data = fd.read(chunk_size)
            if data == b"":
                break
            fo.write(data)
    except streamlink.StreamError as e:
        # print(e)
        traceback.print_exc()
        err_msg = str(e)
    except OSError as e:
        traceback.print_exc()
        err_msg = str(e)
    finally:
        fd.close()
        fo.close()

    return err_msg


def get_link_henan(camera_code, video_type):
    # https://weixin.hngscloud.com/camera/playUrl?cameraNUm=ba676aaa-85c9-e304-ce78-7eb5192cd0d7&videoType=2&videoRate=0&t=1726667155885&_=1726667101862
    url = f"https://weixin.hngscloud.com/camera/playUrl"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36 Edg/128.0.0.0',
        'Origin': 'https://weixin.hngscloud.com',
        'Referer': 'https://weixin.hngscloud.com/'
    }

    params = {
        'cameraNUm': camera_code,
        'videoType': video_type,
        'videoRate': 0,
        't': int(datetime.now().timestamp() * 1000),
        '_': int(datetime.now().timestamp() * 1000)
    }
    # print(video_type)
    response = requests.get(url, headers=headers, params=params)
    rsp_data = response.json()
    hls_link = rsp_data['data']['playUrl']
    return hls_link


def main():
    link = get_link_henan('d2c2f1e8-969b-499a-b032-3a7444412948', 2)
    print(link)
    msg = pull_stream(link, 'test.ts', duration=30.0, proxy=None)
    print(f"ERROR: {msg}")


def crawl_one_stream(camera_id, video_type, duration, output_file):
    stream_link = get_link_henan(camera_id, video_type)
    msg = pull_stream(stream_link, output_file, duration=duration, proxy=None)
    return msg


def crawl_streams():
    with open('henan_camera_list.json', 'r', encoding='utf-8') as f:
        camera_list = json.load(f)
    Path('raw').mkdir(exist_ok=True)
    tasks = []
    for cam in camera_list:
        camera_id = cam['cameraNum']
        camera_name = cam['cameraName']
        camera_type = cam['cameraType']
        st_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        duration = random.randint(10, 30)
        tasks.append((camera_id, camera_type, duration, Path('raw', f'{camera_id}_{camera_name}_{st_date}.ts')))

    results = Parallel(n_jobs=5, verbose=10)(
        delayed(crawl_one_stream)(camera_id, 2, duration, output_file)
        for camera_id, _, duration, output_file in tasks[:10]
    )
    with open('download_log.json', 'w+') as f:
        json.dump(results, f)


if __name__ == '__main__':
    crawl_streams()
