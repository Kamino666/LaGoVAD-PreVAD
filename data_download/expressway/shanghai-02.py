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


def get_link_shanghai(camera_code):
    url = f"https://eeca.jtw.sh.gov.cn/vcloud-api/vcloud/devCamera/queryHlsUrlByCameraCode?cameraCode={camera_code}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36 Edg/128.0.0.0',
        'Origin': 'https://epsn.jtw.sh.gov.cn',
        'Referer': 'https://epsn.jtw.sh.gov.cn/'
    }
    response = requests.get(url, headers=headers)
    rsp_data = response.json()
    hls_link = rsp_data['result']
    return hls_link


def main():
    link = get_link_shanghai('f754f099-2404-4eed-8a50-76d35286a1cd')
    print(link)
    msg = pull_stream(link, 'test.ts', duration=30.0, proxy=None)
    print(f"ERROR: {msg}")


def crawl_one_stream(camera_id, duration, output_file):
    stream_link = get_link_shanghai(camera_id)
    msg = pull_stream(stream_link, output_file, duration=duration, proxy=None)
    return msg


def crawl_streams():
    with open('shanghai_camera_list.json', 'r', encoding='utf-8') as f:
        camera_list = json.load(f)
    Path('raw').mkdir(exist_ok=True)
    tasks = []
    for cam in camera_list:
        camera_id = cam['cameraCode']
        camera_name = cam['cameraName']
        st_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        duration = random.randint(20, 120)
        tasks.append((camera_id, duration, Path('raw', f'{camera_id}_{camera_name}_{st_date}.ts')))

    results = Parallel(n_jobs=5, verbose=10)(
        delayed(crawl_one_stream)(camera_id, duration, output_file)
        for camera_id, duration, output_file in tasks
    )
    with open('download_log.json', 'w+') as f:
        json.dump(results, f)


if __name__ == '__main__':
    crawl_streams()
