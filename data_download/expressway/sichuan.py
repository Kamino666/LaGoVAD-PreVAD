import json
import re
import time
import random

import streamlink
import traceback
import json
from datetime import datetime
from typing import Optional
from joblib import Parallel, delayed
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
import selenium.common
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support import expected_conditions as EC


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


def click_and_parse(driver, cam_item):
    # Simulate click
    try:
        try:
            ele = driver.find_element(By.XPATH, '//*[@class="detail-box animate__animated animate__fadeInDown"]')
            driver.execute_script('arguments[0].style="display: none;"', ele)
        except selenium.common.NoSuchElementException:
            pass

        ele = driver.find_element(By.XPATH, '//*[@class="svg-pan-zoom_viewport"]')
        driver.execute_script('arguments[0].style="transform: matrix(0.022094, 0, 0, 0.022094, -186.757, 244.107);"', ele)
        ActionChains(driver).move_to_element(cam_item).click().perform()
    except selenium.common.MoveTargetOutOfBoundsException:
        # print('MoveTargetOutOfBoundsException')
        time.sleep(3)  # wait for 5 seconds and retry
        try:
            ActionChains(driver).move_to_element(cam_item).click().perform()
        except selenium.common.MoveTargetOutOfBoundsException:
            return 'page_load_err'

    # Continuously get logs
    wait_start_time = time.time()
    while True:
        if time.time() - wait_start_time > 10:  # Exceeded 10s, failed to get, indicates video is not available
            return 'stream_not_available'
        logs = driver.get_log('browser')
        for l in logs:
            if l['level'] != 'INFO':
                continue
            # print(l['message'])
            if "url====" in l['message']:
                # match = re.search(r'https://scpull06\.scjtonline\.cn/[\w/?&.=]*', l['message'])
                # if match:
                #     link = match.group()
                #     print(f"matched link: {link}")
                #     return link
                link = l['message'].split(' ')[-1].strip('"')
                print(f"matched link: {link}")
                return link
        time.sleep(0.1)


def get_link_sichuan(visited_camera_ids: list, batch_size=5):
    """
    Randomly select a camera from Sichuan highway and pull a short duration of video from the live stream.
    using selenium-4.9.0 with Chrome!!

    Args:
        visited_camera_ids (list): a list of camera ids that have been visited before.
        batch_size (int): the number of cameras to randomly select.

    Returns:
        A dictionary containing the camera id, camera name and the visited list.
    """
    result = {
        'visited_camera_ids': visited_camera_ids,
        'cameras': [],
        'msg': "",
    }

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")
    capabilities = DesiredCapabilities.CHROME
    capabilities['goog:loggingPrefs'] = {'browser': 'ALL'}

    driver = webdriver.Chrome(
        executable_path='./chromedriver',
        # options=chrome_options, desired_capabilities=capabilities,
    )

    try:
        driver.get('https://etc.scjtonline.cn/ScWeChatAvatarNew/#/homePage')
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, '//*[@id="自定义地图11"]/*/*')))
    except selenium.common.NoSuchElementException:  # Failed to load
        result['msg'] = 'page_load_err'
        driver.quit()
        return result
    except selenium.common.TimeoutException:  # Load timeout
        result['msg'] = 'page_load_err'
        driver.quit()
        return result

    # Parse with bs4 (don't know why selenium can't get the corresponding attributes)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    soup_items = soup.find('g', attrs={'id': '自定义地图11'}).find_all('image')
    cam_items = driver.find_elements(By.XPATH, '//*[@id="自定义地图11"]/*/*')
    # Get batchsize nodes from webpage results
    batch_cam_items = []
    for idx, item in enumerate(soup_items):
        if len(batch_cam_items) >= batch_size:
            break
        # print(item)
        cam_info = json.loads(item.get('storagedata'))
        cam_id = cam_info['videoResourceId']
        if cam_id not in result['visited_camera_ids']:
            result['cameras'].append({
                'camera_id': cam_id,
                'camera_name': cam_info['name'],
            })
            result['visited_camera_ids'].append(cam_id)
            batch_cam_items.append(cam_items[idx])
    if len(batch_cam_items) == 0:  # If there are no new nodes
        result['msg'] = 'no_more_camera'
        driver.quit()
        return result

    # Click these nodes to get links
    for i, cam_item in enumerate(batch_cam_items):
        click_res = click_and_parse(driver, cam_item)
        if click_res.startswith('https://'):  # Video is available
            result['cameras'][i]['camera_link'] = click_res
            result['cameras'][i]['msg'] = 'good'
        else:  # Video is not available
            result['cameras'][i]['camera_link'] = None
            result['cameras'][i]['msg'] = click_res

    driver.quit()
    return result


def main():
    link, _, name = get_link_sichuan(list())
    print(f"{name}: {link}")
    # link = get_link_henan('d2c2f1e8-969b-499a-b032-3a7444412948', 2)
    # print(link)
    # msg = pull_stream(
    #     'https://scpull02.scjtonline.cn/scgro2/143233FFA0506B063920B16379B975E6.m3u8?t=66ebcab5&k=f4981fc9eba589be48c13b0fa6a0c8ba',
    #     'test.ts', duration=30.0, proxy=None)
    # print(f"ERROR: {msg}")

def crawl_streams_v2():
    num_workers = 2
    Path('raw').mkdir(exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=num_workers)
    futures = []
    res_dict = {'visited_camera_ids': []}

    """
    Infinite loop, exit after all cameras have been traversed
    Use thread pool to pull streams, when there are idle threads, use selenium to get live stream links from webpage for processing
    """
    cnt = 0
    while True:
        # pool._work_queue.qsize(): Number of tasks waiting to be processed
        if pool._work_queue.qsize() == 0:
            batch_size = 5
            print(f'Getting: {batch_size} items')
            res_dict = get_link_sichuan(res_dict['visited_camera_ids'], batch_size=batch_size)
            if res_dict['msg'] == 'page_load_err':  # May cause infinite retry
                print('page_load_err, Retrying...')
                continue
            elif res_dict['msg'] == 'no_more_camera':  # End
                break
            else:
                cnt += batch_size
                print(f'Count: {cnt}')
                if cnt >= 20:
                    break
                for item in res_dict['cameras']:
                    camera_id = item['camera_id']
                    camera_name = item['camera_name']
                    link = item['camera_link']
                    if link is None:  # This camera is not available
                        print(f"Camera not ok ==> {item['msg']} {camera_id} {camera_name}")
                        continue
                    st_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    duration = random.randint(20, 120)
                    output_path = Path('raw', f'{camera_id}_{camera_name}_{st_date}.ts')
                    futures.append(
                        pool.submit(pull_stream, url=link, output_file=str(output_path), duration=float(duration))
                    )

    for f in futures:
        f.result()
    pool.shutdown()


if __name__ == '__main__':
    crawl_streams_v2()
