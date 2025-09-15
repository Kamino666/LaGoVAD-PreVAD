import requests
import json


url = "https://eeca.jtw.sh.gov.cn/vcloud-api/vcloud/devCamera/queryDevCameraList"

headers = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36 Edg/128.0.0.0',
    'Origin': 'https://epsn.jtw.sh.gov.cn',
    'Referer': 'https://epsn.jtw.sh.gov.cn/'
}

response = requests.get(url, headers=headers)
rsp_data = response.json()
cameras = rsp_data['result']
with open('shanghai_camera_list.json', 'w', encoding='utf-8') as f:
    json.dump(cameras, f)

# https://eeca.jtw.sh.gov.cn/vcloud-api/vcloud/devCamera/queryHlsUrlByCameraCode?cameraCode=c0bcec9e-da6a-4c26-958f-dfd9ab53a02e
# print(response.text)
