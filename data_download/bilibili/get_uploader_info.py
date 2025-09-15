"""
$ pip3 install bilibili-api-python
Documentation: https://github.com/Nemo2011/bilibili-api
You must obtain sessdata, method see https://nemo2011.github.io/bilibili-api/#/get-credential
Note: Don't crawl for too long at once to avoid account anomalies
"""
import json
from bilibili_api import user, sync, Credential

credential = Credential(
    sessdata="Get your own from cookies",
    bili_jct="Get your own from cookies",
    buvid3="Get your own from cookies",
    dedeuserid="Get your own from cookies",
)

# Instantiate
u = user.User(uid=539418077, credential=credential)  # uid is the UP's uid


async def main():
    # print(await u.get_user_info())
    result = []
    for i in range(1, 54):  # Here i is the page number, starting from 1
        info = await u.get_videos(pn=i)
        for item in info['list']['vlist']:
            result.append({
                'bvid': item['bvid'],
                'title': item['title'],
                'description': item['description'],
                'length': item['length'],
            })

    return result

# Entry point
res = sync(main())

with open('vids.json', 'w+', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)