import scrapetube

videos = scrapetube.get_channel(
    channel_url="https://www.youtube.com/@Hamptons/streams",
    proxies={'https': 'http://127.0.0.1:7890', 'http': 'http://127.0.0.1:7890'},
    content_type='streams'
)

for video in videos:
    print(video)
