import scrapetube
from tqdm import tqdm

KEYWORDS = [
    'webcam', 'live+cam', '24%2F7+cam'
]

videos = scrapetube.scrapetube.get_videos(
    'https://www.youtube.com/results?search_query=24%2F7+cam&sp=EgJAAQ%253D%253D',
    "https://www.youtube.com/youtubei/v1/search",
    "contents",
    "videoRenderer",
        limit=300,
        sleep=1,
        proxies={'https': 'http://127.0.0.1:7890', 'http': 'http://127.0.0.1:7890'},
)


for video in tqdm(videos):
    # filter according to live badge
    if video['badges'][0]['metadataBadgeRenderer']['label'] != 'LIVE':
        continue
    title = video['title']['runs'][0]['text']
    if 'fire' in title.lower():
        continue

    video_id = video['videoId']

    with open('stream.txt', 'a', encoding='utf-8') as f:
        f.write(title + '\n')
        f.write(video_id + '\n')

