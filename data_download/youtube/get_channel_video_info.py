import scrapetube

videos = scrapetube.get_channel(
    channel_url="https://www.youtube.com/@witnessedofficial/search?query=gang",
    proxies={'https': 'http://127.0.0.1:7890', 'http': 'http://127.0.0.1:7890'},
    content_type='videos'
)

# The videos variable is a generator
for video in videos:
    print(video)
    print(video['lengthText']['simpleText'])  # Video duration in string format
    print(video['videoId'])  # Video ID
    print(video['title']['runs'][0]['text'])  # Video title
    print(video['publishedTimeText']['simpleText'])  # Publish time as string, e.g., "6 years ago"