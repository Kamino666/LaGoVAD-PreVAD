import streamlink
import traceback


def download_stream(url, output_path, download_time=30, proxy="http://127.0.0.1:7890/",
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
    session.set_option('hls-duration', download_time)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36 Edg/128.0.0.0',
        'Origin': 'https://www.youtube.com',
        'Referer': 'https://www.youtube.com/',
    }
    session.set_option('http-headers', headers)
    # print(url)
    try:
        streams = session.streams(url)
        if '480p' in streams:
            stream = streams['480p']
        else:
            stream = streams['best']
    except streamlink.PluginError as e:
        return str(e)

    fd = stream.open()
    fo = open(output_path, 'wb')
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


# output .ts file
download_stream('https://www.youtube.com/watch?v=5BjZEkHWOn0',
                'test.ts',
                download_time=20,
                proxy="http://127.0.0.1:7890/")
