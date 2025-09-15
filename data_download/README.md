# LaGoVAD Data Preparation Toolkit

This toolkit includes separate files to assist you in collecting data in the same manner as PreVAD.

- 🌐 **Data Scraping Disclaimer**: This toolkit is for research purposes only. Users bear full responsibility for compliance with local laws and target platform policies.
- 🛠️ **Code Availability**: The code is provided "as-is" without any guarantees of functionality, maintenance, or support.
- 💾 **Data Usage Responsibility**: Data collected remains the user's sole responsibility. We disclaim all liability for data misuse or distribution.

## Project Structure
```bash
data_download/
├── bilibili/           # Bilibili Video Data Collection
├── expressway/         # Highway Camera Data Collection
├── youtube/            # YouTube Data Collection
├── annotator/          # A simple web app for annotating videos
│   ├── app.py          # APP entry point
│   ├── config.json     # Configuration file
│   └── gen_exe.sh      # Build executable
└── README.md
```

## Dependency Installation
```bash
pip install -r requirements.txt
# Additional requirements:
chromedriver (required for sichuan.py)
```

## Functions

### 1. Bilibili
- `get_uploader_info.py`: Collects video information from a specified Bilibili uploader (requires user credentials).
- `download_videos.py`: Downloads Bilibili videos based on video IDs provided in a JSON file.

### 2. Expressway
- `henan-01.py`: Retrieves the list of highway cameras in Henan province.
- `henan-02.py`: Downloads live streams from Henan highway cameras.
- `shanghai-01.py`: Retrieves the list of highway cameras in Shanghai.
- `shanghai-02.py`: Downloads live streams from Shanghai highway cameras.
- `sichuan.py`: Downloads live streams from Sichuan highway cameras (requires chromedriver).

### 3. YouTube

videos:
- `download_videos.py`: Downloads YouTube videos with optional time trimming.
- `get_channel_video_info.py`: Retrieves information about videos from a specified YouTube channel.

streaming:
- `search_streaming_info.py`: Searches for live YouTube streams based on keywords and saves the results.
- `get_channel_streaming_info.py`: Retrieves information about live streams from a specified YouTube channel.
- `download_streaming.py`: Downloads YouTube live streams to a local file.

### 4. Annotator

We developed a simple-to-use, lightweight annotation tool designed to boost efficiency and eliminate redundant features.

1.  Keyboard-Centric Design: Annotate efficiently using just your keyboard for maximum speed.
2.  One-Click Manual Backup.
3.  Customizable Labels.
4.  Variable Playback Speed.
5.  Built-in Tutorial (in Chinese).

Regrettably, I don't have the time to internationalize it at the moment.

The code is very easy to understand and modify.

**Usage:**

1. Configure `config.json`.

    Specify the input directory for MP4 files and the output directory for TXT annotations by modifying the `input_dir` and `output_dir` in `config.json`. The `labels` corresponds to the nine tags displayed in a row on the annotation interface, which can be quickly selected using the number keys 1–9. The `port` parameter indicates the running port of the tool. Additionally, the `label2chinese` parameter provides Chinese translations for each label on the interface.

2. Run `python app.py` to start the app.

3. [Optional] Build an executable using `gen_exe.sh`
