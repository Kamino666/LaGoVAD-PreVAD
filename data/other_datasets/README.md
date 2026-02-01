# Annotation Format

Each JSON file contains a list of objects representing video samples.

## JSON Fields

*   **`path`**: Relative path to the feature file (`.npy`).
*   **`video_path`**: Relative path to the raw video file.
*   **`class_name`**: Category label (e.g., `"Abuse"`, `"Normal"`). Must match dataset-specific classes.
*   **`descriptions`** (Optional): Text description of the event. Can be `null`.
*   **`anomaly_span`**: List of temporal segments `[start, end]` (normalized 0-1) where anomalies occur. Empty `[]` for normal videos.

## Example

```json
{
    "path": "other_datasets/ucf_features/Abuse028_x264.npy",
    "video_path": "other_datasets/ucf_videos/Abuse028_x264.mp4",
    "class_name": "Abuse",
    "descriptions": null,
    "anomaly_span": [[0.116, 0.170]]
}
```
