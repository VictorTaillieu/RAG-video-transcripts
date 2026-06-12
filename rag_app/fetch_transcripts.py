import argparse
import json
from pathlib import Path

import requests
from youtube_transcript_api import YouTubeTranscriptApi

from rag_app.config import settings


def get_video_metadata(video_id: str) -> dict[str, str]:
    """
    Get video metadata (title and creator) from YouTube oEmbed API.
    """
    url = "https://www.youtube.com/oembed"
    params = {"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"}

    response = requests.get(url, params=params)
    video_info = response.json()

    return {"title": video_info["title"], "creator": video_info["author_name"]}


def fetch_video_transcript(video_id: str) -> list[dict]:
    """
    Fetch the transcript of a YouTube video using the YouTube Transcript API.
    """
    transcipt = YouTubeTranscriptApi().fetch(video_id, languages=["fr"])

    return transcipt.to_raw_data()


def main(video_ids: list[str]) -> None:
    Path(settings.RAW_DOCUMENTS_PATH).mkdir(exist_ok=True)

    for video_id in video_ids:
        print(f"Fetching {video_id}...")

        metadata = get_video_metadata(video_id)
        transcript = fetch_video_transcript(video_id)

        document = {**metadata, "transcript": transcript}

        with open(
            f"{settings.RAW_DOCUMENTS_PATH}/{video_id}.json", "w", encoding="utf-8"
        ) as file:
            json.dump(document, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_ids", nargs="+", help="One or more YouTube video IDs")
    args = parser.parse_args()

    main(args.video_ids)
