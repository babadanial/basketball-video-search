from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import CLIPProcessor, CLIPModel
from cohere.errors import TooManyRequestsError
from typing import Dict, Tuple, List
from itertools import count, repeat
from contextlib import nullcontext
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import subprocess
import pickle
import yt_dlp
import cohere
import base64
import random
import modal
import httpx
import torch
import pytz
import time
import json
import csv
import sys
import cv2
import os
import re

"""
TODO: https://github.com/QwenLM/Qwen2.5-VL
TODO: in functions where GPU processes batches of images (only run_queries_with_clip at the moment),
      implement multi-query processing when batch_size > len(images)
"""

# =================================================================================================
# ⬇️ Constants & image selection
# =================================================================================================

# Default arg values
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=4M1e83JSjB4"
DEFAULT_NUM_RESULTS = 10
DEFAULT_NUM_GPUS = 10
DEFAULT_FRAME_INTERVAL = 0.5
DEFAULT_TEXT_EMBEDDING_BATCH_SIZE = 96

# paths in this file are specified relative to the project root (rather than src/)
#   since this file is executed from the project root using the wrapper/launcher script
VIDEOS_CSV = "utils/test_videos.csv"
YT_COOKIES_FILE = "utils/cookies.txt"

# Define the Modal app
app = modal.App("basketball-video-search")

# Modal volume & GPU selection
MODAL_VOLUME_NAME = "basketball-video-search"  # previously "basketball-analysis"
MODAL_VOLUME_PATH = "/vol"
VOL = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

VOLUME_DOWNLOAD_DIR = MODAL_VOLUME_PATH + "/downloads"
VOLUME_FRAME_DIR = MODAL_VOLUME_PATH + "/frames"
VOLUME_EMBEDDINGS_DIR = MODAL_VOLUME_PATH + "/embeddings"
VOLUME_RESULTS_DIR = MODAL_VOLUME_PATH + "/results"
VOLUME_RESULTS_CACHE_DIR = MODAL_VOLUME_PATH + "/results-cache"

DEFAULT_QUERIES = [
    "slam dunk at the rim",
    "basketball jump shot",
    "basketball layup at the rim",
    "three-point shot",
    "basketball inbound",
    "basketball rebound",
    "basketball free throw",
    "basketball crossover dribble",
    "basketball fast break",
    "basketball post-up",
    "isolation basketball offense",
    "basketball double team",
    "basketball zone defense",
    "full-court press",
    "pick and roll",
    "LeBron James",
    "Nikola Jokic",
    "celebration",
]
RESULT_SEPARATOR_STRING = (
    "▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂\n"
    "▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂\n"
)
TITLE_SEPARATOR_STRING = "==============================================================================\n"

# Image selection
CUDA_VERSION = "12.8.1"
PYTHON_VERSION = "3.10"
FLAVOR = "devel"  # includes full CUDA toolkit
OPERATING_SYS = "ubuntu22.04"
CUDNN8 = True

# max number of tokens allowed in a vision model response
#   when describing a video frame
MAX_DESCRIPTION_TOKENS = 1024

COHERE_VISION_MODEL = "c4ai-aya-vision-8b"
COHERE_EMBED_MODEL = "embed-english-v3.0"
COHERE_RERANK_MODEL = "rerank-v3.5"
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_VISION_MODEL = "gpt-4o"  # gpt-4.5-preview, gpt-4o
LLAMA_VISION_MODEL = "llama-3.2-90b-vision-preview"  # llama-3.2-11b-vision-preview, llama-3.2-3b-preview
CLIP_MODEL_TAG = "openai/clip-vit-base-patch32"

TAG = f"{CUDA_VERSION}-{('cudnn-' if CUDNN8 else '')}{FLAVOR}-{OPERATING_SYS}"
NVIDIA_IMAGE = f"nvidia/cuda:{TAG}"

SHARED_PIP_PKGS = [
    "opencv-python-headless",
    "numpy<2",
    "modal",
    "cohere",
    # "yt_dlp",
    "python-dotenv",
    "psycopg2-binary",
    "decord",
    "torch>=2.2",
    "httpx",
    "openai",
    "pillow",
    "transformers",
    "tqdm",
    "pytz",
]

SHARED_GPU_PIP_PKGS = [
    "ninja",
    "packaging",
    "wheel",
    "accelerate>=0.26.0",  # needed to use device_map in CLIP
]

SHARED_ENV_VARS = {
    "DEBIAN_FRONTEND": "noninteractive",
}

SHARED_CMDS = [
    "pip install -q --upgrade pip",
    "pip install -U cohere",
    "python3 -m pip install -U --pre \"yt-dlp[default]\"",
    "apt-get -qq update",
    "apt-get -qq -y install curl ffmpeg libglib2.0-0 libgl1-mesa-glx",
]

FLASH_ATTN_CMDS = [
    "apt-get -qq -y install build-essential gcc g++",  # flash-attn deps
    "MAX_JOBS=4 pip install flash-attn --no-build-isolation",  # needed to run CLIP model
]

LOCAL_FILE_MAPPINGS = [
    [YT_COOKIES_FILE, "/root/cookies.txt"],
]

LOCAL_PYTHON_MODULES = [
    "_remote_module_non_scriptable",
]

CPU_IMAGE = (
    modal.Image.debian_slim(PYTHON_VERSION)
    .env(SHARED_ENV_VARS)
    .pip_install(*SHARED_PIP_PKGS, extra_options="-q")
    .run_commands(*SHARED_CMDS)
    .add_local_file(*LOCAL_FILE_MAPPINGS[0], copy=True)
    .add_local_python_source(*LOCAL_PYTHON_MODULES)
)

# includes flash-attn to make CLIP model faster
TORCH_FLASH_ATTN_BASE_IMAGE = (
    modal.Image
    .from_registry(
        # "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel",
        NVIDIA_IMAGE,
        # secret=REGISTRY_SECRET,
        add_python=PYTHON_VERSION
    )
    .pip_install(*SHARED_GPU_PIP_PKGS, extra_options="-q")
    .run_commands(*FLASH_ATTN_CMDS)
)

REGULAR_GPU_BASE_IMAGE = (
    modal.Image
    .from_registry(NVIDIA_IMAGE, add_python=PYTHON_VERSION)
    .pip_install("transformers", extra_options="-q")
)

# Pass API keys along as secrets to containers that need them
# and select GPU image based on whether --o (optimization flag) is present
if modal.is_local():
    API_KEYS_SECRET = modal.Secret.from_dict({
        "COHERE_API_KEY": os.environ["COHERE_API_KEY"],
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        "GROQ_API_KEY": os.environ["GROQ_API_KEY"],
    })
    GPU_PARAMS_SECRET = modal.Secret.from_dict({
        "GPU_IMAGE_CHOICE": "flash-attn" if "--o" in sys.argv else "regular",
    })
    optimize_gpu_params : bool = ("--o" in sys.argv)
else:
    API_KEYS_SECRET = modal.Secret.from_dict({})
    GPU_PARAMS_SECRET = modal.Secret.from_dict({})
    optimize_gpu_params : bool = (os.environ.get("GPU_IMAGE_CHOICE", "") == "flash-attn")

if optimize_gpu_params:
    GPU_BASE_IMAGE = TORCH_FLASH_ATTN_BASE_IMAGE
    GPU_CHOICE = "H100"
else:
    GPU_BASE_IMAGE = REGULAR_GPU_BASE_IMAGE
    GPU_CHOICE = "L40S"

GPU_IMAGE = (
    GPU_BASE_IMAGE
    .env(SHARED_ENV_VARS)
    .pip_install(*SHARED_PIP_PKGS, *SHARED_GPU_PIP_PKGS, extra_options="-q")
    .run_commands(*SHARED_CMDS)
    .add_local_file(*LOCAL_FILE_MAPPINGS[0], copy=True)
    .add_local_python_source(*LOCAL_PYTHON_MODULES)
)

DESCRIPTION_PROMPT = (
    "You are an experienced, insightful and detailed basketball/sports/media analyst. "
    "Describe this basketball frame in detail, including:\n"

    "1. Main Action: Identify the primary basketball action occurring (shooting, dribbling, passing, "
    "defending, rebounding, screening, etc.). Specify shot types (jump shot, layup, dunk, floater, "
    "etc.) if applicable.\n"

    "2. Court Position: Note where on the court the action is taking place (paint, perimeter, corner, top "
    "of the key, baseline, etc.) and the game phase (transition, half-court offense, inbound play, etc.). "
    "Also clearly identify where in the frame the ball is located, or where you think it is located.\n"

    "3. Offensive Scheme: Identify the offensive strategy being employed (pick and roll, isolation,"
    " post-up, drive and kick, fast break, horns set, motion offense, triangle offense, etc.).\n"

    "4. Defensive Scheme: Note the defensive setup (man-to-man, zone (2-3, 3-2, 1-3-1), switching, "
    "help defense, double-team, full-court press, etc.).\n"

    "5. Game Context: Include visible game information such as score, time remaining, quarter/period, "
    "shot clock, and any critical game situations (close game in final minutes, etc.).\n"

    "6. Player Positioning: Describe the spacing and positioning of players on the court, including "
    "off-ball movement and floor spacing.\n"

    "7. Additional Details: Note any unique elements such as celebrations, coach interactions, "
    " referee signals, crowd reactions, or other contextual information visible in the frame.\n"

    "8. Player Identification: Identify visible players by jersey number and team color. "
    "If any recognizable star players are present, include their names. "
    "However, do not name any player you are not certain of.\n"
    "If you cannot identify any player or team clearly, state this clearly, then describe as best as"
    " you can based on what you see.\n"

    "Combine all elements into a single, detailed list of bullet points describing this moment in "
    " the basketball game.\n"

    "Return only your bullet points, containing all your observations, commentary, and analysis; "
    "Be sure to use basketball jargon where appropriate."

    "If, for any bullet points, you are unsure of your observation, "
    "simply state that you are unsure and do not provide any further details."

    "Your bullet points shouldn't be paragraphs; instead, nest sub-bullet points as necessary. "
    "If the scene is not footage of live basketball action, give a brief 5 bullet-point summary of "
    "what you see instead."
)


# =================================================================================================
# ⬆️ Constants and image selection
#
# ⬇️ Helper functions
# =================================================================================================

def get_formatted_datetime():
    # get the current time in UTC
    utc_now = datetime.now(pytz.utc)

    # convert to timezone
    local_tz = pytz.timezone('America/Toronto')
    local_now = utc_now.astimezone(local_tz)

    # format datetime string
    formatted_datetime = local_now.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
    return formatted_datetime


def sanitize_filename(filename: str) -> str:
    # Define a regex pattern for characters that are not safe for file names
    unsafe_chars = r'[<>:"/\\&|?*]'

    # Replace unsafe characters with the specified replacement character
    sanitized_filename = re.sub(unsafe_chars, '', filename)
    sanitized_filename_no_spaces = sanitized_filename.replace(' ', '_')
    return sanitized_filename_no_spaces


def shorten_yt_link(url: str) -> str:
    """
    Shorten a YouTube video link, if possible.
    """
    if "youtube.com" in url:
        video_id = url.split("v=")[1]
        url_stem = "https://youtu.be/"
        return url_stem + video_id
    else:
        return url


def commit_to_vol_with_exp_backoff():
    """
    Implement retry for volume commit with exponential backoff to be
    able to handle GRPC data loss errors and other network issues
    """
    max_retries = 10
    retry_count = 1
    base_delay = 2  # seconds

    while retry_count < max_retries:
        try:
            VOL.commit()
            if (retry_count > 1):
                print(f"Volume commit #{retry_count} successful.")
            break
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"💔 Failed to commit to Modal volume {MODAL_VOLUME_NAME} after {max_retries} attempts.")
                print(f"💔 Error: {type(e).__name__}: {str(e)}")
                raise e
            else:
                delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                severity = ""
                for _ in range(retry_count):
                    severity += "❗"
                print(
                    f"{severity} Volume commit error: {type(e).__name__}.\n"
                    f"Retry {retry_count}/{max_retries} after {delay:.2f}s"
                )
                time.sleep(delay)


@app.function(image=CPU_IMAGE, volumes={MODAL_VOLUME_PATH: VOL})
def encode_image_to_string_with_cache(image_path: str) -> str:
    """
    Convert an image to a base64-encoded string for use with Cohere.
    Caches encodings in Modal volume.
    """
    VOL.reload()
    cache_dir = f"{VOLUME_DOWNLOAD_DIR}/image_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{Path(image_path).stem}.txt"

    # return encoding if it already exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return f.read()

    # compute the encoding ...
    with open(image_path, "rb") as img_file:
        utf8_encoding = base64.b64encode(img_file.read()).decode('utf-8')
    encoding = f"data:image/jpeg;base64,{utf8_encoding}"

    # ... and cache it
    with open(cache_file, "w") as f:
        f.write(encoding)
    commit_to_vol_with_exp_backoff()

    return encoding


# =================================================================================================
# ⬆️ Helper functions
#
# ⬇️ Pipeline preprocessing functions
# =================================================================================================

@app.function(image=CPU_IMAGE, volumes={MODAL_VOLUME_PATH: VOL})
def download_video_with_cache(url: str) -> Tuple[str, Dict]:
    """
    Download a video from a URL using yt-dlp.
    Args:
        url: URL of the video to download
    Returns:
        Tuple containing the file path and video metadata
    """

    ydl_opts = {
        'cookiefile': "/root/cookies.txt",

        # using this format will obtain 1080p video if it is available and store it in an mp4 file
        'format': 'mp3/bestvideo',
        'verbose': True,
        # 'extractor_args': {
        #     'youtubetab': {
        #         'skip': 'authcheck',
        #     },
        # },
        'http_headers': {
            'User-Agent': (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/90.0.4430.93 Safari/537.36"
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        },
        'outtmpl': VOLUME_DOWNLOAD_DIR + "/" + str('%(id)s.%(ext)s'),
    }

    retry_count = 0
    max_retries = 5

    while retry_count < max_retries:
        VOL.reload()
        print(f"Attempt {retry_count + 1}/{max_retries} to download video from {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                filepath = VOLUME_DOWNLOAD_DIR + "/" + str(f"{info['id']}.{info['ext']}")

                video_height = info.get('height')
                video_width = info.get('width')
                resolution = f"{video_width}x{video_height}" if video_width and video_height else "unknown"
                print(f"🔍 Video resolution: {resolution} 🔍")

                commit_to_vol_with_exp_backoff()

                print(f"Downloaded {filepath}")
                return filepath, {
                    'title': info.get('title'),
                    'duration_ms': info.get('duration', 0) * 1000,
                    'source_url': url
                }
            except yt_dlp.DownloadError as e:
                if "country" in e.msg:
                    print(
                        "🌎⚠️ Encountered a 'video not available in your country' error. 🌎⚠️\n"
                        "🌎⚠️ Exiting with exit code 20; simply retry the script again.  🌎⚠️\n"
                        "🌎⚠️ If the pipeline is being run with the convenience script,  🌎⚠️\n"
                        "🌎⚠️ do nothing: it will rerun the pipeline automatically.      🌎⚠️\n"
                    )

                    # use os._exit() to exit with non-zero return value but without
                    #  any extra error output other than the explanation directly above
                    os._exit(20)

                print(f"Error downloading video: {str(e)}")
                print(f">>> Error exc_info: {e.exc_info}")
                print(f">>> Error message: {e.msg}")
                retry_count += 1
                print("Retrying in 1 second...\n")
                time.sleep(1)

    raise Exception(f"Failed to download video from {url} after {max_retries} attempts")


@app.function(image=CPU_IMAGE, volumes={MODAL_VOLUME_PATH: VOL}, timeout=10000)
def extract_single_frame(params):
    """
    Extract a single frame from a video at a specified timestamp.
    """
    VOL.reload()
    video_path, frames_dir, timestamp, width, height, i, num_frames = params

    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    # Calculate milliseconds for more precise timestamp in filename
    milliseconds = int((timestamp % 1) * 1000)

    frame_basename = f"frame-{minutes:02d}-{seconds:02d}-{milliseconds:03d}"
    frame_path = f"{frames_dir}/{frame_basename}.jpg"

    # Check if frame has already been extracted
    if os.path.exists(frame_path):
        print(
            f"⚡️1️⃣ Found cached frame for the timestamp {minutes:02d}:{seconds:02d}"
            f".{milliseconds:03d} at {frame_path} ⚡️1️⃣\n"
        )
        return frame_path

    # Create ffmpeg command to extract frame at this timestamp with full resolution
    extract_cmd = [
        'ffmpeg',
        '-ss', str(timestamp),
        '-i', video_path,
        '-vframes', '1',
        '-q:v', '1',  # High quality (1 is best, 31 is worst)
        '-vf', f'scale={width}:{height}',  # Force original resolution
        '-y',  # Overwrite if exists
        frame_path,
    ]

    retry_count = 1
    max_retries = 5
    # Run extraction command and commit it to the volume
    while (retry_count < max_retries):
        # print(
        #     f"🟡 Extracting frame at {minutes:02d}:{seconds:02d}.{milliseconds:03d} "
        #     f"| ({i+1}/{num_frames}) 🟡\n"
        # )
        result = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        commit_to_vol_with_exp_backoff()
        if os.path.exists(frame_path) and cv2.imread(frame_path) is not None:
            return frame_path
        elif retry_count < max_retries:
            sleep_time = 0.25
            print(
                f"🟠 Failed to extract frame at {timestamp}s after attempt ({retry_count}/{max_retries}) "
                f"- retrying in {sleep_time} seconds 🟠")
            retry_count += 1
            time.sleep(1)
        else:
            print(f"🔴 Frame was never extracted - {frame_path} did not exist! 🔴\n")
            print(f"⚠️ ffmpeg stderr: {result.stderr}")
            print(f"⚠️ ffmpeg stdout: {result.stdout}")


@app.function(image=CPU_IMAGE, volumes={MODAL_VOLUME_PATH: VOL}, timeout=10000)
def extract_frames(video_path: str, video_metadata: dict, frame_interval: float = 0.5):
    """
    Extract frames from a video at specified intervals using FFmpeg directly.
    This method is more reliable than Decord for problematic videos.
    """
    # Reload volume and check that video is available
    VOL.reload()
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_id = Path(video_path).stem
    frames_dir = f"{VOLUME_FRAME_DIR}/{video_id}"
    os.makedirs(frames_dir, exist_ok=True)
    commit_to_vol_with_exp_backoff()

    # Get video info using ffprobe (most reliable way)
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'json',
        video_path
    ]

    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    metadata = json.loads(result.stdout)

    # Parse video metadata
    # -> frame rate: often comes as fraction like "30000/1001"
    stream_info = metadata['streams'][0]
    r_frame_rate = stream_info.get('r_frame_rate', '30/1')
    num, denom = map(int, r_frame_rate.split('/'))
    fps = num / denom
    print(f"⚙️ Video FPS: {fps:.2f} ⚙️")
    # -> duration and total frames
    if stream_info.get('duration') is not None:
        duration = float(stream_info['duration'])
    else:
        # if duration isn't provided by ffmpeg, fall back to
        #   using duration from yt-dlp metadata
        duration = video_metadata.get('duration_ms') / 1000
    total_frames = int(duration * fps)
    print(f"⚙️ Video duration: {duration:.2f}s")
    print(f"⚙️ Total frames in video: {total_frames}")

    # -> video dimensions
    width = stream_info.get('width', "unknown")
    height = stream_info.get('height', "unknown")
    print(f"⚙️ Video resolution: {width} x {height}")

    # Calculate frame timestamps to extract (based on frame_interval)
    timestamps = [t for t in np.arange(0, duration, frame_interval)]
    num_frames_to_extract = len(timestamps)

    print(f"⚙️ Total frames in video: {total_frames}")
    print(f"⚙️ Number of frames to be extracted: {num_frames_to_extract} ⚙️")

    # Extract frames
    inputs = [(video_path, frames_dir, timestamp, width, height, index, num_frames_to_extract)
              for index, timestamp in enumerate(timestamps)]
    frame_paths = []
    with tqdm(total=len(inputs), desc="Extracting frames") as pbar:
        for result in extract_single_frame.map(inputs):
            frame_paths.append(result)
            pbar.update(1)

    print(f"✅✅✅ Extracted {len(frame_paths)} frames from {video_path} to {frames_dir} ✅✅✅\n")

    commit_to_vol_with_exp_backoff()

    return frame_paths


# =================================================================================================
# ⬆️ Pipeline preprocessing functions
#
# ⬇️ Obtaining embeddings of textual descriptions of frames
# =================================================================================================

@app.function(
    image=CPU_IMAGE,
    secrets=[API_KEYS_SECRET],
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000
)
def frame_to_cohere_aya_description(
    frame_path_str : str,
    frame_index: int,
    num_frames: int
) -> str:
    """
    Processes a single jpg frame to get its text description
    using the Cohere Aya vision model.

    Helper function used by video_to_text_embeddings.
    """
    frame_path = Path(frame_path_str)
    frame_name = frame_path.name
    description_file = os.path.join(
        frame_path.parent,
        f"{frame_path.stem}-cohere-tokens={MAX_DESCRIPTION_TOKENS}.txt")

    # Check if description already exists
    VOL.reload()
    if os.path.exists(description_file):
        # Note: should not usually reach this code, since if cached descriptions exist,
        #   cached embeddings of these descriptions should also exist
        print(
            f"⚡️*️⃣🍎 Cached Cohere Aya description found at {description_file}"
            f" - ({frame_index}/{num_frames}) ⚡️*️⃣\n")
        with open(description_file, "r") as f:
            return f.read()
    # else: print(f"🍎 Getting Cohere Aya description of {frame_name}\n")

    # Create client for just this frame (more efficient than sharing)
    client = cohere.ClientV2(os.environ["COHERE_API_KEY"])

    # Create messages for chat
    messages = [
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": DESCRIPTION_PROMPT
            }]
        }
    ]
    messages[0]["content"].append({
        "type": "image_url",
        "image_url": {"url": encode_image_to_string_with_cache.local(frame_path_str)}
    })

    # Implement exponential backoff
    max_retries = 15
    retry_count = 0
    base_delay = 3.5  # seconds

    while retry_count < max_retries:
        try:
            response = client.chat(
                model=COHERE_VISION_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=MAX_DESCRIPTION_TOKENS
            )
            description = response.message.content[0].text

            # Save description to file
            with open(description_file, "w") as f:
                f.write(description)

            commit_to_vol_with_exp_backoff()
            # print(f"✅🍎 Saved Cohere Aya description to {description_file}\n")
            return description

        except TooManyRequestsError:
            retry_count += 1
            if retry_count >= max_retries:
                raise  # Re-raise if we've exhausted our retries

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
            print(
                f"⏳ Rate limited on {frame_name} w/ Cohere Aya. "
                f"Retry {retry_count}/{max_retries} after {delay:.2f}s\n"
            )
            time.sleep(delay)


@app.function(
    image=CPU_IMAGE,
    secrets=[API_KEYS_SECRET],
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000
)
def frame_to_llama_description(
    frame_path_str : str,
    frame_index: int,
    num_frames: int
) -> str:
    frame_path = Path(frame_path_str)
    frame_name = frame_path.name
    description_file = os.path.join(
        frame_path.parent,
        f"{frame_path.stem}-llama-tokens={MAX_DESCRIPTION_TOKENS}.txt")

    # Check if description already exists
    VOL.reload()
    if os.path.exists(description_file):
        print(f"⚡️*️⃣🍊 Cached Llama description found at {description_file}"
              f" - ({frame_index}/{num_frames}) ⚡️*️⃣\n")
        with open(description_file, "r") as f:
            return f.read()
    # else: print(f"🍊 Getting Llama description of {frame_name} - ({frame_index}/{num_frames}) 🍊\n")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_to_string_with_cache.local(frame_path_str),
                        "detail": "high",
                    }
                }
            ],
        }
    ]

    max_retries = 15
    retry_count = 0
    base_delay = 3.5  # seconds
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=LLAMA_VISION_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=MAX_DESCRIPTION_TOKENS,
            )
            description = completion.choices[0].message.content

            # Save description to file
            with open(description_file, "w") as f:
                f.write(description)

            commit_to_vol_with_exp_backoff()
            # print(f"✅🍊 Saved Llama description to {description_file}\n")
            return description

        except (InternalServerError, APIConnectionError) as e:
            if isinstance(e, APIConnectionError):
                print("🌐⚠️  Encountered Groq API connection error. Retrying in 5 seconds\n")
            elif isinstance(e, InternalServerError):
                print("🌐⚠️  Encountered Groq internal server error. Retrying in 5 seconds\n")
            retry_count += 1
            time.sleep(5)

        except RateLimitError as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise e  # Re-raise if we've exhausted our retries

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
            print(
                f"⏳ Rate limited on {frame_name} w/ Llama on Groq. "
                f"Retry {retry_count}/{max_retries} after {delay:.2f}s\n"
            )
            time.sleep(delay)


@app.function(
    image=CPU_IMAGE,
    secrets=[API_KEYS_SECRET],
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000
)
def frame_to_gpt_description(
    frame_path_str : str,
    frame_index: int,
    num_frames: int
) -> str:
    frame_path = Path(frame_path_str)
    frame_name = frame_path.name
    description_file = os.path.join(
        frame_path.parent,
        f"{frame_path.stem}-{OPENAI_VISION_MODEL}-tokens={MAX_DESCRIPTION_TOKENS}.txt")

    # Check if description already exists
    VOL.reload()
    if os.path.exists(description_file):
        print(
            f"⚡️*️⃣🍋 Cached {OPENAI_VISION_MODEL} description found at {description_file} "
            f"- ({frame_index}/{num_frames}) ⚡️*️⃣\n")
        with open(description_file, "r") as f:
            return f.read()
    # else: print(f"🍋 Getting {OPENAI_VISION_MODEL} description of {frame_name} - ({frame_index}/{num_frames}) 🍋\n")

    client = OpenAI()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_to_string_with_cache.local(frame_path_str),
                        "detail": "high",
                    }
                }
            ],
        }
    ]

    max_retries = 15
    retry_count = 0
    base_delay = 3.5  # seconds
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=MAX_DESCRIPTION_TOKENS,
            )
            description = completion.choices[0].message.content

            # Save description to file
            with open(description_file, "w") as f:
                f.write(description)

            commit_to_vol_with_exp_backoff()
            # print(f"✅🍋 Saved GPT description to {description_file}\n")
            return description
        except (APIConnectionError, InternalServerError) as e:
            if isinstance(e, APIConnectionError):
                print("🌐⚠️ Encountered OpenAI API connection error. Retrying in 5 seconds\n")
            elif isinstance(e, InternalServerError):
                print("🌐⚠️ Encountered OpenAI internal server error. Retrying in 5 seconds\n")
            time.sleep(5)
        except RateLimitError as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise e  # Re-raise if we've exhausted our retries

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
            print(
                f"⏳ Rate limited on {frame_name} w/ {OPENAI_VISION_MODEL}. "
                f"Retry {retry_count}/{max_retries} after {delay:.2f}s\n"
            )
            time.sleep(delay)


@app.function(
    image=CPU_IMAGE,
    secrets=[API_KEYS_SECRET],
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000
)
def generate_cohere_embeddings_with_exp_backoff(params):
    """
    Generate embeddings using Cohere Embed model for either a batch of textual descriptions
    or a single image string.
    """
    candidate, index, length = params

    description_batch, image_str = None, None
    log_val = None
    log_index = f"({index+1}/{length})"

    if type(candidate) is list and type(candidate[0]) is str:
        description_batch = candidate
        log_val = description_batch[0][:60]
        args = {"texts": description_batch, }
        input_type = "search_document"
        # print(f"🔵🔵 Generating Cohere embeddings for text descriptions batch starting with {log_val} - {log_index}\n")
    elif type(candidate) is str:
        image_str = candidate
        log_val = image_str[:50]
        args = {"images": [image_str], }
        input_type = "image"
        # print(f"🟤🟤 Generating Cohere embedding for image {log_val} - {log_index}\n")
    else:
        raise ValueError(
            "Error in generate_cohere_embeddings: must provide either a list of strings "
            "containing frame descriptions or a single string with image data"
        )

    args = args | {
        "input_type": input_type,
        "model": COHERE_EMBED_MODEL,
    }
    # add these params separately since their values are lists to avoid errors
    args["embedding_types"] = ["float"]

    client = cohere.ClientV2(os.environ["COHERE_API_KEY"])
    max_retries = 15
    retry_count = 0
    base_delay = 4  # seconds
    while retry_count < max_retries:
        try:
            response = client.embed(**args)
            return np.array(response.embeddings.float)
        except (
            TooManyRequestsError, httpx.ConnectError, httpx.ReadTimeout,
            httpx.ConnectTimeout, httpx.ReadError, ConnectionError, OSError
        ) as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise  # Re-raise if we've exhausted our retries

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
            if isinstance(e, TooManyRequestsError):
                print(
                    f"‼️ Rate limited on {log_val} w/ Cohere Embed. "
                    f"Retry {retry_count}/{max_retries} after {delay:.2f}s - {log_index}‼️\n"
                )
            else:
                print(
                    f"🌐 Network error: {type(e).__name__}: {str(e)} on {log_val}. "
                    f"Retry {retry_count}/{max_retries} after {delay:.2f}s - {log_index}\n"
                )
            time.sleep(delay)


@app.function(
    image=CPU_IMAGE,
    secrets=[API_KEYS_SECRET],
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000
)
def generate_openai_text_embeddings_with_exp_backoff(params):
    """
    Generate embeddings using OpenAI text-embedding model for either a batch of textual descriptions
    or a single image string.
    """
    description_batch, index, length = params
    log_val = description_batch[0][:60]
    log_index = f"({index+1}/{length})"

    args = {
        "model": OPENAI_EMBED_MODEL,
        "input": description_batch,
        "encoding_format": "float"
    }

    client = OpenAI()
    max_retries = 15
    retry_count = 0
    base_delay = 4  # seconds
    while retry_count < max_retries:
        try:
            response = client.embeddings.create(**args)
            embeddings = [entry.embedding for entry in response.data]
            return np.array(embeddings)
        except (
            TooManyRequestsError, httpx.ConnectError, httpx.ReadTimeout,
            httpx.ConnectTimeout, httpx.ReadError, ConnectionError, OSError
        ) as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise  # Re-raise if we've exhausted our retries

            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
            if isinstance(e, TooManyRequestsError):
                print(
                    f"‼️ Rate limited on {log_val} w/ OpenAI text embedding. "
                    f"Retry {retry_count}/{max_retries} after {delay:.2f}s - {log_index}‼️\n"
                )
            else:
                print(
                    f"🌐 Network error: {type(e).__name__}: {str(e)} on {log_val}. "
                    f"Retry {retry_count}/{max_retries} after {delay:.2f}s - {log_index}\n"
                )
            time.sleep(delay)


@app.function(image=CPU_IMAGE, volumes={MODAL_VOLUME_PATH: VOL}, timeout=10000)
def video_to_text_embeddings(
    frame_paths : List[str],
    batch_size: int,
    frame_interval: int,
):
    """
    Compute embeddings for a video by:
    - first generating textual descriptions for them
      (e.g. using Cohere Aya vision model)
    - then embedding these descriptions using **Cohere Embed model**
    """
    result_file_and_source_tuples = []

    VOL.reload()

    # make sure we have at least one frame
    if not frame_paths or not os.path.exists(frame_paths[0]):
        raise FileNotFoundError(f"Frame file not found: {frame_paths[0] if frame_paths else 'No frames'}")
    num_frames = len(frame_paths)

    # Check if embeddings already exist; if they do, return them
    os.makedirs(VOLUME_EMBEDDINGS_DIR, exist_ok=True)
    video_id = Path(frame_paths[0]).parent.name

    descriptions_to_compute = []

    gpt_embeddings_filepath = (
        f"{VOLUME_EMBEDDINGS_DIR}/{video_id}-interval={frame_interval}"
        f"-{OPENAI_VISION_MODEL}-text-embeddings.txt"
    )
    descriptions_to_compute.append((
        gpt_embeddings_filepath,
        OPENAI_VISION_MODEL,
        frame_to_gpt_description))

    # cohere_aya_embeddings_filepath = (
    #     f"{VOLUME_EMBEDDINGS_DIR}/{video_id}-interval={frame_interval}"
    #     "-cohere-text-embeddings.txt"
    # )
    # descriptions_to_compute.append((
    #     cohere_aya_embeddings_filepath,
    #     "cohere-aya-embeddings",
    #     frame_to_cohere_aya_description))

    # llama_embeddings_filepath = (
    #     f"{VOLUME_EMBEDDINGS_DIR}/{video_id}-interval={frame_interval}"
    #     "-llama-text-embeddings.txt"
    # )
    # descriptions_to_compute.append((
    #     llama_embeddings_filepath,
    #     "llama-embeddings",
    #     frame_to_llama_description))

    embeddings_to_compute = []

    for (
        embeddings_filepath,
        source_name,
        frame_to_description_modal_func
    ) in descriptions_to_compute:
        if os.path.exists(embeddings_filepath):
            # if cached descriptions exist
            print(f"⚡️2️⃣ {source_name} embeddings already exist in {embeddings_filepath} ⚡️2️⃣\n")
            result_file_and_source_tuples.append((embeddings_filepath, source_name))
        else:
            # map frame paths to descriptions using the Modal function provided
            description_list = []
            with tqdm(total=num_frames, desc=f"Generating {source_name} descriptions") as pbar:
                for result in frame_to_description_modal_func.map(
                    frame_paths,
                    count(1, 1),
                    repeat(num_frames),
                    return_exceptions=True
                ):
                    description_list.append(result)
                    pbar.update(1)

            print(f"🟢🟢🟢 {source_name} - successfully computed all descriptions from frames 🟢🟢🟢")
            embeddings_to_compute.append((
                description_list,
                source_name,
                embeddings_filepath,))
            commit_to_vol_with_exp_backoff()

    print("🟦 Generating embeddings for text descriptions 🟦\n") if embeddings_to_compute else None

    for unvalidated_description_list, source_name, embeddings_filepath in embeddings_to_compute:
        # Filter out exceptions in description sets
        description_list = []
        for i, desc in enumerate(unvalidated_description_list):
            if isinstance(desc, Exception):
                print(f"⚠️ Error in frame {i}: {desc}")
            else:
                description_list.append(desc)

        # create batches of descriptions to be embedded
        description_batches = [description_list[i:i + batch_size]
                               for i in range(0, len(description_list), batch_size)]
        num_batches = len(description_batches)

        description_batches = [(batch, i, num_batches)
                               for i, batch in enumerate(description_batches)]
        embedding_set = []
        # process the description batches into batches of embeddings and vstack them into a single array
        with tqdm(total=num_batches, desc=f"Generating embeddings using Cohere Embed from descriptions generated by {source_name}") as pbar:
            for result in generate_cohere_embeddings_with_exp_backoff.map(description_batches):
                embedding_set.append(result)
                pbar.update(1)

        if not embedding_set:
            raise ValueError("ERROR: No embeddings generated for video text descriptions\n")
        embedding_set = np.vstack(embedding_set)

        # create vector db of embeddings and save it
        #   to a file to be used when processing queries
        vector_db = {i: (frame_paths[i], embedding) for i, embedding in enumerate(embedding_set)}
        with open(embeddings_filepath, "wb") as f:
            pickle.dump(vector_db, f)

        commit_to_vol_with_exp_backoff()

        print(
            f"📜📜 {source_name}: {embedding_set.shape[0]} embeddings of "
            f"frame descriptions saved in {embeddings_filepath}\n"
        )

        result_file_and_source_tuples.append((embeddings_filepath, source_name))

    return result_file_and_source_tuples


# =================================================================================================
# ⬆️ Obtaining embeddings of textual descriptions of frames
#
# ⬇️ Obtaining embeddings of frame images
# =================================================================================================

@app.function(image=CPU_IMAGE, volumes={MODAL_VOLUME_PATH: VOL}, timeout=10000)
def video_to_cohere_image_embeddings(
    frame_paths: List[str],
    frame_interval: int,
):
    """
    Compute embeddings for a video by **directly** generating embeddings from
    frame jpg images using Cohere Embed model (as opposed to first converting
    to text, as in video_to_text_embeddings())
    """
    VOL.reload()
    results = []
    video_id = Path(frame_paths[0]).parent.name
    cohere_image_embeddings_filepath = \
        f"{VOLUME_EMBEDDINGS_DIR}/{video_id}-interval={frame_interval}-cohere-image-embeddings.txt"
    if os.path.exists(cohere_image_embeddings_filepath):
        print(f"⚡️3️⃣ Found existing Cohere image embeddings from {cohere_image_embeddings_filepath} ⚡️3️⃣\n")
        return [(cohere_image_embeddings_filepath, "cohere-image-embeddings")]

    print(f"🔺🔺 Generating Cohere-compatible image strings for video {video_id} 🔺🔺\n")
    encoded_images = []
    with tqdm(total=len(frame_paths), desc="Encoding images into strings") as pbar:
        for result in encode_image_to_string_with_cache.map(frame_paths):
            encoded_images.append(result)
            pbar.update(1)

    print(f"🟥🟥 Generating Cohere image embeddings for video {video_id} 🟥🟥\n")
    num_frames = len(encoded_images)
    frame_strings = [(encoded_image, i, num_frames) for i, encoded_image in enumerate(encoded_images)]
    embedding_set = []
    with tqdm(total=num_frames, desc="Generating Cohere image embeddings") as pbar:
        for result in generate_cohere_embeddings_with_exp_backoff.map(frame_strings):
            embedding_set.append(result)
            pbar.update(1)

    if not embedding_set:
        raise ValueError("ERROR: No image embeddings generated for video frames\n")

    embedding_set = np.vstack(embedding_set)
    os.makedirs(VOLUME_EMBEDDINGS_DIR, exist_ok=True)
    vector_db = {i: (frame_paths[i], embedding) for i, embedding in enumerate(embedding_set)}
    with open(cohere_image_embeddings_filepath, "wb") as f:
        pickle.dump(vector_db, f)

    commit_to_vol_with_exp_backoff()

    print(
        f"🖼️🖼️ Saved {embedding_set.shape[0]} Cohere image embeddings "
        f"to {cohere_image_embeddings_filepath}\n"
    )
    results.append((cohere_image_embeddings_filepath, "cohere-image-embeddings"))
    return results


# =================================================================================================
# ⬆️ Obtaining embeddings of frame images
#
# ⬇️ Performing queries given embeddings
# =================================================================================================

@app.function(
    secrets=[GPU_PARAMS_SECRET],
    image=GPU_IMAGE,
    gpu=GPU_CHOICE,
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000)
# used to compute image embeddings when flash-attn is not being used
def frame_to_clip_image_embeddings(clip_processor, clip_model, images, device, flash_attn_available):
    # Produce PyTorch tensors from the image batch
    image_inputs = clip_processor(images=images, return_tensors="pt")
    # Move the tensor batch to the GPU
    pixel_values = image_inputs['pixel_values'].to('cuda')

    with torch.no_grad():
        with torch.autocast(device) if flash_attn_available else nullcontext():  # for maintainability
            # Generate embeddings for the image input tensor batch using the CLIP model
            image_embeddings = clip_model.get_image_features(pixel_values)
        normalized_embeddings = F.normalize(image_embeddings, p=2, dim=-1).cpu().numpy()

    return normalized_embeddings


@app.function(
    secrets=[GPU_PARAMS_SECRET],
    image=GPU_IMAGE,
    gpu=GPU_CHOICE,
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000)
# used to compute text embedding (e.g. of a query) when flash-attn is not being used
def compute_clip_text_embedding(clip_processor, clip_model, query, flash_attn_available, device):
    # convert query into a PyTorch tensor
    query_inputs = clip_processor(text=query, return_tensors="pt", padding=True).to('cuda')

    # move each tensor returned above to the GPU
    query_inputs = {k: v.cuda() for k, v in query_inputs.items()}

    # generate embeddings for text input tensors using the CLIP model
    with torch.no_grad():
        with torch.autocast(device) if flash_attn_available else nullcontext():  # for maintainability
            text_embedding = clip_model.get_text_features(**query_inputs)

    # normalize the embeddings using Euclidean norm (p=2) along the last dimension (dim=-1)
    # then move the tensor back to the CPU and convert it to a numpy array
    # and remove the first dimension (batch dimension) if it has size 1
    text_embedding = F.normalize(text_embedding, p=2, dim=-1).cpu().numpy().squeeze(0)
    return text_embedding


@app.function(
    secrets=[GPU_PARAMS_SECRET],
    image=GPU_IMAGE,
    gpu=GPU_CHOICE,
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000
)
def run_queries_with_clip(
    queries: list[str],
    frame_paths: List[str],
    frame_interval: float,
    num_results: int,
    video_url: str,
    video_title: str,
    video_id: str,
    video_results_cache_dir: str,
):
    VOL.reload()
    embeddings_source = "clip-image-embeddings"
    clip_image_embeddings_filepath = \
        f"{VOLUME_EMBEDDINGS_DIR}/{video_id}-interval={frame_interval}-clip-image-embeddings.txt"

    os.makedirs(video_results_cache_dir, exist_ok=True)

    # include processing parameters (frame sampling interval, result count, etc.)
    #   in name of cached file to avoid recomputing results
    filename_safe_queries = [sanitize_filename(query) for query in queries]
    query_output_file_names = [
        (
            f"{video_results_cache_dir}/"
            f"q=\"{filename_safe_query}\""
            f"-i=\"{frame_interval}\""
            f"-n=\"{num_results}\""
            f"-e=\"{embeddings_source}\""
            ".txt"
        )
        for filename_safe_query in filename_safe_queries
    ]

    # check for existing results; if all results have been computed already,
    #   return immediately without even loading the CLIP model
    unprocessed_queries = []
    for i, query in enumerate(queries):
        if os.path.exists(query_output_file_names[i]):
            print(f"⚡️5️⃣ Found existing CLIP results for query {query} in {query_output_file_names[i]} ⚡️5️⃣\n")
        else:
            unprocessed_queries.append(query)
    if not unprocessed_queries:
        return query_output_file_names
    queries = unprocessed_queries

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    similarity_metric_name = "cosine similarity"

    # function to load CLIP model and processor concurrently for efficiency
    def load_models_concurrently(load_functions_map: dict) -> dict:
        model_id_to_model = {}
        with ThreadPoolExecutor(max_workers=len(load_functions_map)) as executor:
            future_to_model_id = {
                executor.submit(load_fn): model_id
                for model_id, load_fn in load_functions_map.items()
            }
            for future in as_completed(future_to_model_id.keys()):
                model_id_to_model[future_to_model_id[future]] = future.result()
        return model_id_to_model

    print("5️⃣💥💥💥 Running queries against CLIP embeddings 💥💥💥5️⃣\n")
    print("5️⃣ Loading CLIP model and processor 5️⃣\n")

    clip_model_load_args = {
        "pretrained_model_name_or_path": CLIP_MODEL_TAG,
    }

    # check if flash-attn is installed: if so, use it for CLIP model
    flash_attn_available = False
    try:
        import flash_attn
        flash_attn_available = True
        print("5️⃣🟩🟩 flash-attn is installed, using it for CLIP model 🟩🟩5️⃣\n")
        clip_model_load_args["attn_implementation"] = "flash_attention_2"
        clip_model_load_args["device_map"] = "cuda"
        clip_model_load_args["torch_dtype"] = torch.float16
    except ImportError:
        print("5️⃣ flash-attn not found, continuing with standard attention 5️⃣\n")

    # load the model and processor & open images
    components = load_models_concurrently({
        "clip_model": lambda: CLIPModel.from_pretrained(**clip_model_load_args).to('cuda'),
        "clip_processor": lambda: CLIPProcessor.from_pretrained(CLIP_MODEL_TAG),
    })
    clip_model, clip_processor = components["clip_model"], components["clip_processor"]

    print("5️⃣ CLIP model and processor loaded 5️⃣\n")
    images = [Image.open(frame_path) for frame_path in frame_paths]

    # compute embeddings separately if we're not processing using flash-attn
    #   directly on GPU
    if not flash_attn_available:
        image_embeddings = []

        if os.path.exists(clip_image_embeddings_filepath):
            print(f"⚡️5️⃣ Found existing CLIP image embeddings from {clip_image_embeddings_filepath} ⚡️5️⃣\n")
            image_embeddings = np.load(clip_image_embeddings_filepath)

        else:
            print(f"5️⃣ Computing frame embeddings using CLIP model for {video_id} 5️⃣\n")
            frame_to_clip_image_embeddings.local(
                clip_processor,
                clip_model,
                images,
                'cuda',
                flash_attn_available)

            image_embeddings = np.vstack(image_embeddings)
            np.save(clip_image_embeddings_filepath, image_embeddings)
    else:
        # --- TODO: implement multi-query processing when batch_size > len(images) ---
        gpu_memory = torch.cuda.get_device_properties(0).total_memory  # in bytes
        image_size_in_bytes = np.prod(images[0].size) * 3  # 3 bytes per RGB pixel
        batch_size = int(gpu_memory / image_size_in_bytes)
        print(f"🟪🟪5️⃣ Batch size is {batch_size}")
        print(f"🟪🟪5️⃣   = ( {gpu_memory} bytes in GPU memory ÷ {image_size_in_bytes} bytes per image)")

    # handle each query individually, writing each to the filename designated for it in query_output_file_names
    for i, query in enumerate(tqdm(queries, desc="🔆5️⃣ Processing queries with CLIP embeddings 5️⃣🔆")):
        result_string = ""
        print(f"5️⃣ Processing query {queries[i]} ({i+1}/{len(queries)}\n")
        query_output_file = query_output_file_names[i]

        if not flash_attn_available:
            # if not using flash-attn, compute text embedding and similarities sequentially
            query_embedding = compute_clip_text_embedding.local(
                clip_processor=clip_processor,
                clip_model=clip_model,
                query=query,
                flash_attn_available=flash_attn_available,
                device='cuda',
            )
            similarities = [cosine_similarity(query_embedding, embedding)
                            for embedding in tqdm(image_embeddings, desc="🟨5️⃣ Calculating similarities naively 5️⃣🟨")]

        else:
            # if using flash-attn: process images in batches
            similarities = []

            for batch_start in range(0, len(images), batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch_images = images[batch_start:batch_end]

                # perform single forward pass for this batch
                with torch.no_grad():
                    # process the batch of images and the query, returning PyTorch tensors to
                    #   be used by CLIP

                    inputs = clip_processor(
                        text=[query],
                        images=batch_images,
                        return_tensors="pt",
                        padding=True
                    )

                    # move each tensor returned above to the GPU
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}

                    batch_string = f"[{batch_start}:{batch_end}]"

                    # Use autocast for mixed precision if available
                    with torch.autocast('cuda'):
                        print(f"  ♻️5️⃣ Batch computing embeddings w/ CLIP for images {batch_string}... 5️⃣♻️\n")
                        outputs = clip_model(**inputs)

                    print(f"  ♻️5️⃣ Computing similarities for images {batch_string}... 5️⃣♻️\n")

                    # Get similarity scores directly from the model
                    batch_similarities = outputs.logits_per_image.squeeze(1).cpu().numpy()
                    print(f"  🟩5️⃣ Similarities for images {batch_string} computed! 5️⃣🟩\n")
                    similarities.extend(batch_similarities)

        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:num_results].astype(int)

        result_string += (
            f"🔮 Results for: \"{query}\"\n"
            "📂 Embeddings source: CLIP embeddings\n"
            f"🕰 Computed on: {get_formatted_datetime()} 🕰\n"
        )
        result_string += TITLE_SEPARATOR_STRING

        for j, idx in enumerate(top_indices):
            # idx is index of the path in frame_paths
            # j is its rank in the top results

            path = frame_paths[idx]
            # Extract timestamp from path
            frame_name = Path(path).name

            # Extract timestamp components from filename (frame_MM_SS_MS.jpg)
            if frame_name.startswith("frame-"):
                parts = frame_name.split("-")
                minutes = parts[1]
                seconds = parts[2]
                milliseconds = parts[3].split(".")[0]
                timestamp = f"{int(minutes)}:{seconds}.{milliseconds}"
                timestamp_in_seconds = int(minutes) * 60 + int(seconds)
                url_to_frame = video_url + "&t=" + str(timestamp_in_seconds) + "s"
                result_string += f"{j+1}: {timestamp} ({url_to_frame})"
                result_string += f" - similarity: {similarities[idx]:.3f}"
            else:
                result_string += f"{j+1}: {path} - similarity: {similarities[idx]:.3f}"
            result_string += "\n"

        result_string += f"\n🎭 Similarity metric used: {similarity_metric_name}\n"
        result_string += (
            f"🎭 Similarity scores ranged from {min(similarities):.3f} "
            f"to {max(similarities):.3f}\n")
        result_string += RESULT_SEPARATOR_STRING

        with open(query_output_file, "w") as f:
            f.write(result_string)

    commit_to_vol_with_exp_backoff()

    return query_output_file_names


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.function(
    image=CPU_IMAGE,
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000,
    secrets=[API_KEYS_SECRET]
)
def run_queries_with_embedding_set(
    video_url: str,
    video_results_cache_dir: str,
    embeddings_file: str,
    embeddings_source: str,
    queries: list[str],
    num_results: int,
    frame_interval: int,
) -> str:  # returns path to result file

    VOL.reload()

    print(f"🔍 Attempting to use {embeddings_source} embeddings to run queries \n")
    # Load embeddings
    try:
        with open(embeddings_file, "rb") as f:
            embeddings_vector = pickle.load(f)
    except Exception as e:
        print(f"Error loading embeddings from {embeddings_file}: {str(e)}")
        return []
    embedding_set = [kvp[1][1] for kvp in embeddings_vector.items()]
    print("⚡️4️⃣ Embeddings loaded from file successfully ⚡️4️⃣\n")

    os.makedirs(video_results_cache_dir, exist_ok=True)
    filename_safe_queries = [sanitize_filename(query) for query in queries]
    query_output_file_names = [
        (
            f"{video_results_cache_dir}/"
            f"q=\"{filename_safe_query}\""
            f"-i=\"{frame_interval}\""
            f"-n=\"{num_results}\""
            f"-e=\"{embeddings_source}\""
            ".txt"
        )
        for filename_safe_query in filename_safe_queries
    ]

    # Initialize Cohere client
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))

    # Encode the query
    response = co.embed(
        texts=queries,
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    query_embeddings = response.embeddings.float

    # actual query processing occurs here
    # note results will be in order of the queries
    for i, query in enumerate(queries):
        query_output_file = query_output_file_names[i]

        if os.path.exists(query_output_file):
            print(f"⚡️4️⃣ Found existing {embeddings_source} results for {query} at {query_output_file} ⚡️4️⃣\n")
            continue

        print(f"➡️➡️➡️ Running query for {query} using {embeddings_source}\n")
        query_embedding = query_embeddings[i]
        similarity_scores = [cosine_similarity(query_embedding, embedding) for embedding in embedding_set]
        top_indices = np.argsort(similarity_scores)[::-1][:num_results]

        result_string = ""
        result_string += (
            f"🔮 Results for: \"{query}\"\n"
            f"📂 Embeddings source: {embeddings_source}\n"
            f"🕰 Computed on: {get_formatted_datetime()} 🕰\n"
        )
        result_string += TITLE_SEPARATOR_STRING
        for i, idx in enumerate(top_indices):
            path, _ = embeddings_vector[idx]
            # Extract timestamp from path
            frame_name = Path(path).name

            # Extract timestamp components from filename (frame_MM_SS_MS.jpg)
            if frame_name.startswith("frame-"):
                parts = frame_name.split("-")
                minutes = parts[1]
                seconds = parts[2]
                milliseconds = parts[3].split(".")[0]
                timestamp = f"{int(minutes)}:{seconds}.{milliseconds}"
                timestamp_in_seconds = int(minutes) * 60 + int(seconds)
                url_to_frame = video_url + "&t=" + str(timestamp_in_seconds) + "s"
                result_string += f"{i+1}: {timestamp} ({url_to_frame})"
                result_string += f" - similarity: {similarity_scores[idx]:.3f}"
            else:
                result_string += f"{i+1}: {path} - similarity: {similarity_scores[idx]:.3f}"
            result_string += "\n"
        result_string += "\n🎭 Similarity metric used: cosine similarity\n"
        result_string += (
            f"🎭 Similarity scores ranged from {min(similarity_scores):.3f} "
            f"to {max(similarity_scores):.3f}\n"
        )
        result_string += RESULT_SEPARATOR_STRING

        with open(query_output_file, "w") as f:
            f.write(result_string)

    commit_to_vol_with_exp_backoff()

    return query_output_file_names


# =================================================================================================
# ⬆️ Performing queries given embeddings
#
# ⬇️ Entrypoint and main functions
# =================================================================================================

@app.function(
    image=CPU_IMAGE,
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=50000
)
def main(
    url: str,
    queries: list[str],
    num_results: int,
    gpu_count: int,
    frame_interval: float,
    batch_size: int
):
    """
    Args:
        See args in entrypoint_from_cli
        query: Can be either a single string or a list of strings
    """
    video_path, video_metadata = download_video_with_cache.local(url)
    video_id = Path(video_path).stem
    video_title = video_metadata['title']
    print(f"Downloaded video: {video_title}")
    frame_paths = extract_frames.remote(
        video_path=video_path,
        frame_interval=frame_interval,
        video_metadata=video_metadata)

    embeddings_files_and_sources = []
    results = video_to_text_embeddings.remote(
        frame_paths=frame_paths,
        batch_size=batch_size,
        frame_interval=frame_interval,
    )
    embeddings_files_and_sources.extend(results)

    results = video_to_cohere_image_embeddings.remote(
        frame_paths=frame_paths,
        frame_interval=frame_interval,
    )
    embeddings_files_and_sources.extend(results)

    video_title_dir = sanitize_filename(video_title)
    video_results_cache_dir = f"{VOLUME_RESULTS_CACHE_DIR}/{video_title_dir}"
    video_results_dir = f"{VOLUME_RESULTS_DIR}/{video_title_dir}"
    os.makedirs(video_results_cache_dir, exist_ok=True)
    os.makedirs(video_results_dir, exist_ok=True)

    all_query_output_files = []

    # run queries with all embeddings collections with general-purpose function
    for embeddings_file, embedding_source in embeddings_files_and_sources:
        query_output_files = run_queries_with_embedding_set.remote(
            url,
            video_results_cache_dir,
            embeddings_file,
            embedding_source,
            queries,
            num_results,
            frame_interval,
        )
        all_query_output_files.extend(query_output_files)

    # run custom-written function for computing and then searching through CLIP embeddings
    query_output_files = run_queries_with_clip.remote(
        queries=queries,
        frame_paths=frame_paths,
        frame_interval=frame_interval,
        num_results=num_results,
        video_url=url,
        video_title=video_metadata['title'],
        video_results_cache_dir=video_results_cache_dir,
        video_id=video_id,
    )
    all_query_output_files.extend(query_output_files)

    VOL.reload()

    #  concatenate results for a query, from all embedding sources, together in one file
    for file in tqdm(all_query_output_files, desc="📜 Concatenating result files by query"):
        # removes the name of the embedding source from the file stem
        filestem = Path(file).stem
        new_filestem = filestem[:filestem.find("-e=")]
        result_file = f"{video_results_dir}/{new_filestem}.txt"

        with open(file, "r") as f:
            with open(result_file, "a") as f_out:
                f_out.write(f.read())

    commit_to_vol_with_exp_backoff()

    print(f"📩 All query results stored in {video_results_dir} on Modal volume {MODAL_VOLUME_NAME}")

    return video_results_dir


def load_video_from_csv(v: int, csv_file: str, url_index : int = 2):
    """
    Load a video URL from a CSV file.
    Args:
        v: Row index in the CSV file
        csv_file: Path to the CSV file (default: 'test_videos.csv')
        url_index: Column index for the video URL (default: 2)
    """
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

            if v < 0 or v >= len(rows):
                raise IndexError(f"Argument v={v} out of bounds; max is {len(rows)}.")

            row = rows[v]
            if len(row) < url_index + 1:
                raise IndexError(f"Row {v}: expected least {url_index + 1}, but found {len(row)}.")

            print(f"☑️ Loaded video from CSV file {csv_file} ☑️")
            print(f" >> Title (summarized): {row[0]}")
            print(f" >> Runtime: {row[1]}")
            print(f" >> URL: {row[url_index]}")
            return row[url_index]

    except FileNotFoundError:
        print(f"CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None


@app.local_entrypoint()
def entrypoint_from_cli(
    # basketball video footage to ingest
    u: str = DEFAULT_VIDEO_URL,

    # what we are looking for in the basketball video
    # (if unset, will default to list of default queries in DEFAULT_QUERIES)
    q: str = "",

    # number of results to generate per query
    n: int = DEFAULT_NUM_RESULTS,

    # number of gpus available to use
    g: int = DEFAULT_NUM_GPUS,

    # interval (in seconds) in between each frame that is sampled
    # note: this value shouldn't be under ~1/[video FPS rate]
    # e.g. for a 30 FPS video, this value should be >= 1/30 = 0.033
    i: float = DEFAULT_FRAME_INTERVAL,

    # batch size of descriptions to embed at once
    # max batch size for Cohere is 96
    # max batch size for OpenAI is 2048
    b: int = DEFAULT_TEXT_EMBEDDING_BATCH_SIZE,

    # if set to > 0, will load video v from test_videos.csv and use it as input
    v: int = 0,

    # csv file to load video from if the above option (--v) is set
    csv: str = VIDEOS_CSV,

    # if set, will copy the result file to results/{video_id} in the local filesystem
    c: bool = False,

    # if set, will use optimized image w/ flash-attn when running CLIP model
    o: bool = False,
):
    # load URL from CSV (if so specified), otherwise from --u arg
    if v > 0:
        url = load_video_from_csv(v - 1, csv)
    else:
        url = u
    # shorten URL if possible
    url = shorten_yt_link(url)

    # process query string into array if provided;
    #  otherwise use default query set
    queries = q.split(";") if q else DEFAULT_QUERIES

    if o:
        print("🌟 Using optimized GPU image with flash-attn for CLIP model 🌟")

    remote_results_dir = main.remote(
        url=url,
        queries=queries,
        num_results=n,
        gpu_count=g,
        frame_interval=i,
        batch_size=b
    )

    if c:
        remote_results_dir = remote_results_dir.removeprefix(MODAL_VOLUME_PATH)

        # Create local directory for results if it doesn't exist
        local_results_dir = "results/"
        os.makedirs(local_results_dir, exist_ok=True)

        # Download results from Modal volume
        cmd = f"modal volume get --force {MODAL_VOLUME_NAME} {remote_results_dir} {local_results_dir}"
        print(f"📥 Downloading results with command:\n📥 {cmd}")
        subprocess.run(cmd, shell=True)
