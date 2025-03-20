from sklearn import base
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging  # AutoModelForCausalLM
from torchvision.transforms.functional import InterpolationMode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple
from decord import VideoReader, cpu  # (eva-decord on Mac)
import torchvision.transforms as T
from PIL import Image
import numpy as np
import hashlib
import yt_dlp
import torch
import modal
import json
import os
import time
import argparse
import csv

# performance notes:
#
# - container lifecycle:
#   - modelserver containers stay alive for 20 minutes (scaledown_window=1200)
#   - model is loaded once when container initializes
#   - subsequent calls reuse the loaded model without reloading
#
# - efficiency gains:
#   - original approach: ~30-60 seconds per clip (mostly model loading)
#   - modelserver approach: ~1-5 seconds per clip (no model loading)
#   - 10x+ speedup for processing multiple clips
#
# - resource usage:
#   - gpu memory stays allocated between calls
#   - container caches model files in filesystem
#   - all hooks are properly cleaned up to prevent memory leaks

# =================================================================================================
# ⬇️ Constants & image selection
# =================================================================================================

# Set HuggingFace logging level to ERROR to suppress warnings
transformers_logging.set_verbosity_error()

# Define the Modal app
app = modal.App("basketball-video-search")


# Modal volume & GPU selection
VOL = modal.Volume.from_name("basketball-analysis")
MODAL_VOLUME_PATH = "/vol/"
GPU_CHOICE = "L40S"

# Image selection
CUDA_VERSION = "12.3.2"  # need 12.1.0 (deprecated) for faiss-gpu
PYTHON_VERSION = "3.10"
FLAVOR = "devel"         # includes full CUDA toolkit
OPERATING_SYS = "ubuntu22.04"
CUDNN8 = False

TAG = f"{CUDA_VERSION}-{('cudnn8-' if CUDNN8 else '')}{FLAVOR}-{OPERATING_SYS}"

# Define the Modal images with all dependencies

PIP_PKGS = [
    "packaging",
    "transformers==4.40.1",
    "yt-dlp",
    "torch",
    "torchvision",
    "decord",
    "imageio",
    "av",
    "sentencepiece",
    "einops",
    "timm",
    "pillow",
    "numpy",
    "opencv-python",
    "tqdm",
    "faiss-cpu",
    "huggingface_hub",
    "ninja",
    "wheel",
    "scenedetect[opencv]",
]

SHARED_ENV_VARS = {
    "DEBIAN_FRONTEND": "noninteractive",
}

SHARED_CMDS = [
    "pip install -U scikit-learn",
    "apt-get -qq update",
    "apt-get -qq -y install ffmpeg",
]

GPU_CMDS = [
    "apt-get -qq install -y build-essential gcc g++",  # flash-attn deps
    "pip install flash-attn --no-build-isolation",     # needed to run CLIP model
]

CPU_CMDS = []

LOCAL_FILE_MAPPINGS = [
    ["cookies.txt", "/root/cookies.txt"],
]

LOCAL_PYTHON_SOURCES = [
    "_remote_module_non_scriptable",
]

CPU_IMAGE = (
    modal.Image
    .debian_slim(PYTHON_VERSION)
    .env(SHARED_ENV_VARS)
    .run_commands(*SHARED_CMDS, *CPU_CMDS)
    .pip_install(*PIP_PKGS, extra_options="-q")
    .add_local_file(*LOCAL_FILE_MAPPINGS[0])
    .add_local_python_source(*LOCAL_PYTHON_SOURCES)
)

GPU_IMAGE = (
    modal.Image
    .from_registry(f"nvidia/cuda:{TAG}", add_python=PYTHON_VERSION)
    .env(SHARED_ENV_VARS)
    .run_commands(*SHARED_CMDS)
    .pip_install(*PIP_PKGS, extra_options="-q")
    .run_commands(*GPU_CMDS)
    .add_local_file(*LOCAL_FILE_MAPPINGS[0])
    .add_local_python_source(*LOCAL_PYTHON_SOURCES)
)


# =================================================================================================
# ⬆️ Constants and image selection
#
# ⬇️ Preprocessing functions
# =================================================================================================

@app.function(image=CPU_IMAGE)
def print_results(prompts: list[str], outputs : list[str], processing_times, exec_time : float):
    print("========================================================================================================================")
    for i in range(len(prompts)):
        print(f"Prompt {i+1}: {prompts[i]}\n")
        print(f"Answer for prompt {i+1}:\n{outputs[i]}\n\n")
        print(f"Processing time for prompt {i+1}: {processing_times[i]} seconds\n")

    print(f" ⏳⏳ Total processing time for all queries was {exec_time} seconds using InternVideo2.5 ⏳⏳")
    print("========================================================================================================================")


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
        'format': 'best',
        'outtmpl': MODAL_VOLUME_PATH + str('%(id)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        VOL.reload()
        try:
            info = ydl.extract_info(url, download=True)
            filepath = MODAL_VOLUME_PATH + str(f"{info['id']}.{info['ext']}")
            VOL.commit()

            return filepath, {
                'title': info.get('title'),
                'duration_ms': info.get('duration', 0) * 1000,
                'source_url': url
            }
        except yt_dlp.DownloadError as e:
            print(f"Error downloading video: {str(e)}")
            print(f">>> Error exc_info: {e.exc_info}")
            print(f">>> Error message: {e.msg}")
            return None, {}


@app.function(image=CPU_IMAGE)
def build_transform(input_size):
    """
    image preprocessing pipeline:
    - convert to RGB
    - resize to square (default 448x448)
    - convert to a PyTorch tensor
    - normalize pixel values using ImageNet mean & stdev
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    print(f"📕📕📕 Building transform with mean {MEAN} and stdev {STD}")
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


@app.function(image=CPU_IMAGE)
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest aspect ratio for image preprocessing."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


@app.function(image=CPU_IMAGE, timeout=2400)
def preprocess_image(image, min_num_patches=1, max_num_patches=6, image_size=448, use_thumbnail=False):
    """
    - Divides images into multiple patches based on aspect ratio for optimal model input
    - Determines best patch configuration (rows x columns) while maintaining aspect ratio
    - Resizes input then crops into equal-sized patches of image_size x image_size
    - Optionally adds full-image thumbnail when multiple patches are generated
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num_patches, max_num_patches + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num_patches and i * j >= min_num_patches)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio.remote(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


@app.function(image=CPU_IMAGE)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=28):
    """Get frame indices for video processing."""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                             for idx in range(num_segments)])
    return frame_indices


@app.function(image=CPU_IMAGE)
def get_num_frames_by_duration(duration):
    """Determine number of frames based on video duration."""
    local_num_frames = 4
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments

    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)
    return num_frames


@app.function(image=CPU_IMAGE, volumes={MODAL_VOLUME_PATH: VOL}, timeout=10000)
def get_tensors_from_video(video_path,
                           bound=None,
                           input_size=448,
                           chunk_duration=15.0,
                           num_segments=32,
                           overlap=0.5,
                           sampling_rate=5,
                           max_num_patches=6,
                           get_frame_by_duration=False,):
    VOL.reload()

    human_cache_key = f"{video_path}_{input_size}_{max_num_patches}_{num_segments}"
    cache_key = hashlib.md5(human_cache_key.encode()).hexdigest()
    cache_dir = os.path.join(MODAL_VOLUME_PATH, "video_cache")
    os.makedirs(cache_dir, exist_ok=True)
    pixel_values_path = os.path.join(cache_dir, f"{cache_key}_pixels.pt")
    patches_path = os.path.join(cache_dir, f"{cache_key}_patches.json")

    # Check if cached version exists
    if os.path.exists(pixel_values_path) and os.path.exists(patches_path):
        print((
            f" 🌈🌈🌈 Using cached video data for {os.path.basename(video_path)}: 🌈🌈🌈\n"
            f" => Pixel values from {os.path.basename(pixel_values_path)}\n"
            f" => Patches from {os.path.basename(patches_path)}\n"
        ))
        pixel_values = torch.load(pixel_values_path)
        with open(patches_path, 'r') as f:
            num_patches_list = json.load(f)
        print(f"🟪⚪️ Pixel values loaded from cache: {pixel_values}")
        print(f"🟪⚪️ Number of patches: {num_patches_list}")
        return pixel_values, num_patches_list

    pixel_values_path = os.path.join(cache_dir, f"{cache_key}_pixels.pt")
    patches_path = os.path.join(cache_dir, f"{cache_key}_patches.json")

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform.local(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration.local(duration)
    frame_indices = get_index.local(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = preprocess_image.remote(img, image_size=input_size, use_thumbnail=True,)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)

    # Save to cache
    torch.save(pixel_values, pixel_values_path)
    with open(patches_path, 'w') as f:
        json.dump(num_patches_list, f)
    VOL.commit()

    print(f"🔷🔶 Pixel values computed from scratch: {pixel_values}")
    print(f"🔷🔶 Number of patches: {num_patches_list}")

    return pixel_values, num_patches_list


# =================================================================================================
# ⬆️ Preprocessing functions
#
# ⬇️ Model server and functionality
# =================================================================================================

@app.cls(
    image=GPU_IMAGE,
    gpu=GPU_CHOICE,
    scaledown_window=1200,  # 20 minutes before an idle container is taken down
    timeout=12000,          # 3 hours before a running container is taken down
    volumes={MODAL_VOLUME_PATH: VOL},
)
class InternVideoModelServer:
    def __init__(self):
        """
        Container initialization method - runs once when container starts.
        - save time by loading the model only ONCE per GPU instead of per function call => enormous savings
        - cache shards so that we only need to download each shard once and then load them into each
            subsequent container using the cached shard, instead of downloading each shard one time
            for each GPU
        """
        print("Initializing persistent InternVideoModelServer container...")
        VOL.reload()

        # Set device - detect CUDA availability for GPU acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create model cache directory in Modal volume
        model_cache_dir = os.path.join(MODAL_VOLUME_PATH, "model_cache", MODEL_PATH)
        os.makedirs(model_cache_dir, exist_ok=True)
        # Check if model cache directory already exists
        if os.path.exists(model_cache_dir):
            print(f"⚡️⚡️⚡️ Using existing InternVideo2.5 model cache at: {model_cache_dir} ⚡️⚡️⚡️")

        else:
            print(f"🛠️ Creating new InternVideo2.5 model cache at: {model_cache_dir} 🛠️")

        # Load tokenizer - same as before, but only happens once per container
        print("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        # Load model with appropriate precision and caching of shards
        print("📝 Loading InternVideo 2.5 model", end="")
        if self.device.type == "cuda":
            # For GPU, use half precision to reduce memory footprint
            self.model = AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                cache_dir=model_cache_dir  # Cache model shards
            ).half().cuda()

            print(" in GPU mode", end="")
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                print(" and converted to bfloat16", end="")
                self.model = self.model.to(torch.bfloat16)
            print("...")
        else:
            print(" in CPU mode...")
            self.model = AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                cache_dir=model_cache_dir  # Cache model shards
            )

        # Save the cache to volume
        VOL.commit()

        # Set model to evaluation mode to disable dropout and other training behaviors
        self.model.eval()
        print("☢️  InternVideo 2.5 model loaded and ready  ☢️")


# Instantiate InternVideoModelServer class to avoid redundant model loading
model_server = InternVideoModelServer()


@app.function(image=CPU_IMAGE)
def get_prompt_from_play(play: str):
    # prompt = (
    #     f"Find the 5 clearest examples of {play}s in this video. Format your output as follows:\n"
    #     "  - The first line should be in the following format: \"Results for [play] \",\n"
    #     "    where [play] is the NAME IN CAPITAL LETTERS of the play you are searching for.\n"
    #     "  - Then, for each clip you find, provide the following output:\n"
    #     "    - give the start and end timestamps between square brackets\n"
    #     "    - name the NBA player who made the play\n"
    #     "    - 3 bullet points on the defensive basketball scheme during the play,\n"
    #     "      focusing on the most influential players' positions and movements\n"
    #     "    - 3 bullet points on the offensive basketball scheme run during the play,"
    #     "      focusing on the most influential players' positions and movements.\n"
    # )

    prompt_refined = (
        f"Analyze this video segment carefully for the following basketball play: {play}."
        "Only describe what you actually see in these exact frames.\n\n"
        f"If there are NO clear examples of {play}s in this specific segment, respond with 'No {play}s found in this segment.'\n\n"
        f"For any {play}s you observe, describe:\n"
        "1. The player executing the move\n"
        "2. The defensive positioning\n"
        "3. The offensive execution\n\n"
        "DO NOT invent or hallucinate examples. Only describe what's clearly visible in this specific video segment."
    )

    return prompt_refined


@app.function(image=CPU_IMAGE)
def clean_model_output(raw_output):
    """Clean up model output by removing tracking tags and internal data"""
    # Handle empty output
    if not raw_output:
        return ""

    # Remove answer prefix
    if raw_output.startswith("Answer:"):
        raw_output = raw_output[7:].strip()

    # Remove tracking tags and everything after them
    if "<track_begin>" in raw_output:
        raw_output = raw_output.split("<track_begin>")[0].strip()

    # Remove any other tags with regex
    import re
    raw_output = re.sub(r'<[^>]+>', '', raw_output)

    return raw_output.strip()


@app.function(image=GPU_IMAGE, gpu=GPU_CHOICE, volumes={MODAL_VOLUME_PATH: VOL}, timeout=10000)
def internvideo_single_query_pipeline(params):
    # prompt: str,
    # video_path: str,
    # num_segments: int,
    # chunk_duration: float,
    # sampling_rate: int,
    # overlap : float,
    prompts, video_path, num_segments, chunk_duration, sampling_rate, overlap = params
    outputs, processing_times = [], []

    # clear GPU cache before processing (to avoid overflows when processing long videos)
    # torch.cuda.empty_cache()

    generation_config = dict(
        # if false: uses GREEDY DECODING:
        #   model always selects the highest probability token at each step => results are deterministic
        # if true: uses sampling: model samples from the distribution of token probabilities at each step,
        #   which can lead to more creative but inconsistent outputs
        do_sample=False,

        # sets max length (in "tokens" = words/word pieces) of the output text
        max_new_tokens=1024,

        # beam search explores multiple possible text sequences simultaneously to find optimal outputs
        #  - when set to 1: essentially greedy search, only considers the single most likely next token at each step
        #  - when set to >1: considers multiple possible next tokens at each step, leading to
        #    more high-quality but slower outputs
        num_beams=5,

        # - values closer to 0 make the output more deterministic and focused on high-probability options
        # - values closer to 1 or above increase randomness and creativity in the generated content
        # temperature=0.7,
    )

    with torch.no_grad():
        pixel_values, num_patches_list = get_tensors_from_video.remote(
            video_path,
            num_segments=num_segments,
            get_frame_by_duration=False,
            chunk_duration=chunk_duration,
            overlap=overlap,
            sampling_rate=sampling_rate,
        )

        pixel_values = pixel_values.to(torch.bfloat16).to(model_server.model.device)

        start_time = time.time()
        chat_history = None

        for prompt in prompts:
            enhanced_prompt = (
                "You are a professional basketball analyst. Analyze these video frames carefully and thoroughly.\n\n"
                "IMPORTANT: You MUST provide detailed descriptions for what you see in these frames. "
                "Even if the content is unclear, describe what appears to be happening based on the visual evidence. "
                "Be specific about players, movements, and actions.\n\n"
                f"{prompt}"
            )

            # add frame prefix to prompt
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            query = video_prefix + enhanced_prompt
            query_start_time = time.time()
            print(f"⚠️⚠️ Querying model with prompt: {query} ⚠️⚠️")

            try:
                output, chat_history = model_server.model.chat(
                    model_server.tokenizer,
                    pixel_values,
                    query,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=chat_history,
                    return_history=True)

                processing_times.append(time.time() - query_start_time)
                print(f"⚠️⚠️ Output is: {output} ⚠️⚠️")
                outputs.append(output)
            except Exception as e:
                error_msg = f"Error during model inference: {str(e)}"
                print(f"ERROR: {error_msg}")
                outputs.append(error_msg)
                processing_times.append(time.time() - query_start_time)

        # calculate and print execution time
        total_time = time.time() - start_time

    return prompts, outputs, processing_times, total_time


@app.function(image=CPU_IMAGE)
def get_temporal_ordering_prompt():
    prompt = (
        "You are watching a series of basketball video frames presented in chronological order. "
        "I need you to analyze the sequence of events and player movements in great detail.\n\n"

        "IMPORTANT: Your task is to narrate what happens in these frames as if you're a professional basketball analyst. "
        "Focus on:\n"
        "- The specific movements and actions of individual players\n"
        "- The ball movement and possession changes\n"
        "- Any scoring plays or defensive stops\n"
        "- The tactical decisions and plays being executed\n"
        "- The temporal progression of the action from start to finish\n\n"

        "Be extremely detailed in your description. Even if some frames are unclear, describe what appears to be happening "
        "based on the context and visual evidence available. DO NOT be vague or general.\n\n"

        "If you can identify specific players, teams, or plays, mention them. If not, describe players by their jersey colors, "
        "numbers, or positions on the court.\n\n"

        "MANDATORY: Your response must be detailed and substantive, analyzing the full sequence of basketball action shown in the frames."
    )
    return prompt

# =================================================================================================
# ⬆️ Model server and functionality
#
# ⬇️ Entrypoint and main functions
# =================================================================================================

@app.function(
    image=CPU_IMAGE,
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=50000
)
def main(
    url: str = "https://www.youtube.com/watch?v=wgVOgGLtPtck",
    query: str = None,
    num_results: int = 5,
    embeddings_dir: str = f"{MODAL_VOLUME_PATH}embeddings/",
    output_dir: str = f"{MODAL_VOLUME_PATH}results/",
    gpu_count: int = 10,
    do_queries_test: bool = False,
    do_summarize_test: bool = False,
    do_temporal_order_test: bool = False,
    num_segments: int = 28,
    sampling_rate: int = 5,
    chunk_duration: float = 15.0,
    overlap_fraction : float = 0.5,
):
    """
    Args:
        See args in entrypoint_from_cli
    """
    # keep at least 1 GPU warm for model loading since we're gonna need at least 1 model instance
    model_server.keep_warm(1)

    video_path, metadata = download_video_with_cache.remote(url)
    # Convert duration from milliseconds to minute:second format
    duration_seconds = metadata.get('duration_ms', 0) / 1000
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    formatted_duration = f"[{minutes:02d}:{seconds:02d}]"

    print(f"📹📹📹 The video \"{metadata.get('title')}\" {formatted_duration} was loaded successfully. 📹📹📹")

    query_prompts = []

    if do_temporal_order_test:
        temporal_order_prompts = [
            get_temporal_ordering_prompt.local(),
            (
                "Based on what you observed in the video frames, who appears to be the best player on the court? "
                "Try to identify them (name, jersey number, or other features) and explain why you think they're the best performer."
            )
        ]
        query_prompts.append(temporal_order_prompts)

    if do_summarize_test:
        summary_prompt = [(
            "Analyze this basketball video and provide a comprehensive summary in point form. Include:\n"
            "1. Key game information (teams playing, general context if visible)\n"
            "2. Major scoring plays (dunks, three-pointers, layups)\n"
            "3. Standout defensive plays\n"
            "4. Notable player performances and statistics if shown\n"
            "5. Any tactical patterns or strategies employed by either team\n"
            "6. Highlight moments with timestamps if possible\n"
            "7. Game flow and momentum shifts\n"
            "Be specific about what you can clearly see in the footage.\n"
            "If certain details aren't visible or clear, acknowledge the limitations rather than making assumptions."
        )]
        query_prompts.append(summary_prompt)

    if do_queries_test:
        plays = ["slam dunk", "three-pointer shot", "crossover dribble", "layup", "flashy pass", "regular pass"]
        query_prompts.append([get_prompt_from_play.local(play)] for play in plays)

    if not query_prompts:
        print("No query options specified.")
        return

    # process queries in parallel
    results = internvideo_single_query_pipeline.map(
        [(prompt_set, video_path, num_segments, chunk_duration, sampling_rate, overlap_fraction)
         for prompt_set in query_prompts]
    )

    for result in results:
        print_results.local(*result)

    return


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
                raise IndexError(f"Index {v} out of bounds. This CSV has {len(rows)} videos to choose from.")

            row = rows[v]
            if len(row) < url_index + 1:
                raise IndexError(f"Row {v} doesn't have enough columns. Expected at least 3, got {len(row)}.")

            print(f"☑️ Loaded video {row[0]} of length {row[1]} from CSV file {csv_file} ☑️")
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
    url: str = "https://www.youtube.com/watch?v=wgVOgGLtPtc",
    # what we are looking for in the basketball video
    query: str = None,
    # number of results to generate per query
    num_results: int = 5,
    # where to store embeddings
    embeddings_dir: str = f"{MODAL_VOLUME_PATH}embeddings",
    # where to store results
    output_dir: str = f"{MODAL_VOLUME_PATH}results",
    # number of gpus available to use
    gpus: int = 10,
    # do the queries test
    q: bool = False,
    # do the summarize test
    s: bool = False,
    # do the list frames test
    l: bool = False,
    # chunk length in seconds
    c: float = 15.0,
    # overlap fraction
    o: float = 0.5,
    # number of segments
    n: int = 48,
    # sampling rate of frames
    r: int = 3,
    # if set to > 0, will load video v from test_videos.csv and use it as input
    v: int = 0,
    # csv file to load video from if the above option (--v) is set
    csv: str = "test_videos.csv"
):
    if v > 0:
        url = load_video_from_csv(v - 1, csv)
    # Call ghost.remote with parsed arguments
    main.remote(
        url=url,
        query=query,
        num_results=num_results,
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        gpu_count=gpus,
        do_queries_test=q,
        do_summarize_test=s,
        do_temporal_order_test=l,
        num_segments=n,
        sampling_rate=r,
        chunk_duration=c,
        overlap_fraction=o,
    )
