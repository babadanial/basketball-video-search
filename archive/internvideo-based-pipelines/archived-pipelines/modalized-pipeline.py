from transformers import AutoModel, AutoTokenizer, logging as transformers_logging  # AutoModelForCausalLM
from torchvision.transforms.functional import InterpolationMode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple
from decord import VideoReader, cpu
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

# Set HuggingFace logging level to ERROR to suppress warnings
transformers_logging.set_verbosity_error()

# Define the Modal app
app = modal.App("basketball-video-search")

# Constants for image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_PATH = 'OpenGVLab/InternVideo2_5_Chat_8B'

# Modal volume & GPU selection
VOL = modal.Volume.from_name("basketball-analysis")
MODAL_VOLUME_PATH = "/vol/"
GPU_CHOICE = "L40S"

# Image selection
CUDA_VERSION = "12.3.2"  # need 12.1.0 (deprecated) for faiss-gpu
FLAVOR = "devel"         # includes full CUDA toolkit
OPERATING_SYS = "ubuntu22.04"
CUDNN8 = False

TAG = f"{CUDA_VERSION}-{('cudnn8-' if CUDNN8 else '')}{FLAVOR}-{OPERATING_SYS}"

# Define the Modal image with all dependencies
image = (
    modal.Image
    .from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
    .env({
        "DEBIAN_FRONTEND": "noninteractive"
    })
    .pip_install(
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
        "packaging",
        "wheel",
        "scenedetect[opencv]",
        extra_options="-q"
    )
    .run_commands(
        "pip install flash-attn --no-build-isolation",
        "pip install -U scikit-learn",

        # ffmpeg needed for image processing
        "apt-get -qq update",
        "apt-get -qq -y install ffmpeg",
    )
    .add_local_python_source("_remote_module_non_scriptable")
    .add_local_file("cookies.txt", "/root/cookies.txt")
)


@app.cls(
    image=image,
    gpu=GPU_CHOICE,
    scaledown_window=1200,  # 20 minutes before an idle container is taken down
    timeout=12000,          # 3 hours before a running container is taken down
    volumes={MODAL_VOLUME_PATH: VOL}
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

        # Set device - detect CUDA availability for GPU acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create model cache directory in Modal volume
        model_cache_dir = os.path.join(MODAL_VOLUME_PATH, "model_cache")
        os.makedirs(model_cache_dir, exist_ok=True)
        # Check if model cache directory already exists
        if os.path.exists(model_cache_dir):
            print(f"⚡️ Using existing InternVideo2.5 model cache at: {model_cache_dir} ⚡️")

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


@app.function(image=image, volumes={MODAL_VOLUME_PATH: VOL})
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


@app.function(image=image)
def build_transform(input_size):
    """
    image preprocessing pipeline:
    - convert to RGB
    - resize to square (default 448x448)
    - convert to a PyTorch tensor
    - normalize pixel values using ImageNet mean & stdev
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


@app.function(image=image)
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


@app.function(image=image, timeout=2400)
def dynamic_preprocess(image, min_num_patches=1, max_num_patches=6, image_size=448, use_thumbnail=False):
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


@app.function(image=image)
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


@app.function(image=image)
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


@app.function(image=image, volumes={MODAL_VOLUME_PATH: VOL})
def detect_scenes(video_path, threshold=30.0, min_scene_length=15):
    """
    Detect scene changes in a video using PySceneDetect.

    Args:
        video_path: Path to the video file
        threshold: Threshold for scene change detection (higher = fewer scenes)
        min_scene_length: Minimum length of a scene in frames

    Returns:
        List of scene boundaries as (start_frame, end_frame) tuples
    """
    from scenedetect import detect, ContentDetector

    # Detect scenes using content detection method
    scene_list = detect(video_path, ContentDetector(threshold=threshold, min_scene_len=min_scene_length))

    # Convert scene list to frame indices
    scene_boundaries = []
    for scene in scene_list:
        scene_boundaries.append((scene[0].frame_num, scene[1].frame_num))

    print(f"📸  Detected {len(scene_boundaries)} scenes in {video_path}  📸")
    return scene_boundaries


@app.function(
    image=image,
    timeout=10000
)
def calculate_motion_score(frame1, frame2):
    """
    Calculate motion score between two consecutive frames using OpenCV.

    Args:
        frame1: First frame as numpy array
        frame2: Second frame as numpy array

    Returns:
        Motion score (higher value indicates more motion)
    """
    import cv2
    import numpy as np

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude of flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Return mean magnitude as motion score
    return np.mean(magnitude)


@app.function(
    image=image,
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=10000)
def analyze_motion_in_video(video_path, sampling_rate=5):
    """
    Analyze motion throughout a video and identify high-motion segments.

    Args:
        video_path: Path to the video file
        sampling_rate: Sample every Nth frame for motion analysis

    Returns:
        List of frame indices with high motion scores
    """
    import cv2
    import numpy as np

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # fps =
    cap.get(cv2.CAP_PROP_FPS)

    # Calculate motion scores
    motion_scores = []
    prev_frame = None
    frame_indices = []

    for i in range(0, total_frames, sampling_rate):
        print(f"💎 Analyzing frame {i}/{total_frames} 💎")
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        if prev_frame is not None:
            # Calculate motion score between consecutive frames
            score = calculate_motion_score.remote(prev_frame, frame)
            motion_scores.append(score)
            frame_indices.append(i)

        prev_frame = frame

    cap.release()

    # No motion scores calculated
    if not motion_scores:
        return []

    # Find frames with high motion (above mean + 1 std)
    mean_score = np.mean(motion_scores)
    std_score = np.std(motion_scores)
    threshold = mean_score + std_score

    high_motion_indices = [idx for idx, score in zip(frame_indices, motion_scores)
                           if score > threshold]

    print(f"Found {len(high_motion_indices)} high-motion frames in {video_path}")
    return high_motion_indices


@app.function(image=image, volumes={MODAL_VOLUME_PATH: VOL}, timeout=10000)
def smart_load_video(video_path, bound=None, input_size=448, max_num_patches=1, num_segments=28, sampling_rate=5):
    """
    Intelligently load and preprocess video frames using scene detection and motion analysis.

    Args:
    - video_path: path to the video file
    - bound: time boundary as [start_time, end_time]
    - input_size:
        - control spatial resolution of processed video frames.
        - higher values increase visual detail but require more memory/processing
    - max_num_patches:
        - limits max number of patches each frame can be divided into
        - higher values allow dividing frames w/ irregular aspect ratio into multiple tiles,
            which preserves information (i.e. a rectangle-shaped frame is not squashed into a square)
    - num_segments:
        - number of evenly-spaced frames to sample from the video
        - higher values increase temporal detail, but require more memory
    - get_frame_by_duration:
        - if set to True, the number of frames is determined by the video duration.

    Returns:
    - pixel_values:
        - stores processed video frames as PyTorch tensors
    - num_patches_list:
        - list of number of patches that frame i has been divided into
    """
    # Calculate scene boundaries to find scene transitions in basketball footage
    scene_boundaries = detect_scenes.remote(video_path)

    # Determine time bound frames
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    start_frame = 0
    end_frame = max_frame

    if bound:
        start_time, end_time = bound
        start_frame = max(0, int(start_time * fps))
        end_frame = min(max_frame, int(end_time * fps))

    # Filter scenes to only those within our bound
    filtered_scenes = []
    for scene_start, scene_end in scene_boundaries:
        if scene_end >= start_frame and scene_start <= end_frame:
            # Adjust scene bounds to our time bound
            adjusted_start = max(scene_start, start_frame)
            adjusted_end = min(scene_end, end_frame)
            filtered_scenes.append((adjusted_start, adjusted_end))

    if not filtered_scenes:
        # Fall back to the entire bound if no scenes detected
        filtered_scenes = [(start_frame, end_frame)]

    # Find high-motion frames within our bound
    print(f"💎 Analyzing motion in video {video_path} 💎")
    high_motion_frames = analyze_motion_in_video.remote(video_path, sampling_rate)
    high_motion_frames = [f for f in high_motion_frames if start_frame <= f <= end_frame]
    # Combine scene detection with motion analysis for intelligent frame selection
    selected_frames = []

    # First, ensure we sample from scene boundaries
    for scene_start, scene_end in filtered_scenes:
        # Always take the first frame of each scene
        selected_frames.append(scene_start)

        # For longer scenes, sample frames from within the scene
        if scene_end - scene_start > 30:  # Arbitrary threshold for "long" scenes
            # Prefer high motion frames within this scene
            scene_motion_frames = [f for f in high_motion_frames if scene_start <= f <= scene_end]

            # Take up to 3 high-motion frames from each scene
            for frame in scene_motion_frames[:3]:
                if frame not in selected_frames:
                    selected_frames.append(frame)

    # If we still need more frames to reach num_segments
    if len(selected_frames) < num_segments:
        # Add remaining high motion frames
        for frame in high_motion_frames:
            if frame not in selected_frames:
                selected_frames.append(frame)
                if len(selected_frames) >= num_segments:
                    break

    # If we still don't have enough frames, add evenly spaced frames
    if len(selected_frames) < num_segments:
        # Calculate remaining frames needed
        remaining = num_segments - len(selected_frames)
        # Get evenly spaced frame indices
        step = (end_frame - start_frame) / (remaining + 1)
        for i in range(remaining):
            frame = int(start_frame + step * (i + 1))
            if frame not in selected_frames:
                selected_frames.append(frame)

    # sort frames chronologically and limit to num_segments
    selected_frames = sorted(selected_frames)[:num_segments]

    # define preprocessing pipeline we will use on each frame
    transform = build_transform.local(input_size=input_size)

    pixel_values_list, num_patches_list = [], []

    for frame_index in selected_frames:
        # load frame and convert to RGB numpy array
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")

        # dividing frames into multiple square patches based on aspect ratio to optimize model input
        img = dynamic_preprocess.local(img, image_size=input_size, use_thumbnail=True, max_num_patches=max_num_patches)

        # apply transformation to each patch (turning it into a tensor of normalized pixel values) ...
        pixel_values = [transform(tile) for tile in img]

        # ... and stack them into a single tensor
        pixel_values = torch.stack(pixel_values)

        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    # create a single tensor, which is a combination of all the individual frames that have been processed into tensors
    # -> creates contiguous block of memory for all frame data, optimizing GPU processing
    pixel_values = torch.cat(pixel_values_list)

    return pixel_values, num_patches_list


@app.function(image=image, volumes={MODAL_VOLUME_PATH: VOL})
def load_video(video_path, bound=None, input_size=448, max_num_patches=1, num_segments=28, get_frame_by_duration=False):
    """
    Load and preprocess video frames for InternVideo model.
    - input_size:
        - control spatial resolution of processed video frames.
        - higher values increase visual detail but require more memory/processing
    - max_num_patches:
        - limits max number of patches each frame can be divided into
        - higher values allow dividing frames w/ irregular aspect ratio into multiple tiles,
            which preserves information (i.e. a rectangle-shaped frame is not squashed into a square)
    - num_segments:
        - number of evenly-spaced frames to sample from the video
        - higher values increase temporal detail, but require more memory
    - get_frame_by_duration:
        - if set to True, the number of frames is determined by the video duration.

    Returns:
    - pixel_values:
        - stores processed video frames as PyTorch tensors
    - num_patches_list:
        - list of number of patches that frame i has been divided into
    """

    # open video file and use CPU context for video decoding
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    # define preprocessing pipeline we will use on each frame
    transform = build_transform.local(input_size=input_size)

    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)

    # calculates which frames to sample so that they are evenly-spaced
    frame_indices = get_index.local(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    # print(f" 🌞🌞🌞 FPS: {fps}, Segments: {num_segments} 🌞🌞🌞")
    # print(f" 🌞🌞🌞 Selected frames: {frame_indices} 🌞🌞🌞")
    pixel_values_list, num_patches_list = [], []

    for frame_index in frame_indices:
        # load frame and convert to RGB numpy array
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")

        # dividing frames into multiple square patches based on aspect ratio to optimize model input
        img = dynamic_preprocess.local(img, image_size=input_size, use_thumbnail=True, max_num_patches=max_num_patches)

        # apply transformation to each patch (turning it into a tensor of normalized pixel values) ...
        pixel_values = [transform(tile) for tile in img]

        # ... and stack them into a single tensor
        pixel_values = torch.stack(pixel_values)

        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    # create a single tensor, which is a combination of all the individual frames that have been processed into tensors
    # -> creates contiguous block of memory for all frame data, optimizing GPU processing
    pixel_values = torch.cat(pixel_values_list)

    return pixel_values, num_patches_list


@app.function(image=image, volumes={MODAL_VOLUME_PATH: VOL}, timeout=10000)
def load_video_with_cache(video_path, bound=None, input_size=448, max_num_patches=1, num_segments=28, sampling_rate=5):
    """Cache-aware video loading function"""
    VOL.reload()

    float_bound = [float(b) if b is not None else None for b in bound]

    # Generate a cache key based on parameters
    human_cache_key = f"{video_path}_{float_bound}_{input_size}_{max_num_patches}_{num_segments}"
    cache_key = hashlib.md5(human_cache_key.encode()).hexdigest()
    cache_dir = os.path.join(MODAL_VOLUME_PATH, "video_cache")
    os.makedirs(cache_dir, exist_ok=True)

    pixel_values_path = os.path.join(cache_dir, f"{cache_key}_pixels.pt")
    patches_path = os.path.join(cache_dir, f"{cache_key}_patches.json")
    print(f"🟨🟨🟨🟨🟨 Human cache key: {human_cache_key}")
    print(f"🟨🟨🟨🟨🟨 Hashed cache key: {cache_key}")
    print(f"🟨🟨🟨🟨🟨 Pixel values path: {pixel_values_path}")
    print(f"🟨🟨🟨🟨🟨 Patches path: {patches_path}")

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
        return pixel_values, num_patches_list

    # Otherwise, process video and cache results
    # pixel_values, num_patches_list = load_video.remote(
    #     video_path, bound, input_size, max_num_patches, num_segments
    # )
    print(f"💣💣💣 No cache for {os.path.basename(video_path)} - calling smart_load_video 💣💣💣")
    pixel_values, num_patches_list = smart_load_video.remote(
        video_path,
        bound,
        input_size,
        max_num_patches,
        num_segments,
        sampling_rate,
    )

    # Save to cache
    torch.save(pixel_values, pixel_values_path)
    with open(patches_path, 'w') as f:
        json.dump(num_patches_list, f)
    VOL.commit()

    return pixel_values, num_patches_list


@app.function(image=image)
def similarity(text1, text2):
    """Calculate simple cosine similarity between texts using TF-IDF"""

    print(f"🦕🦕  Calculating similarity between texts {text1} and {text2}...  🦕🦕")
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])

    # Calculate cosine similarity
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    print(f"🦕🦕   Similarity is {similarity}  🦕🦕")
    return similarity


@app.function(image=image)
def consolidate_chunk_results(chunk_results, query_prompt):
    """Filter and aggregate results from video chunks"""
    print(f"❤️‍🩹❤️‍🩹❤️‍🩹 Consolidating {len(chunk_results)} chunk results for {query_prompt}... ❤️‍🩹❤️‍🩹❤️‍🩹")
    # Sort by start time
    chunk_results.sort(key=lambda x: x["start_time"])

    # Extract play type from prompt if applicable
    play_type = "basketball plays"
    if "for " in query_prompt and "s." in query_prompt:
        play_part = query_prompt.split("for ")[1].split("s.")[0]
        if len(play_part) < 30:  # Reasonable length for a play type
            play_type = play_part + "s"

    # Format header
    consolidated = f"Results for {play_type}\n\n"

    # Track valid results and total execution time
    valid_results = []
    total_exec_time = 0

    # Process each chunk result
    for result in chunk_results:
        # Add to total execution time
        total_exec_time += result["exec_time"]

        # Skip empty results or "No examples" results
        output = result["output"]
        if not output or "No " in output and " visible in this segment" in output:
            continue

        # Add to valid results with proper timestamps
        valid_results.append({
            "start_time": result["start_time"],
            "end_time": result["end_time"],
            "description": output
        })

    # Deduplicate similar descriptions (often happens with overlapping chunks)
    distinct_results = []
    for result in valid_results:
        # Check if this result is distinct from what we already have
        is_distinct = True
        for existing in distinct_results:
            # If descriptions are very similar, consider them duplicates
            if similarity(result["description"], existing["description"]) > 0.7:
                is_distinct = False
                break

        if is_distinct:
            distinct_results.append(result)

    # Format final output (up to 5 results)
    top_results = distinct_results[:min(5, len(distinct_results))]

    for idx, result in enumerate(top_results, 1):
        consolidated += f"Example {idx}: [{result['start_time']:.1f}s - {result['end_time']:.1f}s]\n"
        consolidated += result["description"] + "\n\n"

    if not top_results:
        consolidated += f"No clear examples of {play_type} were found in this video."

    return consolidated, total_exec_time


@app.function(
    image=image,
    gpu=GPU_CHOICE,
    volumes={MODAL_VOLUME_PATH: VOL},
    timeout=50000
)
def internvideo_chunked_query_pipeline(
    prompt: str,
    video_path: str,
    num_segments: int = 96,
    chunk_duration: float = 15.0,
    sampling_rate: int = 5,
    overlap : float = 0.5,
):
    # clear GPU cache before processing (to avoid overflows when processing long videos)
    torch.cuda.empty_cache()

    generation_config = dict(
        do_sample=False,
        max_new_tokens=1024,
        num_beams=1,
    )

    # get video duration
    vr = VideoReader(video_path, ctx=cpu(0))
    total_duration = len(vr) / float(vr.get_avg_fps())
    del vr

    all_results = []

    for start_time in np.arange(0, total_duration - chunk_duration / 2, chunk_duration * (1 - overlap)):
        end_time = min(start_time + chunk_duration, total_duration)
        bound = [start_time, end_time]

        print(f"⭕️⭕️⭕️⭕️ Processing chunk {start_time:.1f}s to {end_time:.1f}s... ⭕️⭕️⭕️⭕️")

        with torch.no_grad():
            pixel_values, num_patches_list = load_video_with_cache.remote(
                video_path=video_path,
                bound=bound,
                input_size=448,
                max_num_patches=1,
                num_segments=num_segments,
                sampling_rate=sampling_rate)
            if model_server.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                print(" 🍺🍺 Using bfloat16 precision for GPU with compute capability >= 8.0 🍺🍺")
                pixel_values = pixel_values.to(model_server.device).to(torch.bfloat16)
            else:
                print(" 🍺🍺 Using normal precision for GPU with compute capability < 8.0 🍺🍺")
                print(f" 🍺🍺 Pixel values: shape {pixel_values.shape}, type {type(pixel_values[0])} 🍺🍺")
                pixel_values = pixel_values.to(model_server.device)

            chunk_prompt = f"Video segment from {start_time:.1f}s to {end_time:.1f}s:\n{prompt}"
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            query = video_prefix + chunk_prompt

            # Record time before the call
            start_time = time.time()

            # Call model.chat
            output, _ = model_server.model.chat(
                model_server.tokenizer,
                pixel_values,
                query,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True)

            # Calculate and print execution time
            exec_time = time.time() - start_time

            all_results.append({
                "start_time": start_time,
                "end_time": end_time,
                "output": output,
                "exec_time": exec_time
            })

    # Combine results from all chunks
    final_output, total_exec_time = consolidate_chunk_results.remote(all_results, prompt)
    return (final_output, total_exec_time)


@app.function(image=image)
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


@app.function(image=image)
def print_output(prompt: str, output : str, exec_time : float):
    print("=====================================================================")
    print(f" ⏳⏳ Computed in {exec_time} seconds using InternVideo2.5 ⏳⏳")
    print(f" Prompt: {prompt}\n\n")
    print(f" Result: {output}")
    print("=====================================================================")


def ensure_divisible_by_seven(num):
    """Ensure the number of segments is divisible by 7"""
    return ((num + 6) // 7) * 7


@app.function(
    image=image,
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
    do_list_frames_test: bool = False,
    num_segments: int = 28,
    sampling_rate: int = 5,
    chunk_duration: float = 15.0,
    overlap_fraction : float = 0.5,
):
    """
    Args:
        url: URL of the basketball video to analyze
        query: Query to search for in the videos
        num_results: Number of results to return
        embeddings_dir: Directory to store embeddings
        output_dir: Directory to save result clips
        iv: Whether or not to use InternVideo2.5 query-based pipeline
    """
    # keep at least 1 GPU warm for model loading since we're gonna need at least 1 model instance
    model_server.keep_warm(1)

    video_path, metadata = download_video_with_cache.remote(url)
    query_prompts = []

    if do_list_frames_test:
        list_prompt = (
            "Please provide a detailed frame-by-frame analysis of this basketball video.\n\n"
            "For EACH frame you see:\n"
            "1. Describe all visible players (jersey numbers, positions on court, actions)\n"
            "2. Track the ball movement\n"
            "3. Note any tactical formations or plays being executed\n"
            "4. Identify court position and game context\n"
            "5. Describe any visible scoreboard information\n\n"
            "Be extremely specific about what you can see in each individual frame. "
            "Start each frame description with 'FRAME X:' where X is the frame number. "
            "Focus only on what is directly observable in each specific frame, not on inferences about the game."
        )
        query_prompts.append(list_prompt)

    if do_summarize_test:
        summary_prompt = (
            "Analyze this basketball video and provide a comprehensive summary in point form. Include:\n"
            "1. Key game information (teams playing, general context if visible)\n"
            "2. Major scoring plays (dunks, three-pointers, layups)\n"
            "3. Standout defensive plays\n"
            "4. Notable player performances and statistics if shown\n"
            "5. Any tactical patterns or strategies employed by either team\n"
            "6. Highlight moments with timestamps if possible\n"
            "7. Game flow and momentum shifts\n"
            "Be specific about what you can clearly see in the footage.\n"
            "If certain details aren't visible or clear, acknowledge the limitations rather than making assumptions.\n"
        )
        query_prompts.extend(summary_prompt)

    if do_queries_test:
        plays = ["slam dunk", "three-pointer shot", "crossover dribble", "layup", "flashy pass", "regular pass"]
        query_prompts.extend([get_prompt_from_play.local(play) for play in plays])

    model_server.keep_warm(min(gpu_count // 2, len(query_prompts)))

    # Process each prompt with appropriate pipeline
    for prompt in query_prompts:
        output, exec_time = internvideo_chunked_query_pipeline.remote(
            prompt,
            video_path=video_path,
            num_segments=num_segments,
            sampling_rate=sampling_rate,
            chunk_duration=chunk_duration,
            overlap=overlap_fraction,
        )
        print_output.local(prompt, output, exec_time)


@app.local_entrypoint()
def entrypoint(
    url: str = "https://www.youtube.com/watch?v=wgVOgGLtPtc",
    query: str = None,
    num_results: int = 5,
    embeddings_dir: str = f"{MODAL_VOLUME_PATH}embeddings",
    output_dir: str = f"{MODAL_VOLUME_PATH}results",
    gpus: int = 10,
    q: bool = False,    # do the queries test
    s: bool = False,    # do the summarize test
    l: bool = False,    # do the list frames test
    c: float = 15.0,    # chunk length in seconds
    o: float = 0.5,     # overlap fraction
    n: int = 32,        # number of segments
    r: int = 5,         # sampling rate
):
    main.remote(
        url=url,
        query=query,
        num_results=num_results,
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        gpu_count=gpus,
        do_queries_test=q,
        do_summarize_test=s,
        do_list_frames_test=l,
        num_segments=ensure_divisible_by_seven(n),
        sampling_rate=r,
        chunk_duration=c,
        overlap_fraction=o,
    )


# ARCHIVE OF CODE THAT MIGHT BE USEFUL LATER
# .run_commands(
    #     # needed for faiss install
    #     "apt-get -qq -y update && apt-get -qq -y upgrade",
    #     "apt-get -qq -y install git wget cmake libblas-dev liblapack-dev swig",  # clang-12",
    #     "apt-get -qq install -y --allow-downgrades clang-8",

    #     # installer newer version of CMake necessary to build faiss
    #     ("wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31."
    #      "6-linux-x86_64.tar.gz"),
    # )
    # .run_commands(
    #     "mkdir /cmake-3.31.6",
    #     "tar xzf cmake-3.31.6-linux-x86_64.tar.gz",
    #     "mv cmake-3.31.6-linux-x86_64/* /cmake-3.31.6 && rm -rf cmake-3.31.6-linux-x86_64/",
    #     "ls /cmake-3.31.6",
    #     "cp /cmake-3.31.6/bin/* /usr/bin/",
    #     "cp -r /cmake-3.31.6/share/* /usr/share/",
    #     # "cd /cmake-3.31.6",
    #     # "echo alias cmake=$HOME/cmake-3.31.6/bin/cmake >> ~/.bashrc",
    #     "which cmake",
    #     "cmake --version",
    # )
    # .run_commands(
    #     # faiss installation commands
    #     "git clone https://github.com/facebookresearch/faiss.git",
    #     ("cd faiss && "
    #      "cmake -B build "
    #         "-DCMAKE_CXX_COMPILER=clang++-8 "
    #         "-DPython_EXECUTABLE=$(which python3) "
    #         "-DFAISS_ENABLE_PYTHON=ON "
    #         "-DBUILD_SHARED_LIBS=ON "
    #         "-DFAISS_OPT_LEVEL=generic "
    #         "-DCMAKE_BUILD_TYPE=Release "
    #         "-DBUILD_TESTING=OFF "),
    #     "make -C faiss/build -j faiss",
    #     "make -C faiss/build -j swigfaiss",
    #     "cd faiss/build/faiss/python && python setup.py install",
    #     "make -C faiss/build install",
    # )
