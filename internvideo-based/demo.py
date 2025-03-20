import numpy as np
import torch
from typing import Tuple, Dict
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import modal
import yt_dlp

# pip install transformers==4.40.1 av imageio torch opencv-python
# pip install decord (eva-decord on Mac)
# pip install flash-attn --no-build-isolation

app = modal.App("basketball-video-search")

# Modal volume & GPU selection
VOL = modal.Volume.from_name(name="basketball-analysis-ex")
MODAL_VOLUME_PATH = "/vol/"
GPU_CHOICE = "L40S:2"

# Image selection
CUDA_VERSION = "12.3.2"  # need 12.1.0 (deprecated) for faiss-gpu
PYTHON_VERSION = "3.10"
FLAVOR = "devel"         # includes full CUDA toolkit
OPERATING_SYS = "ubuntu22.04"
CUDNN8 = False

TAG = f"{CUDA_VERSION}-{('cudnn8-' if CUDNN8 else '')}{FLAVOR}-{OPERATING_SYS}"

SHARED_ENV_VARS = {
    "DEBIAN_FRONTEND": "noninteractive",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

SHARED_CMDS = [
    "pip install -U scikit-learn",
    "apt-get -qq update",
    "apt-get -qq -y install ffmpeg",  # ffmpeg needed for image processing
]

GPU_CMDS = [
    "pip install flash-attn --no-build-isolation",
]

CPU_CMDS = []

LOCAL_FILE_MAPPINGS = [
    ["cookies.txt", "/root/cookies.txt"],
]

LOCAL_PYTHON_SOURCES = [
    "_remote_module_non_scriptable",
]

PIP_PKGS = [
    "packaging",
    "transformers==4.40.1",
    "yt-dlp",
    "torch",
    "torchvision",
    "decord",
    "imageio",
    # "matplotlib",
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

image = (
    modal.Image
    .from_registry(f"nvidia/cuda:{TAG}", add_python="3.10")
    .env(SHARED_ENV_VARS)
    .run_commands(*SHARED_CMDS)
    .pip_install(*PIP_PKGS, extra_options="-q")
    .run_commands(*GPU_CMDS)
    .add_local_file(*LOCAL_FILE_MAPPINGS[0])
    .add_local_python_source(*LOCAL_PYTHON_SOURCES)
)


@app.function(image=image, gpu=GPU_CHOICE, volumes={MODAL_VOLUME_PATH: VOL}, timeout=2700)
def main(url: str):
    # model setting
    model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

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

    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), 
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), 
            T.ToTensor(), 
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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

    def dynamic_preprocess(image, min_num=1, max_num=6, image_size=1080, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(image, input_size=448, max_num=6):
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
        return frame_indices

    def get_num_frames_by_duration(duration):
        local_num_frames = 4
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)

        return num_frames

    def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        if get_frame_by_duration:
            duration = max_frame / fps
            num_segments = get_num_frames_by_duration(duration)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    # evaluation setting

    gen_config = dict(
        do_sample=False,
        # temperature=0.0,
        max_new_tokens=1024,
        # top_p=0.1,
        num_beams=3
    )
    video_path, _ = download_video_with_cache(url)
    num_segments = 250

    with torch.no_grad():
        question4 = (
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

        pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
        pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        # single-turn conversation
        question1 = "Describe this video in detail."
        question = video_prefix + question1
        output1, chat_history = model.chat(tokenizer, pixel_values, question, gen_config, num_patches_list=num_patches_list, history=None, return_history=True)
        print("Output 1: ", output1)

        # multi-turn conversation
        question2 = "How many people appear in the video?"
        output2, chat_history = model.chat(tokenizer, pixel_values, question2, gen_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)
        output2a, chat_history = model.chat(tokenizer, pixel_values, (video_prefix + question2), gen_config, num_patches_list=num_patches_list, history=None, return_history=True)
        print("Output 2 w/ history: ", output2)
        print("Output 2 w/o history: ", output2a)

        question3 = "Please describe the movements of the players in the video in temporal order."
        output3, chat_history = model.chat(tokenizer, pixel_values, question3, gen_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)
        output3a, chat_history = model.chat(tokenizer, pixel_values, (video_prefix + question3), gen_config, num_patches_list=num_patches_list, history=None, return_history=True)
        print("Output 3 w/ history: ", output3)
        print("Output 3 w/o history: ", output3a)

        output4, chat_history = model.chat(tokenizer, pixel_values, question4, gen_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)
        output4a, chat_history = model.chat(tokenizer, pixel_values, (video_prefix + question4), gen_config, num_patches_list=num_patches_list, history=None, return_history=True)
        print("Output 4 w/ history: ", output4)
        print("Output 4 w/o history: ", output4a)


@app.local_entrypoint()
def local_main(url: str = "https://www.youtube.com/watch?v=_g0Bx6o58aE"):
    main.remote(url)
