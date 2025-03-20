#!.venv/bin/python3
import yt_dlp
import argparse
import os

parser = argparse.ArgumentParser(description=('''
PURPOSE:
    This script is used to test a cookie file's validity and/or  by trying to use
    it to download a video from YouTube using the cookie file as authentication.

USAGE:
    python3 download_test.py
    [{-h || --help}}]
    [{-d | --delete}]
    [{-c | --cookiefile} <cookie_file>]
    [{-v | --video_url} <video_url>]
    [{-o | --output} <output_filename>]

OPTIONS:
    -d: delete video after downloading
    -c <cookie_file>: specify the cookie file to use
       (default: "./cookies.txt")
    -v <video_url>: specify the video URL to download
       (default: "https://www.youtube.com/watch?v=5jh6lrsmKlI")

*** TO OBTAIN A WORKING COOKIE FILE: ***
    - in Chrome, install the "EditThisCookie v3" or "Get cookies.txt LOCALLY" extension
        - in Chrome extension settings, click "Allow in incognito" for the extension
    - open a new private browsing/incognito window and log into YouTube
        - open a new tab and close the YouTube tab
        - export youtube.com cookies from the browser (in Netscape format)
        - then close the private browsing/incognito window
          so the session is never opened in the browser again
    - pass the exported cookie file as the second argument to this script,
        or else name it "cookies.txt" and place it in the same directory as this script
'''.strip()))

parser.add_argument('-d',
                    action='store_true',
                    help='Delete video file(s) after downloading')
parser.add_argument('-c',
                    '--cookiefile',
                    type=str,
                    default='cookies.txt',
                    help='Specify the cookie file to use\n\t(default: "./cookies.txt")')
parser.add_argument('-v',
                    '--video_url',
                    type=str,
                    default='https://www.youtube.com/watch?v=IHTvqrEOVg4',
                    help='Video URL\n\t(default: "https://www.youtube.com/watch?v=IHTvqrEOVg4")')
parser.add_argument('-o',
                    '--output',
                    type=str,
                    help='Specify the filename for the downloaded video')
args = parser.parse_args()

should_delete = args.d
url = args.video_url
ydl_opts = {
    'cookiefile': args.cookiefile,
    # 'format': 'bestvideo+bestaudio/best',
    'format': 'mp3/bestvideo',
    'extractor_args': {
        'youtubetab': {
            'skip': 'authcheck',
        }
    },
    'outtmpl': args.output if args.output else '%(title)s.%(ext)s',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    try:
        info = ydl.extract_info(url, download=True)
        filename = info['title'] + '.' + info['ext']
        print(f"Downloaded successfully to {filename}")
        video_height = info.get('height')
        video_width = info.get('width')
        resolution = f"{video_width}x{video_height}" if video_width and video_height else "unknown"
        print(f"🔍 Video resolution: {resolution} 🔍")

        if should_delete:
            if os.path.exists(ydl.prepare_filename(info)):
                os.remove(ydl.prepare_filename(info))
            else:
                print(f"Could not find file to delete: {ydl.prepare_filename(info)}")
            print('Video deleted as desired.')

    except yt_dlp.utils.DownloadError as e:
        print('\nDownload failed.')
        e.with_traceback(e.__traceback__)
        print("\nExecution info: ", e.exc_info)
        print("\nError message:", e.msg)
        print('\nExiting...')
        exit(0)
