"""Helpers for writing web-friendly MP4 animations with matplotlib."""
import imageio_ffmpeg
import matplotlib as mpl
from matplotlib import animation

mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def add_audio(mp4, wav, bitrate="96k"):
    """Mux `wav` into `mp4` in place (video stream copied, audio encoded as AAC).

    On Windows, antivirus, indexing and sync services can briefly hold a freshly written file,
    and renaming such a file fails. So nothing is renamed: the video is copied to a temp folder
    and ffmpeg writes the final file directly (with a few retries).
    """
    import shutil
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    mp4 = Path(mp4)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "video_only.mp4"
        shutil.copyfile(mp4, src)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-i", str(src), "-i", str(wav),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", bitrate,
               "-shortest", "-movflags", "+faststart", str(mp4)]
        for attempt in range(15):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                break
            time.sleep(2)
        else:
            raise RuntimeError("ffmpeg could not write " + str(mp4) + ":\n" + r.stderr)
    print("added audio to", mp4)


def save_mp4(fig, update, frames, path, fps=30, poster=None):
    """Render `update(i)` for i in range(frames) to an H.264 MP4 (and a poster PNG)."""
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "27", "-movflags", "+faststart"])
    ani.save(str(path), writer=writer)
    if poster is not None:
        update(poster[0])
        fig.savefig(str(poster[1]), dpi=100, bbox_inches=None)
    print("wrote", path)
