"""Select an existing image transmission model for frame-independent video.

Example: python -m scripts.transmit_video --input data/input.mp4 --model jscc
"""
from src.classical.video_transmission import main

if __name__ == "__main__":
    main()
