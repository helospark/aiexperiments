# EmbeddingGemma-2

Index the files on my computer, then media_viewer can search the files (mainly images) based on text.

Usage

 1. Install dependencies: `pip install -U sentence-transformers chromadb pillow` and `sudo apt install python3-tk xdg-utils ffmpeg`
 2. Index files, update path in `index_images.py`, then run: `python3 index_images.py`, it will take a while to index everything
 3. View images using `python3 media_viewer.py`
