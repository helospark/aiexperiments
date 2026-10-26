#!/usr/bin/env python3

"""Simple Tkinter GUI for the existing EmbeddingGemma/ChromaDB media search.

Expected existing module:
    search_images.py
with:
    search_images(model, collection, query, top_k=40) -> list

Each result can be a path string/Path, or a dict containing a "path" key
(or "id" as a fallback), and optionally "filename" and "similarity".
"""

import os
import subprocess
import queue
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from concurrent.futures import ThreadPoolExecutor

import chromadb
from PIL import Image, ImageOps, ImageTk
from sentence_transformers import SentenceTransformer


# Keep these settings aligned with your existing indexer/search script.
MODEL_NAME = "google/embeddinggemma-2"
CHROMA_DIR = Path(".image_chroma_db")
COLLECTION_NAME = "images"
TOP_K = 400
THUMBNAIL_SIZE = (180, 135)
GRID_COLUMNS = 4

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"
}
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"
}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".html", ".py"}


def search_images(model, collection, query, top_k=10):
    """Search the image database using a text query."""

    if collection.count() == 0:
        print("The image database is empty. Run index_images.py first.")
        return

    # Encode the text query.
    # SearchQuery is the recommended task prompt for text retrieval.
    query_embedding = model.encode(
        query,
        prompt_name="SearchQuery",
        convert_to_numpy=True,
    )

    # Search the database using the query embedding.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    if not ids:
        print("No matching images found.")
        return

    print(f"\nQuery: {query}")
    print(f"Top {len(ids)} results:\n")

    results = []

    for rank, (image_id, distance, metadata) in enumerate(
        zip(ids, distances, metadatas),
        start=1,
    ):
        # Chroma's cosine distance is 1 - cosine similarity.
        similarity = 1.0 - distance
        media_path = metadata["path"]

        if not Path(media_path).is_file():
            continue

        results.append({
            "path": media_path,
            "similarity": similarity,
        })

    return results

def normalize_result(item):
    """Convert common search result formats into (Path, label)."""
    if isinstance(item, (str, Path)):
        path = Path(item)
        return path, path.name

    if isinstance(item, dict):
        raw_path = item.get("path") or item.get("id")
        if raw_path:
            path = Path(raw_path)
            label = item.get("filename") or path.name
            score = item.get("similarity", item.get("score"))
            if isinstance(score, (float, int)):
                label = f"{label}\nSimilarity: {score:.3f}"
            return path, label

    raise TypeError(
        "search_images() must return paths or dictionaries containing "
        f"a 'path' (or 'id') key; got {item!r}"
    )


def make_thumbnail(path: Path):
    """Return a PIL thumbnail, or None if a thumbnail cannot be made."""
    suffix = path.suffix.lower()

    try:
        if suffix in IMAGE_EXTENSIONS:
            with Image.open(path) as im:
                # For GIFs, use the first frame.
                try:
                    im.seek(0)
                except EOFError:
                    pass
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail(THUMBNAIL_SIZE)
                return im.copy()

        if suffix in VIDEO_EXTENSIONS:
            # Use ffmpeg, if installed, to extract a frame near the start.
            # No additional Python video package is required.
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                thumbnail_path = tmp.name
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-ss", "1", "-i", str(path),
                        "-frames:v", "1", "-vf",
                        f"scale={THUMBNAIL_SIZE[0]}:{THUMBNAIL_SIZE[1]}:"
                        "force_original_aspect_ratio=decrease",
                        "-y", thumbnail_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                with Image.open(thumbnail_path) as im:
                    im = im.convert("RGB")
                    im.thumbnail(THUMBNAIL_SIZE)
                    return im.copy()
            except (OSError, subprocess.SubprocessError):
                return None
            finally:
                try:
                    os.unlink(thumbnail_path)
                except OSError:
                    pass
    except (OSError, ValueError):
        return None

    return None


def open_media(path: Path):
    """Open with the desktop default handler, then try the requested fallbacks."""
    if not path.exists():
        messagebox.showerror("File not found", f"This file no longer exists:\n{path}")
        return

    # xdg-open delegates to the user's OS/desktop MIME associations.
    try:
        subprocess.Popen(["xdg-open", str(path)])
        return
    except OSError:
        pass

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        fallback = "eom"
    elif suffix in VIDEO_EXTENSIONS:
        fallback = "vlc"
    elif suffix in TEXT_EXTENSIONS:
        fallback = "pluma"
    else:
        fallback = None

    if fallback:
        try:
            subprocess.Popen([fallback, str(path)])
            return
        except OSError:
            pass

    messagebox.showerror(
        "Could not open file",
        f"Could not open {path.name} with xdg-open"
        + (f" or {fallback}." if fallback else ".")
    )


class MediaSearchGUI:
    def __init__(self, root, model, collection):
        self.root = root
        self.model = model
        self.collection = collection
        self.result_cards = []
        self.photo_refs = []  # Keep Tk images alive.
        self.root.title("Media Search")
        self.root.geometry("1000x750")
        self.thumbnail_queue = queue.Queue()
        self.thumbnail_executor = ThreadPoolExecutor(max_workers=4)
        self.thumbnail_jobs = {}       # index -> Future
        self.thumbnail_generation = 0
        self.thumbnail_prefetch_rows = 2
        self.thumbnail_check_pending = False
        self.search_generation = 0
        self.root.after(100, self._process_thumbnail_queue)

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        self.query_var = tk.StringVar()
        self.search_entry = ttk.Entry(top, textvariable=self.query_var)
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<Return>", lambda _event: self.start_search())

        self.search_button = ttk.Button(top, text="Search", command=self.start_search)
        self.search_button.pack(side="left", padx=(8, 0))
        self.search_entry.bind("<Control-a>", self._select_all)
        self.search_entry.bind("<Control-A>", self._select_all)

        self.status_var = tk.StringVar(value="Enter a search query.")
        ttk.Label(root, textvariable=self.status_var, padding=(10, 0, 10, 8)).pack(
            anchor="w"
        )

        # Scrollable results area.
        outer = ttk.Frame(root)
        outer.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.results_frame = ttk.Frame(self.canvas)
        self.results_frame.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.results_frame, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(
                self.canvas_window, width=event.width
            ),
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse-wheel scrolling on Linux.
        self._scroll_remainder = 0

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        
        self.canvas.bind("<Configure>", self._schedule_thumbnail_check)
        self.root.bind_all("<MouseWheel>", self._schedule_thumbnail_check, add="+")
        self.root.bind_all("<Button-4>", self._schedule_thumbnail_check, add="+")
        self.root.bind_all("<Button-5>", self._schedule_thumbnail_check, add="+")

        self.search_entry.focus_set()

    def _schedule_thumbnail_check(self, event=None):
        if self.thumbnail_check_pending:
            return

        self.thumbnail_check_pending = True
        self.root.after_idle(self._check_visible_thumbnails)


    def _check_visible_thumbnails(self):
        self.thumbnail_check_pending = False

        self.root.update_idletasks()

        # Canvas viewport, expressed in canvas coordinates.
        viewport_top = self.canvas.canvasy(0)
        viewport_bottom = viewport_top + self.canvas.winfo_height()

        # Determine the visible card rows using their actual geometry.
        visible_rows = set()

        for index, card in enumerate(self.result_cards):
            # winfo_y() is relative to results_frame, which is inside the canvas.
            top = self.canvas.canvasy(
                card.winfo_rooty() - self.canvas.winfo_rooty()
            )
            bottom = top + card.winfo_height()
            
            if bottom > viewport_top and top < viewport_bottom:
                visible_rows.add(index // GRID_COLUMNS)

        if not visible_rows:
            return

        first_row = max(0, min(visible_rows) - self.thumbnail_prefetch_rows)
        last_row = max(visible_rows) + self.thumbnail_prefetch_rows
        
        first_index = first_row * GRID_COLUMNS
        last_index = min(
            len(self.result_cards),
            (last_row + 1) * GRID_COLUMNS,
        )

        for index in range(first_index, last_index):
            if index in self.thumbnail_jobs:
                continue
            

            path = self.result_paths[index]

            future = self.thumbnail_executor.submit(make_thumbnail, path)
            self.thumbnail_jobs[index] = future

            future.add_done_callback(
                lambda f, i=index, g=self.thumbnail_generation:
                    self.thumbnail_queue.put((i, g, f))
            )

    def _select_all(self, event=None):
        event.widget.select_range(0, tk.END)
        event.widget.icursor(tk.END)
        return "break"

    def _on_mousewheel(self, event):
        self._scroll_remainder -= event.delta / 120

        units = int(self._scroll_remainder)
        if units:
            self.canvas.yview_scroll(units, "units")
            self._scroll_remainder -= units

    def start_search(self):
        query = self.query_var.get().strip()
        if not query:
            return

        self.search_button.configure(state="disabled")
        self.status_var.set("Searching…")
        threading.Thread(target=self._search_worker, args=(query,), daemon=True).start()

    def _search_worker(self, query):
        try:
            results = search_images(self.model, self.collection, query, top_k=TOP_K)
            normalized = [normalize_result(item) for item in (results or [])]
            self.root.after(0, self._show_results, normalized)
        except Exception as exc:
            self.root.after(0, self._search_failed, str(exc))

    def _search_failed(self, error):
        self.search_button.configure(state="normal")
        self.status_var.set("Search failed.")
        messagebox.showerror("Search failed", error)

    def _process_thumbnail_queue(self):
        while True:
            try:
                index, generation, future = self.thumbnail_queue.get_nowait()
            except queue.Empty:
                break

            if generation != self.search_generation:
                continue

            try:
                thumb = future.result()
            except Exception as exc:
                print(f"Thumbnail generation failed: {exc}")
                thumb = None
            
            if index >= len(self.thumbnail_labels):
                continue

            image_label, path = self.thumbnail_labels[index]

            if thumb is None:
                image_label.configure(
                    text=f"No thumbnail\n{path.suffix.lower()}",
                )
                continue

            photo = ImageTk.PhotoImage(thumb, master=self.root)

            # Keep a reference so Tkinter doesn't garbage-collect the image.
            self.photo_refs.append(photo)
            print(f"Size={len(self.photo_refs)}")

            image_label.configure(
                image=photo,
                text="",
            )
            image_label.image = photo

        # Check again in 100 ms.
        self.root.after(100, self._process_thumbnail_queue)

    def _show_results(self, results):
        self.thumbnail_generation += 1
        generation = self.thumbnail_generation

        self.thumbnail_jobs.clear()
        self.result_cards = []
        self.result_paths = []
        self.thumbnail_labels = []
        self.photo_refs.clear()

        for child in self.results_frame.winfo_children():
            child.destroy()
        self.search_button.configure(state="normal")

        # Invalidate callbacks from previous searches.
        self.search_generation += 1
        generation = self.search_generation

        for child in self.results_frame.winfo_children():
            child.destroy()

        self.photo_refs.clear()

        if not results:
            self.status_var.set("No results found.")
            return

        self.status_var.set(
            f"Showing {len(results)} results. Loading thumbnails..."
        )

        # Keep references to widgets that need their thumbnail updated.
        self.thumbnail_labels = []

        for index, (path, label) in enumerate(results):
            row, col = divmod(index, GRID_COLUMNS)

            card = ttk.Frame(
                self.results_frame,
                padding=8,
                relief="groove",
            )
            card.grid(
                row=row,
                column=col,
                sticky="nsew",
                padx=5,
                pady=5,
            )
            self.results_frame.columnconfigure(col, weight=1)

            # Show a placeholder immediately.
            image_label = ttk.Label(
                card,
                text="Loading thumbnail...",
                width=22,
                anchor="center",
                cursor="hand2",
            )
            image_label.pack()

            image_label.bind(
                "<Button-1>",
                lambda _event, p=path: open_media(p),
            )

            name_label = ttk.Label(
                card,
                text=label,
                wraplength=190,
                justify="center",
                cursor="hand2",
            )
            name_label.pack(fill="x", pady=(6, 0))

            name_label.bind(
                "<Button-1>",
                lambda _event, p=path: open_media(p),
            )

            self.result_cards.append(card)
            self.result_paths.append(path)
            self.thumbnail_labels.append((image_label, path))

        self._schedule_thumbnail_check()



def main():
    # Avoid network checks if the model is already cached locally.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    root = tk.Tk()
    MediaSearchGUI(root, model, collection)
    root.mainloop()


if __name__ == "__main__":
    main()

