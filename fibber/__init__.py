"""Top-level package for fibber."""

import os

import nltk


def get_root_dir():
    """Return ``~/.fibber``, the root dir for fibber to store datasets and common resources."""
    root_dir_path=os.getcwd()
    root_dir = os.path.join(root_dir_path, ".fibber")
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


# change cache directory
nltk.data.path += [os.path.join(get_root_dir(), "common", "nltk_data")]
