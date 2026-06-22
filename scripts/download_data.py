import kagglehub
import pathlib
import shutil
import requests


def move_files_to_data_dir(source_dir: str, target_dir: str) -> None:
    """
    Move files from the source directory to the target directory.
    """
    source_path = pathlib.Path(source_dir)
    target_path = pathlib.Path(target_dir)

    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    if not target_path.exists():
        target_path.mkdir(parents=True)

    for file in source_path.iterdir():
        shutil.move(str(file), str(target_path))


def download_dataset_from_kaggle(dataset_name: str) -> str:
    """
    Download the dataset from Kaggle and return the path to the downloaded files.
    """

    data_path = pathlib.Path("data/raw")
    dataset_path = data_path / dataset_name.split("/")[-1]
    
    if not data_path.exists():
        data_path.mkdir(parents=True)

    if not dataset_path.parent.exists():
        dataset_path.parent.mkdir(parents=True)

    if dataset_path.exists():
        print(f"Dataset already exists at {dataset_path}. Skipping download.")
        return str(dataset_path)
    
    downloaded_path = kagglehub.dataset_download(dataset_name)
    downloaded_path = pathlib.Path(downloaded_path)
    if downloaded_path is None:
        raise ValueError(f"Failed to download dataset: {dataset_name}")
    else:
        move_files_to_data_dir(str(downloaded_path), str(data_path))
    return str(dataset_path)


def download_file(url: str, target_path: str) -> None:
    """
    Download a file from the given URL and save it to the target path.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    target_path = pathlib.Path(target_path)
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True)

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded file from {url} to {target_path}")
    return str(target_path)