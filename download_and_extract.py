import requests
from tqdm import tqdm
import zipfile
import os

def download_file_with_progress(url, output_path):
    """
    下载文件并显示进度条
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024  # 每次读取的字节数

    with open(output_path, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            file.write(chunk)
            bar.update(len(chunk))

def extract_zip(file_path, extract_to):
    """
    解压 ZIP 文件到指定文件夹
    """
    # 如果文件夹不存在，则新建
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print(f"Created directory: {extract_to}")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted files to {extract_to}")

if __name__ == "__main__":
    # 文件下载链接
    url = "https://snap.stanford.edu/conflict/conflict_data.zip"

    # 下载文件保存路径
    output_zip_path = "conflict_data.zip"

    # 解压目标文件夹
    extract_folder = "./data"

    # 下载文件
    print("Starting download...")
    download_file_with_progress(url, output_zip_path)

    # 解压文件
    print("Extracting files...")
    extract_zip(output_zip_path, extract_folder)

    print("Done!")