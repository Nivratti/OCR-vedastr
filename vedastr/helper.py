from gdown import download
from loguru import logger


def download_drive_file(file_id, output):
    """
    Download file from google drive

    Args:
        file_id (str): google drive file-id to download
        output (str): path to write file

    Returns:
        Bool: Return True if successfull else False.
    """
    try:
        # download from google drive
        url = f'https://drive.google.com/uc?id={file_id}'
        logger.info("downloading file from google drive.")
        download(url, output, quiet=False)
        return True
    except Exception as e:
        logger.error(e)
        logger.error("Error while downloading file from google drive. Manually add model checkpoint.")
        return False