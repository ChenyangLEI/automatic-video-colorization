import requests

def download_file_from_google_drive(id, destination):
    """
    Download google drive drive to google drive.

    Args:
        id: (str): write your description
        destination: (str): write your description
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    """
    Get the access token.

    Args:
        response: (todo): write your description
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    """
    Saves response content to destination.

    Args:
        response: (todo): write your description
        destination: (str): write your description
    """
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

print('Dowloading VGG-19 Model (510Mb)')
download_file_from_google_drive('0B_B_FOgPxgFLRjdEdE9NNTlzUWc', 'VGG_Model/imagenet-vgg-verydeep-19.mat')

print('Downloading ckpt_woflow (239Mb)')
download_file_from_google_drive('1yL8x7RL_82Mvyh_ebmh3uF6wiPOO2dU7','ckpt_woflow.zip')
