#!/usr/bin/python3

import requests


def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(server_response):
        for key, value in server_response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(server_response, file_destination):
        chunk_size = 32768

        with open(file_destination, "wb") as f:
            for chunk in server_response.iter_content(chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


google_file_id = "1cwXe8ANkhFqe2i_UNxpZu15y2HZ0N9KN"
save_path = "/root/singnet/semantic-segmentation-aerial/service/segnet_final_reference.pth"
try:
    download_file_from_google_drive(google_file_id, save_path)
    exit(0)
except Exception as e:
    print(e)
    exit(1)
