import requests

# Replace these with your actual file paths
file_path = 'docs/docx/2高数考试大纲.docx'

# URL of your file upload endpoint
url = "http://10.48.8.76:1202/"

# Open the files in binary mode
with open(file_path, 'rb') as file:
    files = {
        'file': file
    }
    response = requests.post(url, files=files)

print(response.json())
