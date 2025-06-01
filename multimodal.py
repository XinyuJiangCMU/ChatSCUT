from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import docx
import io
import hashlib
import hmac
import json
import time
import sys
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

if sys.version_info[0] <= 2:
    from http.client import HTTPSConnection
else:
    from http.client import HTTPSConnection


def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

def is_image(file):
    image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return file.filename.split('.')[-1].lower() in image_extensions

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    text = ""
    doc = docx.Document(io.BytesIO(file.read()))
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def perform_ocr(image_file_path, secret_id, secret_key, region='ap-guangzhou', token=""):
    service = "ocr"
    host = "ocr.tencentcloudapi.com"
    version = "2018-11-19"
    action = "GeneralFastOCR"

    with open(image_file_path, 'rb') as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

    payload = json.dumps({
        "ImageBase64": image_base64,
    })

    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
    canonical_headers = f"content-type:{ct}\nhost:{host}\nx-tc-action:{action.lower()}\n"
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (http_request_method + "\n" +
                         canonical_uri + "\n" +
                         canonical_querystring + "\n" +
                         canonical_headers + "\n" +
                         signed_headers + "\n" +
                         hashed_request_payload)

    timestamp = int(time.time())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
    credential_scope = date + "/" + service + "/" + "tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    algorithm = "TC3-HMAC-SHA256"
    string_to_sign = (algorithm + "\n" +
                      str(timestamp) + "\n" +
                      credential_scope + "\n" +
                      hashed_canonical_request)

    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (algorithm + " " +
                     "Credential=" + secret_id + "/" + credential_scope + ", " +
                     "SignedHeaders=" + signed_headers + ", " +
                     "Signature=" + signature)

    headers = {
        "Authorization": authorization,
        "Content-Type": ct,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": timestamp,
        "X-TC-Version": version,
        "X-TC-Region": region
    }
    if token:
        headers["X-TC-Token"] = token

    try:
        req = HTTPSConnection(host)
        req.request("POST", "/", headers=headers, body=payload.encode("utf-8"))
        resp = req.getresponse()
        response_json = resp.read().decode('utf-8')
        data = json.loads(response_json)
        detected_texts = [detection['DetectedText'] for detection in data['Response']['TextDetections']]
        all_text = ''.join(detected_texts)
        return all_text
    except Exception as err:
        print(err)
        return None

def perform_asr(audio_file_path, secret_id, secret_key, region='', token=""):
    service = "asr"
    host = "asr.tencentcloudapi.com"
    version = "2019-06-14"
    action = "SentenceRecognition"

    with open(audio_file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    payload = json.dumps({
        "SubServiceType": 2,
        "EngSerViceType": "16k_zh-PY",
        "SourceType": 1,
        "VoiceFormat": "wav",
        "UsrAudioKey": "unique_audio_key",
        "Data": audio_base64,
        "DataLen": len(audio_data),
    })

    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
    canonical_headers = f"content-type:{ct}\nhost:{host}\nx-tc-action:{action.lower()}\n"
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (http_request_method + "\n" +
                         canonical_uri + "\n" +
                         canonical_querystring + "\n" +
                         canonical_headers + "\n" +
                         signed_headers + "\n" +
                         hashed_request_payload)

    timestamp = int(time.time())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
    credential_scope = date + "/" + service + "/" + "tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    algorithm = "TC3-HMAC-SHA256"
    string_to_sign = (algorithm + "\n" +
                      str(timestamp) + "\n" +
                      credential_scope + "\n" +
                      hashed_canonical_request)

    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (algorithm + " " +
                     "Credential=" + secret_id + "/" + credential_scope + ", " +
                     "SignedHeaders=" + signed_headers + ", " +
                     "Signature=" + signature)

    headers = {
        "Authorization": authorization,
        "Content-Type": ct,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": timestamp,
        "X-TC-Version": version
    }
    if region:
        headers["X-TC-Region"] = region
    if token:
        headers["X-TC-Token"] = token

    try:
        req = HTTPSConnection(host)
        req.request("POST", "/", headers=headers, body=payload.encode("utf-8"))
        resp = req.getresponse()
        response_json = resp.read().decode('utf-8')
        data = json.loads(response_json)
        result_content = data['Response']['Result']
        return result_content
    except Exception as err:
        print(err)
        return None

@app.route('/multimodal', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    combined_text = ""
    
    for file in files:
        if file.filename.endswith('.pdf'):
            combined_text += extract_text_from_pdf(file) + '\n'
        elif file.filename.endswith('.docx'):
            combined_text += extract_text_from_docx(file) + '\n'
        elif is_image(file):
            filename = secure_filename(file.filename)
            file_path = os.path.join("tmp", filename)
            file.save(file_path)
            combined_text += perform_ocr(file_path, secret_id, secret_key) + '\n'
        elif file.filename.endswith('.wav'):
            filename = secure_filename(file.filename)
            file_path = os.path.join("tmp", filename)
            file.save(file_path)
            combined_text += perform_asr(file_path, secret_id, secret_key) + '\n'
        else:
            return jsonify({"error": f"Unsupported file type: {file.filename}"}), 400

    return jsonify({"combined_text": combined_text})

if __name__ == '__main__':
    app.run(debug=True)
