#!/usr/bin/env python3
"""Submit a single-image Hunyuan3D Pro job and download its textured GLB."""

import argparse
import base64
import datetime
import hashlib
import hmac
import json
import os
import time
import urllib.request
from pathlib import Path


HOST = "ai3d.tencentcloudapi.com"
SERVICE = "ai3d"
VERSION = "2025-05-13"
AUTO_REGION = "auto"
DEFAULT_API_REGION = "ap-guangzhou"


def _hmac_sha256(key, message):
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def resolve_region(region):
    if region in (None, "", AUTO_REGION):
        return DEFAULT_API_REGION
    return region


class TencentAi3DClient:
    def __init__(self, secret_id, secret_key, region=AUTO_REGION):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = resolve_region(region)

    def call(self, action, payload):
        timestamp = int(time.time())
        date = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
        body = json.dumps(payload, separators=(",", ":"))
        hashed_body = hashlib.sha256(body.encode("utf-8")).hexdigest()
        headers = (
            "content-type:application/json; charset=utf-8\n"
            f"host:{HOST}\n"
            f"x-tc-action:{action.lower()}\n"
        )
        signed_headers = "content-type;host;x-tc-action"
        canonical_request = (
            f"POST\n/\n\n{headers}\n{signed_headers}\n{hashed_body}"
        )
        scope = f"{date}/{SERVICE}/tc3_request"
        string_to_sign = (
            "TC3-HMAC-SHA256\n"
            f"{timestamp}\n"
            f"{scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )
        secret_date = _hmac_sha256(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = _hmac_sha256(secret_date, SERVICE)
        secret_signing = _hmac_sha256(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        authorization = (
            f"TC3-HMAC-SHA256 Credential={self.secret_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        request = urllib.request.Request(
            f"https://{HOST}",
            data=body.encode("utf-8"),
            headers={
                "Authorization": authorization,
                "Content-Type": "application/json; charset=utf-8",
                "Host": HOST,
                "X-TC-Action": action,
                "X-TC-Region": self.region,
                "X-TC-Timestamp": str(timestamp),
                "X-TC-Version": VERSION,
            },
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            result = json.load(response)["Response"]
        if "Error" in result:
            error = result["Error"]
            raise RuntimeError(f"{error['Code']}: {error['Message']}")
        return result


def submit_job(client, image_path, model="3.1"):
    image_base64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    result = client.call(
        "SubmitHunyuanTo3DProJob",
        {
            "ImageBase64": image_base64,
            "Model": model,
            "GenerateType": "Normal",
            "EnablePBR": False,
            "FaceCount": 50000,
        },
    )
    return result["JobId"]


def _download(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def wait_for_result(client, job_id, output_path, interval=600, sleep=time.sleep, download=_download):
    while True:
        result = client.call("QueryHunyuanTo3DProJob", {"JobId": job_id})
        status = result["Status"]
        if status in ("WAIT", "RUN"):
            sleep(interval)
            continue
        if status == "FAIL":
            raise RuntimeError(
                f"{result.get('ErrorCode', 'FAIL')}: {result.get('ErrorMessage', '')}"
            )
        if status == "DONE":
            for file_3d in result.get("ResultFile3Ds", []):
                if file_3d.get("Type", "").upper() == "GLB":
                    output_path = Path(output_path)
                    download(file_3d["Url"], output_path)
                    return output_path
            raise RuntimeError("DONE response has no GLB result")
        raise RuntimeError(f"Unknown job status: {status}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--im", type=Path, help="single input image path")
    parser.add_argument("--output-path", type=Path, required=True, help="output .glb path")
    parser.add_argument(
        "--query-submitted",
        metavar="JOB_ID",
        help="skip submission and query an already submitted Hunyuan3D Pro job",
    )
    parser.add_argument(
        "--region",
        default=AUTO_REGION,
        help=(
            "Tencent Cloud Region header. Use 'auto' to keep the recommended "
            "nearest-access endpoint and send the current product region."
        ),
    )
    parser.add_argument("--model", default="3.1", choices=("3.0", "3.1"))
    parser.add_argument("--poll-interval", type=int, default=600, metavar="SECONDS")
    args = parser.parse_args(argv)

    if args.query_submitted is None and args.im is None:
        parser.error("--im is required unless --query-submitted is used")

    secret_id = os.environ["TENCENTCLOUD_SECRET_ID"]
    secret_key = os.environ["TENCENTCLOUD_SECRET_KEY"]
    client = TencentAi3DClient(secret_id, secret_key, args.region)
    if args.query_submitted is None:
        job_id = submit_job(client, args.im, args.model)
        print(f"Submitted job: {job_id}")
    else:
        job_id = args.query_submitted
        print(f"Querying submitted job: {job_id}")
    output = wait_for_result(client, job_id, args.output_path, args.poll_interval)
    print(f"Downloaded GLB: {output}")


if __name__ == "__main__":
    main()
