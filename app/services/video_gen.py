"""
AI Video Generation Service
Supports Sora 2, Sora 2 Pro, Veo 3.1 via LLM Hub unified API

This module provides AI-powered video generation capabilities,
allowing users to generate video clips from text prompts using
advanced AI models.
"""

import os
import time
from typing import Optional, List
from dataclasses import dataclass

import requests
from loguru import logger

from app.config import config
from app.models.schema import VideoGenModel, VideoAspect
from app.utils import utils


@dataclass
class VideoGenResult:
    """视频生成结果"""
    video_path: str
    duration: float
    prompt: str


class VideoGenerator:
    """AI 视频生成器基类"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = (30, 300)  # 连接超时30秒，读取超时300秒
        self.max_retries = 3
        self.poll_interval = 5  # 轮询间隔（秒）
        self.max_poll_time = 600  # 最大轮询时间（秒）

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _convert_aspect(self, aspect: VideoAspect) -> str:
        """转换视频宽高比格式"""
        if aspect == VideoAspect.landscape or aspect == VideoAspect.landscape.value:
            return "16:9"
        elif aspect == VideoAspect.portrait or aspect == VideoAspect.portrait.value:
            return "9:16"
        elif aspect == VideoAspect.square or aspect == VideoAspect.square.value:
            return "1:1"
        return "9:16"

    def generate(
        self,
        prompt: str,
        save_dir: str,
        aspect: VideoAspect = VideoAspect.portrait,
        duration: int = 5,
    ) -> Optional[VideoGenResult]:
        """生成视频（子类需实现）"""
        raise NotImplementedError


class SoraVideoGenerator(VideoGenerator):
    """Sora 2 / Sora 2 Pro 视频生成器"""

    def generate(
        self,
        prompt: str,
        save_dir: str,
        aspect: VideoAspect = VideoAspect.portrait,
        duration: int = 5,
    ) -> Optional[VideoGenResult]:
        """使用 Sora 生成视频"""
        # 1. 创建视频生成任务
        job_id = self._create_job(prompt, aspect, duration)
        if not job_id:
            return None

        # 2. 轮询任务状态
        video_url = self._poll_job_status(job_id)
        if not video_url:
            return None

        # 3. 下载视频
        video_path = self._download_video(video_url, save_dir, job_id)
        if not video_path:
            return None

        return VideoGenResult(video_path=video_path, duration=duration, prompt=prompt)

    def _create_job(
        self, prompt: str, aspect: VideoAspect, duration: int
    ) -> Optional[str]:
        """创建视频生成任务"""
        url = f"{self.base_url}/videos"
        headers = self._get_headers()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": self._convert_aspect(aspect),
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Creating Sora video job (attempt {attempt + 1}): {prompt[:50]}..."
                )
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    proxies=config.proxy,
                    timeout=self.timeout,
                )

                if response.status_code in [200, 201, 202]:
                    data = response.json()
                    job_id = data.get("id") or data.get("job_id") or data.get("video_id")
                    if job_id:
                        logger.info(f"Video job created: {job_id}")
                        return job_id
                    else:
                        logger.error(f"No job ID in response: {data}")
                else:
                    logger.error(
                        f"Failed to create job: {response.status_code} - {response.text}"
                    )

            except Exception as e:
                logger.error(f"Error creating video job: {str(e)}")

            if attempt < self.max_retries - 1:
                time.sleep(2)

        return None

    def _poll_job_status(self, job_id: str) -> Optional[str]:
        """轮询任务状态直到完成"""
        url = f"{self.base_url}/videos/{job_id}"
        headers = self._get_headers()
        start_time = time.time()

        while time.time() - start_time < self.max_poll_time:
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=config.proxy,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "").lower()

                    if status in ["completed", "succeeded", "done"]:
                        video_url = (
                            data.get("video_url")
                            or data.get("output_url")
                            or data.get("url")
                        )
                        if video_url:
                            logger.success(f"Video generation completed: {job_id}")
                            return video_url
                        else:
                            # 尝试获取视频内容端点
                            return f"{self.base_url}/videos/{job_id}/content"

                    elif status in ["failed", "error"]:
                        error_msg = data.get("error", "Unknown error")
                        logger.error(f"Video generation failed: {error_msg}")
                        return None

                    else:
                        progress = data.get("progress", "unknown")
                        logger.info(
                            f"Video generation in progress: {status} ({progress})"
                        )

                else:
                    logger.warning(f"Poll request failed: {response.status_code}")

            except Exception as e:
                logger.warning(f"Error polling job status: {str(e)}")

            time.sleep(self.poll_interval)

        logger.error(f"Video generation timed out after {self.max_poll_time}s")
        return None

    def _download_video(
        self, video_url: str, save_dir: str, job_id: str
    ) -> Optional[str]:
        """下载生成的视频"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        video_path = os.path.join(save_dir, f"ai-gen-{job_id}.mp4")

        # 如果文件已存在，直接返回
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            logger.info(f"Video already exists: {video_path}")
            return video_path

        try:
            headers = self._get_headers()
            logger.info(f"Downloading video from: {video_url}")

            response = requests.get(
                video_url,
                headers=headers,
                proxies=config.proxy,
                timeout=(30, 300),
                stream=True,
            )

            if response.status_code == 200:
                with open(video_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    logger.success(f"Video downloaded: {video_path}")
                    return video_path
                else:
                    logger.error("Downloaded file is empty or missing")
            else:
                logger.error(
                    f"Failed to download video: {response.status_code} - {response.text}"
                )

        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")

        return None


class VeoVideoGenerator(VideoGenerator):
    """Veo 3.1 视频生成器"""

    def generate(
        self,
        prompt: str,
        save_dir: str,
        aspect: VideoAspect = VideoAspect.portrait,
        duration: int = 5,
    ) -> Optional[VideoGenResult]:
        """使用 Veo 生成视频"""
        # 1. 启动预测任务
        operation_name = self._start_prediction(prompt, aspect, duration)
        if not operation_name:
            return None

        # 2. 轮询操作状态
        video_data = self._poll_operation(operation_name)
        if not video_data:
            return None

        # 3. 保存视频
        video_path = self._save_video(video_data, save_dir, operation_name)
        if not video_path:
            return None

        return VideoGenResult(video_path=video_path, duration=duration, prompt=prompt)

    def _start_prediction(
        self, prompt: str, aspect: VideoAspect, duration: int
    ) -> Optional[str]:
        """启动 Veo 预测任务"""
        # Veo API 可能使用不同的端点格式
        url = f"{self.base_url}/videos/generate"
        headers = self._get_headers()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "duration_seconds": duration,
            "aspect_ratio": self._convert_aspect(aspect),
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Starting Veo prediction (attempt {attempt + 1}): {prompt[:50]}..."
                )
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    proxies=config.proxy,
                    timeout=self.timeout,
                )

                if response.status_code in [200, 201, 202]:
                    data = response.json()
                    op_name = (
                        data.get("operation_name")
                        or data.get("name")
                        or data.get("id")
                    )
                    if op_name:
                        logger.info(f"Veo prediction started: {op_name}")
                        return op_name
                    else:
                        logger.error(f"No operation name in response: {data}")
                else:
                    logger.error(
                        f"Failed to start prediction: {response.status_code} - {response.text}"
                    )

            except Exception as e:
                logger.error(f"Error starting Veo prediction: {str(e)}")

            if attempt < self.max_retries - 1:
                time.sleep(2)

        return None

    def _poll_operation(self, operation_name: str) -> Optional[bytes]:
        """轮询操作状态并获取视频数据"""
        url = f"{self.base_url}/operations/{operation_name}"
        headers = self._get_headers()
        start_time = time.time()

        while time.time() - start_time < self.max_poll_time:
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=config.proxy,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    done = data.get("done", False)

                    if done:
                        result = data.get("result", {})
                        video_url = result.get("video_url") or result.get("uri")

                        if video_url:
                            # 下载视频
                            video_response = requests.get(
                                video_url,
                                headers=headers,
                                proxies=config.proxy,
                                timeout=(30, 300),
                            )
                            if video_response.status_code == 200:
                                logger.success(f"Veo video ready: {operation_name}")
                                return video_response.content

                        # 尝试从结果中直接获取视频数据
                        video_bytes = result.get("video_bytes")
                        if video_bytes:
                            import base64
                            return base64.b64decode(video_bytes)

                        logger.error("No video data in completed operation")
                        return None

                    else:
                        progress = data.get("metadata", {}).get("progress", "unknown")
                        logger.info(f"Veo prediction in progress: {progress}")

                else:
                    logger.warning(f"Poll request failed: {response.status_code}")

            except Exception as e:
                logger.warning(f"Error polling operation: {str(e)}")

            time.sleep(self.poll_interval)

        logger.error(f"Veo prediction timed out after {self.max_poll_time}s")
        return None

    def _save_video(
        self, video_data: bytes, save_dir: str, operation_name: str
    ) -> Optional[str]:
        """保存视频数据到文件"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 清理操作名称作为文件名
        safe_name = operation_name.replace("/", "_").replace("\\", "_")
        video_path = os.path.join(save_dir, f"ai-gen-{safe_name}.mp4")

        try:
            with open(video_path, "wb") as f:
                f.write(video_data)

            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                logger.success(f"Video saved: {video_path}")
                return video_path

        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")

        return None


def get_video_generator(model: str) -> Optional[VideoGenerator]:
    """
    工厂函数：获取对应的视频生成器

    Args:
        model: 模型名称 (sora-2, sora-2-pro, veo-3.1)

    Returns:
        对应的 VideoGenerator 实例，或 None
    """
    api_key = config.llmhub.get("api_key", "")
    base_url = config.llmhub.get("base_url", "https://api.llmhub.com.cn/v1")

    if not api_key:
        logger.error("LLM Hub API key is not configured")
        return None

    if model in ["sora-2", "sora-2-pro"]:
        return SoraVideoGenerator(api_key, base_url, model)
    elif model == "veo-3.1":
        return VeoVideoGenerator(api_key, base_url, model)

    logger.error(f"Unknown video generation model: {model}")
    return None


def generate_videos_from_prompts(
    task_id: str,
    prompts: List[str],
    model: str = "sora-2",
    aspect: VideoAspect = VideoAspect.portrait,
    duration: int = 5,
) -> List[str]:
    """
    根据提示词列表生成视频

    Args:
        task_id: 任务ID
        prompts: 视频生成提示词列表
        model: 视频生成模型
        aspect: 视频宽高比
        duration: 每个视频时长（秒）

    Returns:
        生成的视频文件路径列表
    """
    generator = get_video_generator(model)
    if not generator:
        logger.error("Failed to create video generator")
        return []

    video_paths = []
    save_dir = utils.task_dir(task_id)

    for i, prompt in enumerate(prompts):
        logger.info(f"Generating video {i + 1}/{len(prompts)}: {prompt[:50]}...")
        result = generator.generate(
            prompt=prompt,
            save_dir=save_dir,
            aspect=aspect,
            duration=duration,
        )

        if result and result.video_path:
            video_paths.append(result.video_path)
            logger.success(f"Video {i + 1}/{len(prompts)} generated: {result.video_path}")
        else:
            logger.warning(f"Failed to generate video for prompt: {prompt[:50]}...")

    logger.info(f"Generated {len(video_paths)}/{len(prompts)} videos")
    return video_paths


if __name__ == "__main__":
    # 测试代码
    test_prompts = ["A beautiful sunset over the ocean, cinematic, 4K quality"]
    videos = generate_videos_from_prompts(
        task_id="test-task",
        prompts=test_prompts,
        model="sora-2",
        aspect=VideoAspect.portrait,
        duration=5,
    )
    print(f"Generated videos: {videos}")
