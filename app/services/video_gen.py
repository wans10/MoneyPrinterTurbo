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
from app.models.schema import VideoGenModel, VideoAspect, VideoGenResolution
from app.utils import utils


# 缓存的模型列表
_cached_video_models = None
_cache_timestamp = 0
_cache_ttl = 300  # 5 分钟缓存


def get_available_video_models() -> List[dict]:
    """
    从 LLM Hub API 动态获取可用的视频生成模型列表
    
    Returns:
        模型列表，每个模型包含 id, name, endpoint_type
    """
    global _cached_video_models, _cache_timestamp
    
    # 检查缓存
    if _cached_video_models and (time.time() - _cache_timestamp) < _cache_ttl:
        return _cached_video_models
    
    api_key = config.llmhub.get("api_key", "")
    base_url = config.llmhub.get("base_url", "https://api.llmhub.com.cn/v1")
    
    if not api_key:
        logger.warning("LLM Hub API key not configured, returning default models")
        return _get_default_video_models()
    
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        
        if response.status_code == 200:
            data = response.json()
            all_models = data.get("data", [])
            
            # 筛选视频生成模型
            video_models = []
            video_keywords = ["video", "sora", "veo", "wan", "kling", "pika", "runway"]
            video_endpoints = ["openai-video", "gemini"]
            
            for model in all_models:
                model_id = model.get("id", "").lower()
                endpoints = model.get("supported_endpoint_types", [])
                
                # 检查是否是视频模型
                is_video = any(kw in model_id for kw in video_keywords)
                has_video_endpoint = any(ep in endpoints for ep in video_endpoints)
                
                # 排除图片生成模型 (t2i)
                is_image_model = "t2i" in model_id or "image" in model_id
                
                if (is_video or has_video_endpoint) and not is_image_model:
                    video_models.append({
                        "id": model.get("id"),
                        "name": _format_model_name(model.get("id")),
                        "endpoint_type": endpoints[0] if endpoints else "openai-video",
                    })
            
            if video_models:
                _cached_video_models = video_models
                _cache_timestamp = time.time()
                logger.info(f"Fetched {len(video_models)} video models from API")
                return video_models
        
        logger.warning(f"Failed to fetch models: {response.status_code}")
    except Exception as e:
        logger.warning(f"Error fetching video models: {e}")
    
    return _get_default_video_models()


def _format_model_name(model_id: str) -> str:
    """格式化模型 ID 为显示名称"""
    name_mapping = {
        "sora-2": "Sora 2",
        "sora-2-pro": "Sora 2 Pro",
        "veo-3.0-generate-001": "Veo 3.0",
        "veo-3.0-fast-generate-001": "Veo 3.0 Fast",
        "veo-3.1-generate-preview": "Veo 3.1",
        "veo-3.1-fast-generate-preview": "Veo 3.1 Fast",
        "wan2.6-t2v": "Wan 2.6",
    }
    return name_mapping.get(model_id, model_id)


def _get_default_video_models() -> List[dict]:
    """返回默认的视频模型列表"""
    return [
        {"id": "sora-2", "name": "Sora 2", "endpoint_type": "openai-video"},
        {"id": "sora-2-pro", "name": "Sora 2 Pro", "endpoint_type": "openai-video"},
        {"id": "veo-3.1-generate-preview", "name": "Veo 3.1", "endpoint_type": "gemini"},
        {"id": "wan2.6-t2v", "name": "Wan 2.6", "endpoint_type": "openai-video"},
    ]


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
    
    def _convert_resolution(self, resolution: str) -> dict:
        """转换分辨率为 API 参数"""
        resolution_map = {
            "480p": {"width": 854, "height": 480},
            "720p": {"width": 1280, "height": 720},
            "1080p": {"width": 1920, "height": 1080},
            "4k": {"width": 3840, "height": 2160},
        }
        return resolution_map.get(resolution, resolution_map["1080p"])

    def generate(
        self,
        prompt: str,
        save_dir: str,
        aspect: VideoAspect = VideoAspect.portrait,
        duration: int = 5,
        image_path: Optional[str] = None,
        resolution: str = "1080p",
    ) -> Optional[VideoGenResult]:
        """生成视频（子类需实现）
        
        Args:
            prompt: 文本提示词
            save_dir: 保存目录
            aspect: 视频宽高比
            duration: 视频时长（秒）
            image_path: 可选，用于 Image-to-Video 的图片路径
            resolution: 视频分辨率 (480p, 720p, 1080p, 4k)
        """
        raise NotImplementedError


class SoraVideoGenerator(VideoGenerator):
    """Sora 2 / Sora 2 Pro 视频生成器"""

    def generate(
        self,
        prompt: str,
        save_dir: str,
        aspect: VideoAspect = VideoAspect.portrait,
        duration: int = 5,
        image_path: Optional[str] = None,
        resolution: str = "1080p",
    ) -> Optional[VideoGenResult]:
        """使用 Sora 生成视频（支持 Text-to-Video 和 Image-to-Video）"""
        # 1. 创建视频生成任务
        job_id = self._create_job(prompt, aspect, duration, image_path, resolution)
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
        self, prompt: str, aspect: VideoAspect, duration: int, 
        image_path: Optional[str] = None, resolution: str = "1080p"
    ) -> Optional[str]:
        """创建视频生成任务"""
        url = f"{self.base_url}/videos"
        headers = self._get_headers()
        res = self._convert_resolution(resolution)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": self._convert_aspect(aspect),
            "width": res["width"],
            "height": res["height"],
        }
        
        # 如果提供了图片，使用 Image-to-Video 模式
        if image_path and os.path.exists(image_path):
            logger.info(f"Using Image-to-Video mode with: {image_path}")
            try:
                import base64
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                # 获取图片 MIME 类型
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".webp": "image/webp",
                }.get(ext, "image/jpeg")
                payload["image"] = f"data:{mime_type};base64,{image_data}"
            except Exception as e:
                logger.warning(f"Failed to encode image, falling back to text-to-video: {e}")

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
        image_path: Optional[str] = None,
        resolution: str = "1080p",
    ) -> Optional[VideoGenResult]:
        """使用 Veo 生成视频（支持 Text-to-Video 和 Image-to-Video）"""
        # 1. 启动预测任务
        operation_name = self._start_prediction(prompt, aspect, duration, image_path, resolution)
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
        self, prompt: str, aspect: VideoAspect, duration: int, 
        image_path: Optional[str] = None, resolution: str = "1080p"
    ) -> Optional[str]:
        """启动 Veo 预测任务"""
        url = f"{self.base_url}/videos/generate"
        headers = self._get_headers()
        res = self._convert_resolution(resolution)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "duration_seconds": duration,
            "aspect_ratio": self._convert_aspect(aspect),
            "width": res["width"],
            "height": res["height"],
        }
        
        # 如果提供了图片，使用 Image-to-Video 模式
        if image_path and os.path.exists(image_path):
            logger.info(f"Veo Image-to-Video mode with: {image_path}")
            try:
                import base64
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".webp": "image/webp",
                }.get(ext, "image/jpeg")
                payload["image"] = f"data:{mime_type};base64,{image_data}"
            except Exception as e:
                logger.warning(f"Failed to encode image for Veo: {e}")

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
        model: 模型名称 (sora-2, sora-2-pro, veo-3.1-generate-preview, veo-3.1-fast-generate-preview)

    Returns:
        对应的 VideoGenerator 实例，或 None
    """
    api_key = config.llmhub.get("api_key", "")
    base_url = config.llmhub.get("base_url", "https://api.llmhub.com.cn/v1")

    if not api_key:
        logger.error("LLM Hub API key is not configured")
        return None

    if model in ["sora-2", "sora-2-pro", "wan2.6-t2v"]:
        return SoraVideoGenerator(api_key, base_url, model)
    elif model in ["veo-3.0-generate-001", "veo-3.0-fast-generate-001", "veo-3.1-generate-preview", "veo-3.1-fast-generate-preview"]:
        return VeoVideoGenerator(api_key, base_url, model)

    logger.error(f"Unknown video generation model: {model}")
    return None


def generate_videos_from_prompts(
    task_id: str,
    prompts: List[str],
    model: str = "sora-2",
    aspect: VideoAspect = VideoAspect.portrait,
    duration: int = 5,
    image_paths: Optional[List[str]] = None,
    resolution: str = "1080p",
) -> List[str]:
    """
    根据提示词列表生成视频（支持 Text-to-Video 和 Image-to-Video）

    Args:
        task_id: 任务ID
        prompts: 视频生成提示词列表
        model: 视频生成模型
        aspect: 视频宽高比
        duration: 每个视频时长（秒）
        image_paths: 可选，图片路径列表用于 Image-to-Video
        resolution: 视频分辨率 (480p, 720p, 1080p, 4k)

    Returns:
        生成的视频文件路径列表
    """
    generator = get_video_generator(model)
    if not generator:
        logger.error("Failed to create video generator")
        return []

    video_paths = []
    save_dir = utils.task_dir(task_id)
    
    # 确保 image_paths 长度与 prompts 匹配或为空
    images = image_paths or []

    for i, prompt in enumerate(prompts):
        # 获取对应的图片（如果有）
        image_path = images[i] if i < len(images) else None
        
        if image_path:
            logger.info(f"Generating video {i + 1}/{len(prompts)} from image: {os.path.basename(image_path)}")
        else:
            logger.info(f"Generating video {i + 1}/{len(prompts)}: {prompt[:50]}...")
        
        result = generator.generate(
            prompt=prompt,
            save_dir=save_dir,
            aspect=aspect,
            duration=duration,
            image_path=image_path,
            resolution=resolution,
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
