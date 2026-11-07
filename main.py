#!/usr/bin/env python3
"""
TEL Robot Vision - Main Application Entrypoint
==============================================
Production-ready vision computing service for target detection and tracking.
Supports Windows w/ NVIDIA GPU and headless deployment (Raspberry Pi 5).

Usage:
    # Run with default configuration (GUI mode, local stream/fallback)
    python main.py

    # Run in headless mode for robot computer (Raspberry Pi 5)
    python main.py --headless

    # Specify custom HTTP MJPEG stream source
    python main.py --source "http://192.168.1.50:8080/stream.mjpg"

    # Specify ONNX model and CPU device
    python main.py --model best.onnx --device cpu --headless
"""

import sys
import argparse
from tel_vision import VisionConfig, VisionPipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TEL Robot Vision Computing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        default=None,
        help="Override video source (HTTP MJPEG URL or camera index, e.g. 0)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Override model weights path (.pt or .onnx)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        help="Inference device ('auto', 'cuda', 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="YOLO detection confidence threshold"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="HTTP POST telemetry destination endpoint URL"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode (no OpenCV GUI windows, suitable for RPi 5)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Force enable GUI display even if config has headless=true"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    config = VisionConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.source is not None:
        if args.source.isdigit():
            config.video.source = int(args.source)
        else:
            config.video.source = args.source

    if args.model is not None:
        config.model.weights = args.model

    if args.device is not None:
        config.model.device = args.device

    if args.conf is not None:
        config.model.conf_threshold = args.conf

    if args.endpoint is not None:
        config.http_sender.endpoint_url = args.endpoint

    if args.headless:
        config.system.headless = True
    elif args.gui:
        config.system.headless = False

    print("==================================================")
    print("   🤖 TEL ROBOT VISION COMPUTING SYSTEM v3.0      ")
    print("==================================================")
    print(f" Source   : {config.video.source}")
    print(f" Model    : {config.model.weights}")
    print(f" Device   : {config.model.device}")
    print(f" Mode     : {'HEADLESS (RPi 5 ready)' if config.system.headless else 'GUI PREVIEW (Tactical HUD)'}")
    print(f" HTTP POST: {'Enabled -> ' + config.http_sender.endpoint_url if config.http_sender.enabled else 'Disabled'}")
    print("==================================================")

    # Initialize and execute pipeline
    pipeline = VisionPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()

