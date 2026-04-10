---
title: OpenEnv Scalar
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv Scalar - Email Triage Challenge

This project is a submission for the Meta OpenEnv RL Challenge Hackathon.

## Goal
The goal is to build an AI agent that can correctly triage emails into categories: `mark_urgent`, `mark_spam`, `ignore`, and `reply`.

## Project Structure
- `inference.py`: entry point for the agent evaluation.
- `server/app.py`: FastAPI server for the environment.
- `env/email_env.py`: Logic for the Email Triage environment.
- `Dockerfile`: Configuration for deployment on Hugging Face Spaces.

## Requirements
- API_BASE_URL (default: https://api.openai.com/v1)
- MODEL_NAME (default: gpt-4.1-mini)
- HF_TOKEN (Required)

## How to run locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `uvicorn server.app:app --reload --port 7860`
3. Run inference: `python inference.py`
