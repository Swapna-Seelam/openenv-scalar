import os
import sys
import json
import logging
from typing import Optional, List
from openai import OpenAI
from env.email_env import EmailTriageEnv, Action
from dotenv import load_dotenv

# Configure basic logging to stdout for visibility in logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)

# Load environment variables from .env file for local development
load_dotenv()

# Read environment variables with safe defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_client() -> Optional[OpenAI]:
    """Safely initialize the OpenAI client."""
    try:
        if not HF_TOKEN:
            logging.warning("HF_TOKEN is missing. API calls will fail, falling back to default actions.")
            return None
        return OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return None

def get_safe_attribute(obj, attr_name, default="N/A"):
    """Safely get an attribute from an object or dict."""
    if obj is None:
        return default
    try:
        if hasattr(obj, attr_name):
            val = getattr(obj, attr_name)
            return val if val is not None else default
        if isinstance(obj, dict):
            return obj.get(attr_name, default)
    except Exception:
        pass
    return default

def run_inference():
    # 1. Initialize the Environment
    try:
        env = EmailTriageEnv()
    except Exception as e:
        logging.error(f"Failed to initialize environment: {e}")
        print(f"[START] task=email-triage env=openenv-scalar model={MODEL_NAME}")
        print(f"[END] success=false steps=0 rewards=0.00 error=env_init_failed")
        return

    task_name = "email-triage"
    benchmark = "openenv-scalar"
    
    # [START] line
    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")

    step_num = 1
    done = False
    rewards = []
    
    try:
        client = get_client()
        observation = env.reset()
        
        # Max steps to prevent infinite loops
        max_steps = 100
        
        while not done and observation and step_num <= max_steps:
            # Safely extract data from observation
            sender = get_safe_attribute(observation, "sender", "unknown")
            subject = get_safe_attribute(observation, "subject", "no subject")
            body = get_safe_attribute(observation, "body", "empty body")

            action_str = "ignore" # Default fallback action
            error_msg = "null"

            if client:
                try:
                    # Prepare prompt for the LLM
                    prompt = f"""
                    You are an email triage agent. Your task is to classify the following email.
                    
                    Sender: {sender}
                    Subject: {subject}
                    Body: {body}
                    
                    Choose exactly one of the following actions:
                    - mark_urgent (for high priority or security issues)
                    - mark_spam (for unsolicited ads or suspicious offers)
                    - reply (for personal or work inquiries that need a response)
                    - ignore (for automated notifications or low-priority updates)
                    
                    Response format: Return only the action name.
                    """

                    # Call LLM with timeout to prevent hanging
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=30.0
                    )
                    
                    if response and response.choices:
                        raw_content = response.choices[0].message.content
                        if raw_content:
                            action_str = raw_content.strip().lower()
                except Exception as api_err:
                    logging.error(f"API call failed at step {step_num}: {api_err}")
                    error_msg = f"api_failure_{type(api_err).__name__}"
            else:
                error_msg = "missing_hf_token"

            # Sanitize action_str (ensure it's one of the valid actions)
            valid_actions = ["mark_urgent", "mark_spam", "reply", "ignore"]
            if action_str not in valid_actions:
                if "urgent" in action_str: action_str = "mark_urgent"
                elif "spam" in action_str: action_str = "mark_spam"
                elif "reply" in action_str: action_str = "reply"
                else: action_str = "ignore"

            # Execute action in env safely
            try:
                action_obj = Action(action=action_str)
                observation = env.step(action_obj)
                
                # Fetch reward and done from observation safely
                reward = float(get_safe_attribute(observation, "reward", 0.0))
                done = bool(get_safe_attribute(observation, "done", True))
            except Exception as step_err:
                logging.error(f"Environment step failed: {step_err}")
                reward = 0.0
                done = True
                error_msg = "env_step_failed"
            
            rewards.append(reward)
            
            # [STEP] line: format reward to 2 decimal places, done as lowercase boolean
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
            
            step_num += 1

        # [END] line
        success = "true" if sum(rewards) > 0 else "false"
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else "0.00"
        print(f"[END] success={success} steps={len(rewards)} rewards={rewards_str}")

    except Exception as e:
        logging.error(f"Unexpected error in inference loop: {e}")
        # Always emit [END] line even on catastrophic failure
        print(f"[END] success=false steps={len(rewards)} rewards=0.00 error=unhandled_exception")

def main():
    """Main entry point that ensures zero exit code."""
    try:
        run_inference()
    except Exception as e:
        logging.critical(f"Critical failure in main: {e}")
    finally:
        # OpenEnv requirements: exit code must be 0 to pass validation
        sys.exit(0)

if __name__ == "__main__":
    main()
