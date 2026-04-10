import os
import sys
from openai import OpenAI
from env.email_env import EmailTriageEnv, Action
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    # In a real hackathon environment, this might be provided by the system
    # For now, we raise an error as per requirements if it's missing
    pass # We will check later in main to avoid early crash if imported

# Initialize OpenAI client
def get_client():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required")
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

def run_inference():
    # 1. Initialize the Environment (Local import as per plan)
    env = EmailTriageEnv()
    task_name = "email-triage"
    benchmark = "openenv-scalar"
    
    # [START] line
    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")

    try:
        client = get_client()
        observation = env.reset()
        step_num = 1
        done = False
        rewards = []

        while not done and observation:
            # Prepare prompt for the LLM
            prompt = f"""
            You are an email triage agent. Your task is to classify the following email.
            
            Sender: {observation.sender}
            Subject: {observation.subject}
            Body: {observation.body}
            
            Choose exactly one of the following actions:
            - mark_urgent (for high priority or security issues)
            - mark_spam (for unsolicited ads or suspicious offers)
            - reply (for personal or work inquiries that need a response)
            - ignore (for automated notifications or low-priority updates)
            
            Response format: Return only the action name.
            """

            # Call LLM
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            action_str = response.choices[0].message.content.strip().lower()
            
            # Sanitize action_str (ensure it's one of the valid actions)
            valid_actions = ["mark_urgent", "mark_spam", "reply", "ignore"]
            if action_str not in valid_actions:
                # Fallback or error handling
                if "urgent" in action_str: action_str = "mark_urgent"
                elif "spam" in action_str: action_str = "mark_spam"
                elif "reply" in action_str: action_str = "reply"
                else: action_str = "ignore"

            # Execute action in env
            action_obj = Action(action=action_str)
            observation = env.step(action_obj)
            reward = observation.reward
            done = observation.done
            
            rewards.append(reward)
            error_msg = "null"
            
            # [STEP] line
            # format reward to 2 decimal places, done as lowercase boolean
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
            
            step_num += 1

        # [END] line
        success = "true" if sum(rewards) > 0 else "false" # Simple success metric
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={success} steps={step_num-1} rewards={rewards_str}")

    except Exception as e:
        # Final [END] line must always be emitted even on exception
        print(f"[END] success=false steps=0 rewards=0.00 error={str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_inference()
