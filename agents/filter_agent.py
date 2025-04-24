# agents/filter_agent.py
from crewai import Agent, Task
# --- Use ChatOpenAI ---
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Instantiate ChatOpenAI with Concurrency Limit ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", # Or your preferred model
    openai_api_key=openai_api_key,
    temperature=0.7,
    max_concurrency=10 # Limit concurrent requests for this LLM instance
)

def create_filter_agent():
    """Creates the Filter Agent using ChatOpenAI."""
    return Agent(
        role="Idea Filter",
        goal="Filter and rank ideas based on relevance and feasibility, outputting the result strictly in JSON format.",
        backstory="You are a precise analytical agent who evaluates ideas and outputs results in structured JSON format. You never include any text outside the JSON object.",
        verbose=False,
        llm=llm, # Use the configured ChatOpenAI instance
        allow_delegation=False,
        max_iterations=1
    )

def filter_ideas_task(agent: Agent, ideas: list, niche: str, target_audience: str, keywords: list) -> Task:
    """Creates the idea filtering task for the Filter Agent."""
    # (Task description remains the same as the previous version)
    if isinstance(ideas, list):
        ideas_str = "\n".join([f"- {idea}" for idea in ideas])
    else:
        ideas_str = str(ideas)

    return Task(
        description=f"""
        **System Instruction**: You are a JSON output generator. Your response must be ONLY a valid JSON string. Start directly with {{ and end directly with }}. No other text, commentary, or explanations.

        **Task**: Filter and rank the following ideas for a '{niche}' focused content piece targeting '{target_audience}' using keywords: {', '.join(keywords)}.

        **Input Ideas**:
        {ideas_str}

        **Steps**:
        1. Evaluate each provided idea based on: relevance to niche, alignment with target audience, effective use of keywords, and general feasibility/creativity.
        2. Select the top 3 most promising ideas based on your evaluation. If fewer than 3 ideas are provided or suitable, select fewer.
        3. Assign each selected idea a numerical score (float between 0.0 and 1.0, where 1.0 is best).
        4. Provide brief reasoning (1 concise sentence) for each idea's score/ranking.
        5. Output ONLY the JSON string with the following structure: {{"Idea": [list of strings], "Score": [list of floats], "Reasoning": [list of strings]}}. The lists must correspond by index and have the same length.
        6. If absolutely no ideas are suitable or provided, output the empty structure: {{"Idea": [], "Score": [], "Reasoning": []}}.

        **Important**:
        - Use the exact idea text from the input in the "Idea" list. Do not rephrase.
        - Ensure all JSON keys and string values are enclosed in double quotes.
        - The final output MUST be only the JSON string.

        **Example**: {{"Idea": ["Idea A", "Idea B"], "Score": [0.9, 0.7], "Reasoning": ["Highly relevant and uses keywords well.", "Good potential but less keyword focus."]}}
        """,
        agent=agent,
        expected_output='A single JSON string containing "Idea", "Score", and "Reasoning" lists.'
    )