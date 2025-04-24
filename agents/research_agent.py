# agents/research_agent.py
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

def create_research_agent():
    """Creates the Research Agent using ChatOpenAI."""
    return Agent(
        role="Researcher",
        goal="Conduct thorough, focused research on a specific topic to gather information for content creation.",
        backstory="You are a diligent researcher skilled at finding relevant, accurate, and detailed information from reliable sources. You focus specifically on the query provided.",
        verbose=False,
        llm=llm # Use the configured ChatOpenAI instance
        # tools=[] # Add WebSearchTool here if needed for researcher
    )

def research_task(agent: Agent, idea: str, additional_context: str = None) -> Task:
    """Creates the research task for the Research Agent."""
    # (Task description remains the same as the previous version)
    query = idea
    if additional_context:
        query += f". Specifically focus on aspects related to: {additional_context}"

    return Task(
        description=f"""
        Conduct in-depth research on the following specific topic:
        '{query}'

        Gather relevant facts, statistics, examples, and key points that would be useful for creating content (like a blog post or newsletter) about this exact topic.
        Summarize the findings clearly and concisely. Ensure the information is accurate and directly related to the query.
        """,
        agent=agent,
        expected_output="A detailed research summary containing key facts, examples, and points relevant to the specific topic and context provided."
    )