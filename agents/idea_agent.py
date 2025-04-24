# agents/idea_agent.py
from crewai import Agent, Task
# --- Use ChatOpenAI ---
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
import os
from dotenv import load_dotenv
import re

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key: raise ValueError("OPENAI_API_KEY not found")

# --- Instantiate ChatOpenAI with Concurrency Limit ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", # Or your preferred model
    openai_api_key=openai_api_key,
    temperature=0.7,
    max_concurrency=10 # Limit concurrent requests for this LLM instance
)

# --- Define the Custom Tool Wrapper (remains the same) ---
class WebSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = "Search the web for information, trends, articles. Input is a search query."
    def _run(self, query: str) -> str:
        duckduckgo_search = DuckDuckGoSearchRun()
        try:
            print(f"Executing search for: {query}")
            response = duckduckgo_search.invoke(query)
            print(f"Search response length: {len(response)}")
            max_length = 2000 # Truncate long search results
            if len(response) > max_length: response = response[:max_length] + "... (truncated)"
            return response
        except Exception as e: print(f"Search failed for '{query}': {e}"); return f"Error: {e}"

# --- Define the Idea Agent using the Custom Tool ---
def create_idea_agent():
    """Creates an Idea Agent using ChatOpenAI."""
    return Agent(
        role="Trend Analyst and Idea Generator",
        goal=(
            "Analyze current trends on Medium related to a specific niche and keywords, "
            "focusing on popular and recent articles. Generate creative content ideas "
            "inspired by these trends, tailored to the specified content type."
        ),
        backstory=(
            "You are an expert market researcher and content strategist. You excel at "
            "identifying buzzing topics and successful content formats by analyzing online platforms "
            "like Medium. You use your search tool to find relevant articles, "
            "synthesize trends, and then brainstorm original content ideas."
        ),
        verbose=False,
        llm=llm, # Use the configured ChatOpenAI instance
        tools=[WebSearchTool()],
        allow_delegation=False
    )

# --- Define the Idea Generation Task (Original objective, asking for 7 ideas) ---
def idea_generation_task(agent: Agent, niche: str, content_type: str, target_audience: str, content_tone: str, keywords: list[str], num_ideas: int = 7) -> Task: # Default to 7 ideas
    """Creates a task for the Idea Agent to generate a specific number of ideas based on trends."""
    keyword_string = ", ".join(keywords)
    search_query_hint = f"recent popular articles {niche} {keyword_string} site:medium.com"

    return Task(
        description=f"""
        **Objective**: Generate exactly {num_ideas} distinct content ideas for '{content_type}' based on current trends observed on Medium.

        **Inputs**:
        - Niche: {niche}
        - Keywords: {keyword_string}
        - Content Type: {content_type}
        - Target Audience: {target_audience}
        - Content Tone: {content_tone}

        **Instructions**:
        1.  **Use 'DuckDuckGo Web Search' tool** to find RECENT and POPULAR articles on Medium related to niche '{niche}' and keywords '{keyword_string}'. Search within `medium.com`. (Hint: Query like '{search_query_hint}')
        2.  **Analyze search results**: Identify recurring themes, popular angles from search result summaries/titles.
        3.  **Identify Trends**: Briefly synthesize 1-2 key trends from the search results.
        4.  **Generate Ideas**: Based primarily on trends, brainstorm **exactly {num_ideas} distinct and creative content ideas** for a '{content_type}' format.
        5.  **Tailor Ideas**: Ensure ideas fit '{target_audience}' and '{content_tone}' tone. Make them actionable.
        6.  **Output Format**: Present ONLY the {num_ideas} ideas as a numbered list (1. Idea one, 2. Idea two, ...). No other text.

        **Example Output (for num_ideas=5)**:
        1. Idea One Title
        2. Idea Two Title
        3. Idea Three Title
        4. Idea Four Title
        5. Idea Five Title
        """,
        agent=agent,
        expected_output=f"A numbered list containing exactly {num_ideas} creative content ideas."
    )
# --- END Task Definition ---