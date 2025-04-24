🌊 MindFlow – AI-Powered Content Generation Pipeline 🧠✨

[![Streamlit](https://img.shields.io/badge/streamlit-%E2%9C%93-blue)](https://streamlit.io) [![Python](https://img.shields.io/badge/python-3.8%2B-orange)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> MindFlow orchestrates a symphony of AI agents to take your niche insights and keywords, then spin up trending, researched, and validated content—automatically.

---

📋 Table of Contents
1. [Key Features](#-key-features-rocket)
2. [Project Structure](#-project-structure)
3. [Setup & Installation](#-setup--installation)
4. [Usage](#-usage-️)
5. [Pipeline Workflow](#-pipeline-workflow-🔄)
6. [Configuration & Customization](#-configuration--customization-⚙️)
7. [Contributing](#-contributing)
8. [License](#-license)

---

## ✨ Key Features 🚀

- **Multi-Agent Architecture**: Specialized AI roles (Idea Generator, Filter, Researcher, Writer, Validator) via CrewAI & LangChain.
- **Trend-Driven Ideas**: DuckDuckGo search of Medium.com surfaces the freshest, most viral topics.
- **Structured Filtering**: Auto-ranks ideas by relevance, feasibility, and keyword alignment with JSON output.
- **Deep Research**: Stash and reuse research summaries to avoid redundant calls.
- **Iterative Validation Loop**: Boss Agent reviews; Writer Agent refines until ✅ or max revisions.
- **Resilient API Handling**:
  - `max_concurrency` throttles parallel prompts.
  - `tenacity` retries transient failures.
  - LLM caching (InMemory or SQLite) for speed.
- **Live Streamlit UI**:
  - `st.status()` panels for real-time agent feedback.
  - Progress bar and status badges for each pipeline stage.
  - Sleek dark theme, custom fonts, and CSS animations.

---

🗂️ Project Structure

```
mindflow/
├── agents/                # CrewAI agent definitions & tasks
├── vectorstore/           # ChromaDB setup (optional)
├── assets/                # Static assets (images, demo GIF)
├── .env                   # API keys & secrets
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

🔧 Setup & Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/mindflow.git
   cd mindflow
   ```
2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   .\venv\Scripts\activate  # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure your `.env`**
   ```env
   OPENAI_API_KEY=your_openai_key_here
   ```
5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

▶️ Usage

1. **Enter** your niche/industry, select keywords, tone, length, and audience in the sidebar.
2. **Click** "Start Pipeline".
3. **Watch** each agent light up:
   - 💡 Idea Agent → 🧹 Filter Agent → 🔍 Research Agent → ✍️ Writer Agent → 🕵️ Boss Agent
4. **Interact** with feedback loops until your draft is approved.
5. **Export** the final approved draft.

---

🔄 Pipeline Workflow

| Step | Agent           | Description                              |
|------|-----------------|------------------------------------------|
| 1    | Idea Generator  | Scrapes Medium trends, brainstorms ideas  |
| 2    | Filter          | Ranks & selects top ideas                |
| 3    | Research        | Gathers facts & stats, caches results    |
| 4    | Writer          | Drafts content, structures flow          |
| 5    | Validator       | Provides JSON feedback & approval loop   |

---

⚙️ Configuration & Customization

- **Models & Concurrency**: In `agents/*.py`, adjust `ChatOpenAI(model_name, max_concurrency=…)`.
- **Max Revisions**: Modify `max_revisions` in `app.py` default state.
- **Cache Backend**: Swap `InMemoryCache` for `SQLiteCache` in `app.py` for persistent caching.
- **CSS & Theme**: Tweak the `<style>` block in `app.py` for fonts, colors, and animations.
- **Task Prompts**: Edit `role`, `goal`, `backstory` and prompt `description` in each agent file.

---

🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AwesomeFeature`)
3. Commit your changes (`git commit -m 'Add AwesomeFeature'`)
4. Push to branch (`git push origin feature/AwesomeFeature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

📝 License

[MIT LICENSE] (LICENSE)

---

Made with ❤️ by AB

---

_*MindFlow — where data-driven AI meets creative content.*_

