# LlamaIndex-Based Immune Aging Analysis Agent

This branch implements a LlamaIndex ReAct agent that combines omics data analysis with RAG-powered literature search and web search.

## Architecture

**Single ReAct Agent** with access to:
1. **Omics Data Tools**: Functions to retrieve aging/intervention signatures from immune cell analysis
2. **Literature RAG**: Vector-indexed scientific papers for biological context
3. **Manuscript RAG**: Vector-indexed main manuscript document
4. **Web Search**: Internet search for current information (Tavily or DuckDuckGo)

## Setup

### 1. Create Conda Environment
```bash
conda create -n llamaindex python=3.10 -y
conda activate llamaindex
```

### 2. Install Dependencies
```bash
pip install llama-index llama-index-llms-openai python-dotenv tavily-python duckduckgo-search
```

### 3. Set API Keys
Create a `.env` file in the project root:
```bash
cp .env.template .env
# Edit .env and add:
# - OPENAI_API_KEY (required)
# - TAVILY_API_KEY (optional - only if using Tavily search)
```

**Web Search Options:**
- **Tavily** (default): AI-optimized search, requires API key ([get free key](https://tavily.com) - 1000 searches/month)
- **DuckDuckGo**: Free, no API key needed

To switch between providers, edit `src/config.py`:
```python
WEB_SEARCH_PROVIDER = "tavily"  # or "duckduckgo"
```

### 4. Prepare Literature
Place your PDF/txt papers in `data/literature/` folder.

## Usage

### Interactive Mode
```bash
bash scripts/run.sh
```

### Single Query
```bash
bash scripts/run.sh "What changes occur in CD8T cells with aging?"
```

### From Python
```python
from src.agent import initialize_agent
import asyncio

async def main():
    agent = initialize_agent()
    handler = agent.run("Your question here")
    response = await handler
    print(response)

asyncio.run(main())
```

## How It Works

1. **User asks a question**
2. **Agent reasons** (ReAct pattern):
   - Determines what tools to call
   - Validates inputs using metadata tools
   - Retrieves omics data as needed
   - Searches literature for context
   - Searches web for current information (if needed)
3. **Agent synthesizes** data + literature + web results into final answer

## Example Queries

**Omics Data:**
- "What transcription factors decrease with age in CD8T cells?"
- "Show me the effects of IL-2 on monocytes"

**Literature Search:**
- "What is known about BACH2 in immune aging?"
- "Find papers about senescence interventions"

**Web Search:**
- "What are the latest clinical trials for immune aging interventions?"
- "Recent research on senolytics in 2025"
- "Current therapies for immunosenescence"

**Combined Analysis:**
- "What transcription factors decrease with age in CD8T cells?"
- "Show me the effects of IL-2 on monocytes"

**Literature Search:**
- "What is known about BACH2 in immune aging?"
- "Find papers about senescence interventions"

**Combined Analysis:**
- "What goes wrong with CD8T aging and what does literature say?"
- "Analyze NK cell aging and suggest interventions"

## File Structure

```
src/
├── agent.py           # Main agent initialization with all tools
├── rag_builder.py     # RAG index building for literature/manuscript
├── helper.py          # Omics data retrieval functions
├── instructions.txt   # System prompt for the agent
├── run.py            # Interactive and single-query runner
└── config.py         # Configuration (model, data sources)

data/
├── literature/       # Place PDF/txt papers here
└── manuscrtipt.txt  # Main manuscript

.index_cache/         # Cached vector indices (auto-generated)
```

## Key Differences from Agent Framework Version

| Feature | Agent Framework (main branch) | LlamaIndex (this branch) |
|---------|------------------------------|--------------------------|
| Architecture | Two separate agents (omics + context) | Single ReAct agent |
| Literature | Loaded into system prompt | RAG with vector search |
| Scalability | Limited by context window | Handles 100s of papers |
| Tool Calling | Basic function calling | ReAct reasoning pattern |
| Output Format | Prompt engineering | Can use structured output |
| Document Types | Text only | PDF, txt, md, etc. |

## Customization

### Change LLM Model
Edit `src/run.py` or pass to `initialize_agent()`:
```python
agent = initialize_agent(model_id="gpt-3.5-turbo")
```

### Modify Instructions
Edit `src/instructions.txt` directly - changes apply immediately.

### Add New Omics Tools
1. Add function to `src/helper.py` (no decorator needed)
2. Add wrapper in `create_omics_tools()` in `src/agent.py`

### Clear Index Cache
```bash
rm -rf .index_cache
```
Next run will rebuild indices from scratch.

## Troubleshooting

**"Import could not be resolved" errors in VS Code:**
- These are expected if you're not in the llamaindex conda environment
- The code will run fine when executed via the script

**Slow first run:**
- Building indices takes time initially
- Subsequent runs use cached indices (much faster)

**Out of memory:**
- Reduce `similarity_top_k` in `create_rag_tools()` in `agent.py`
- Use a smaller model (gpt-3.5-turbo instead of gpt-4o)
