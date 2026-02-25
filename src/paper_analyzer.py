"""
AI Agent for Research Paper Analysis
Based on the approach from The Web Scraping Club tutorial
"""

import asyncio
from dotenv import load_dotenv
import time
import os
import requests
from markitdown import MarkItDown
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent, AgentStream, ToolCallResult
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize the Google Gemini LLM
llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
)

# Use the current run's timestamp as unique filename
timestamp = int(time.time())

def download_and_convert_pdf(url: str) -> str:
    """
    Download a PDF from URL, save it locally, convert to Markdown,
    and return the Markdown content.
    """
    # Retrieve the PDF from the input URL
    response = requests.get(url)
    response.raise_for_status()

    # Create folder for downloaded papers
    papers_folder_path = "../data/papers"
    os.makedirs(papers_folder_path, exist_ok=True)

    # Export the PDF to the output folder
    output_path = os.path.join(papers_folder_path, f"{timestamp}.pdf")
    with open(output_path, "wb") as f:
        f.write(response.content)

    # Convert the PDF to Markdown
    md_converter = MarkItDown()
    result = md_converter.convert(output_path)

    return result.markdown

# Wrap the function as a LlamaIndex FunctionTool
download_and_convert_pdf_tool = FunctionTool.from_defaults(
    fn=download_and_convert_pdf,
    name="download_and_convert_pdf",
    description=(
        """
        Download a PDF from a URL, save it locally, convert it to Markdown,
        and return the Markdown content.
        """
    )
)

# System prompt for the agent
system_prompt = """
You are a research assistant specialized in analyzing academic papers.
Given a research paper PDF URL, you will access the paper's content in Markdown format
through a dedicated tool.

After processing the Markdown version of the paper, your goal is to produce a structured
Markdown report with the following sections:
- A brief summary of the paper's bibliographic information (title, subtitle, authors, publication date, etc.) presented in a table
- A detailed, well-structured summary covering the most important aspects of the paper, including quotes from the original text, plain-English explanations to make the content easier to understand, and links to authoritative sources for further reading
- A final list of concise key takeaways

When you need to access the paper content from its URL, always call the "download_and_convert_pdf" tool with the URL of the paper.
"""

# Define the structured output format
class Report(BaseModel):
    report: str = Field(description="The Markdown version of the report")

# Create the ReAct agent
agent = ReActAgent(
    llm=llm,
    tools=[download_and_convert_pdf_tool],
    system_prompt=system_prompt,
    output_cls=Report
)

async def main():
    # Read PDF URL from user input
    paper_pdf_url = input("Enter the URL of the research paper PDF: ").strip()

    # Run the agent
    handler = agent.run(
        f"""
        Using the given tools, download and analyze the academic paper at this URL:
        {paper_pdf_url}
        """
    )

    # Stream intermediate events for logging
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
        elif isinstance(ev, ToolCallResult):
            print(f"\n[Tool] {ev.tool_name}({ev.tool_kwargs}) → Tool completed.\n")

    # Wait for the final response
    response = await handler
    report_md = response.structured_response["report"]

    # Ensure the output folder exists
    output_folder = "../data/reports"
    os.makedirs(output_folder, exist_ok=True)

    # Save Markdown report to file
    output_path = os.path.join(output_folder, f"{timestamp}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nMarkdown report saved to: '{output_path}'")

if __name__ == "__main__":
    asyncio.run(main())
