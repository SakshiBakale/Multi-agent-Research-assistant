
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
import fitz
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

gr_api_key=os.environ['GROQ_API_KEY']

llm = ChatGroq(model="llama3-8b-8192", api_key=gr_api_key)


summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="Summarize the following academic paper into key arguments, methodologies, and conclusions:\n\n{content}"
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt)


critique_prompt = PromptTemplate(
    input_variables=["summary", "related_content"],
    template=(
        "Critique the following summary based on the additional content:\nSummary: {summary}\n\nRelated Content: {related_content}\n"
        "Identify discrepancies, missing insights, and provide strengths and weaknesses."
    )
)

critique_chain = LLMChain(llm=llm, prompt=critique_prompt)


refinement_prompt = PromptTemplate(
    input_variables=["summary", "critique"],
    template=(
        "Refine the following summary using the critique provided:\nSummary: {summary}\nCritique: {critique}\n"
        "Ensure clarity, logical flow, and add citations where applicable."
    )
)

refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt)


tools = [
    Tool(
        name="SummarizePaper",
        func=summary_chain.run,
        description="Use this tool to summarize academic papers into key arguments, methodologies, and conclusions."
    ),
    Tool(
        name="CritiquePaper",
        func=critique_chain.run,
        description="Use this tool to critique summaries based on additional related content."
    ),
    Tool(
        name="RefineSummary",
        func=refinement_chain.run,
        description="Use this tool to refine a summary based on critique feedback."
    )
]


multi_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


def load_document(file_path: str):
    if file_path.endswith('.pdf'):

        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    else:
        with open(file_path, 'r') as file:
            paper_text=file.read()
            return paper_text


def split_text_into_chunks(text, max_length=3000):
    chunks = []
    while len(text) > max_length:
        chunk = text[:max_length]
        chunks.append(chunk)
        text = text[max_length:]
    if text:
        chunks.append(text)
    return chunks


def chunk_text(text, max_length=1000):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_length])
        start += max_length
    return chunks


def orchestrate_workflow(paper_text):

    paper_chunks = split_text_into_chunks(paper_text)

    summaries = []
    for chunk in paper_chunks:
        summary = summary_chain.run({"content": chunk})
        summaries.append(summary)
        print(f"Summary: {summary}")

    full_summary = " ".join(summaries)
    related_content = "Additional related content that can help with critique."

    critique = critique_chain.run({"summary": full_summary, "related_content": related_content})
    print(f"Critique: {critique}")


    refined_summary = refinement_chain.run({"summary": full_summary, "critique": critique})
    print(f"Refined Summary: {refined_summary}")

def main():
    paper_file = "./Research_paper.pdf"
    paper_content = load_document(paper_file)
    orchestrate_workflow(paper_content)

if __name__ == "__main__":
    main()
