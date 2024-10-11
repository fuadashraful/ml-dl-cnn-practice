from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


if __name__ == "__main__":
    load_dotenv()
    print("Welcome to LLM playground !!!")

    info = """
William Henry Gates III (born October 28, 1955) is an American businessman best known for co-founding the software company Microsoft with his childhood friend Paul Allen. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president, and chief software architect, while also being its largest individual shareholder until May 2014.[a] He was a pioneer of the microcomputer revolution of the 1970s and 1980s.

Gates was born and raised in Seattle, Washington. In 1975, he and Allen founded Microsoft in Albuquerque, New Mexico. Gates led the company as its chairman and chief executive officer until stepping down as CEO in January 2000, succeeded by Steve Ballmer, but he remained chairman of the board of directors and became chief software architect. During the late 1990s, he was criticized for his business tactics, which were considered anti-competitive.
"""
    summery_template = """
    Given short information about a person I want you to create:
        1. A short summery
        2. Description about career
"""

    prompt_template = PromptTemplate(
        input_variables=["info"],
        template=summery_template
    )

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    chain = prompt_template | llm

    answer = chain.invoke(
        input = {"info": info}
    )

    print(answer)