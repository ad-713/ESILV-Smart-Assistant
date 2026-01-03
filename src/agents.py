from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from rag import query_rag

# 1. Define the RAG Tool
class ESILVInfoTool(BaseTool):
    name: str = "ESILV Knowledge Base"
    description: str = "Useful for answering questions about ESILV using the internal knowledge base. Input should be a fully formed question."

    def _run(self, query: str) -> str:
        return query_rag(query)

rag_tool = ESILVInfoTool()

# 2. Define Agents
def create_agents():
    # Agent 1: Information Specialist
    info_agent = Agent(
        role='ESILV Information Specialist',
        goal='Provide accurate information about ESILV programs, admissions, and student life.',
        backstory="""You are an expert on ESILV (École Supérieure d'Ingénieurs Léonard de Vinci). 
        Your job is to answer inquiries from prospective students and parents using the official knowledge base.
        You always provide clear, helpful, and polite answers.""",
        verbose=True,
        allow_delegation=False,
        tools=[rag_tool],
        llm=LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434") # Ensure you have 'llama3' pulled in Ollama
    )

    # Agent 2: Enrollment Assistant
    enrollment_agent = Agent(
        role='Enrollment Coordinator',
        goal='Identify if a user is interested in applying and collect their basic contact info.',
        backstory="""You are responsible for student recruitment. 
        If a user expresses strong interest or asks about applying, your goal is to politely ask for their name and email 
        so the admissions team can follow up. You do not answer technical questions; you focus on the next steps.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
    )

    return info_agent, enrollment_agent

# 3. Define Tasks & Crew
def create_crew(user_question):
    info_agent, enrollment_agent = create_agents()

    # Task 1: Answer the question
    answer_task = Task(
        description=f"""Analyze the user's question: '{user_question}'.
        Use the 'ESILV Knowledge Base' tool to find relevant information.
        Synthesize the retrieved information into a clear and helpful response.
        If the information is not found in the knowledge base, honestly state that you don't know but suggest they visit the website.""",
        expected_output="A helpful and accurate answer to the user's question based on the retrieved context.",
        agent=info_agent
    )
    
    followup_task = Task(
        description=f"""Review the user's input: '{user_question}' and the provided answer.
        If the user seems very interested in applying, asking about deadlines, or fees, 
        append a polite message asking if they would like to leave their contact details (Name/Email) for an admissions counselor.
        If the query is purely informational, just pass the answer through without adding anything.""",
        expected_output="The final response to be shown to the user, potentially with a call to action.",
        agent=enrollment_agent,
        context=[answer_task] # This means it waits for the answer_task
    )

    crew = Crew(
        agents=[info_agent, enrollment_agent],
        tasks=[answer_task, followup_task],
        verbose=True,
        process=Process.sequential
    )

    return crew

def run_crew(question):
    crew = create_crew(question)
    result = crew.kickoff()
    return result
