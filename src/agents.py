from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from rag import query_rag
from leads_manager import save_lead

# 1. Define the RAG Tool
class ESILVInfoTool(BaseTool):
    name: str = "ESILV Knowledge Base"
    description: str = "Useful for answering questions about ESILV using the internal knowledge base. Input should be a fully formed question."

    def _run(self, query: str) -> str:
        return query_rag(query)

class SaveLeadTool(BaseTool):
    name: str = "Save Lead Contact Info"
    description: str = "Useful for saving user contact details when they provide them. Requires 'name' and 'email'. 'topic' is optional."

    def _run(self, name: str, email: str, topic: str = "General Inquiry") -> str:
        # Strict validation against hallucinations
        lower_name = name.lower()
        lower_email = email.lower()
        
        hallucination_indicators = [
            "john doe", "jane doe", "your name", "your email", 
            "placeholder", "example.com", "fake@email.com", "test@test.com"
        ]
        
        # Check for common placeholders
        if any(indicator in lower_name for indicator in hallucination_indicators):
             return f"Error: Invalid name '{name}'. It looks like a placeholder. Please ask the user for their real name."
             
        if any(indicator in lower_email for indicator in hallucination_indicators):
             return f"Error: Invalid email '{email}'. It looks like a placeholder. Please ask the user for their real email."
             
        # Check for generic single-word placeholders if they match exactly
        generic_placeholders = ["name", "email", "user", "none", "unknown", "fake", "test"]
        if lower_name in generic_placeholders:
             return f"Error: Invalid name '{name}'. Please provide a real name."
        if lower_email in generic_placeholders:
             return f"Error: Invalid email '{email}'. Please provide a real email."
             
        if "@" not in email or "." not in email:
             return f"Error: Invalid email format '{email}'. Please ask the user for a valid email address."

        try:
            return save_lead(name, email, topic)
        except Exception as e:
            return f"Error saving lead: {e}"

rag_tool = ESILVInfoTool()
save_lead_tool = SaveLeadTool()

# 2. Define Agents
def create_agents():
    # Agent 1: Information Specialist
    info_agent = Agent(
        role='ESILV Information Specialist',
        goal='Provide accurate information about ESILV programs, admissions, and student life.',
        backstory="""You are an expert on ESILV (École Supérieure d'Ingénieurs Léonard de Vinci). 
        Your job is to answer inquiries from prospective students and parents using the official knowledge base.
        
        RULES:
        1. You always provide clear, helpful, and polite answers.
        2. STRICTLY BASE YOUR ANSWERS ON THE KNOWLEDGE BASE. 
        3. If the information is not found in the knowledge base, honestly state that you don't know but suggest they visit the website.
        4. DO NOT make up information (Hallucination is strictly forbidden).""",
        verbose=True,
        allow_delegation=False,
        tools=[rag_tool],
        llm=LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
    )

    # Agent 2: The Salesman
    sales_agent = Agent(
        role='The Salesman',
        goal='Analyze the user sentiment and if interested, politely offer a follow-up.',
        backstory="""You are the friendly front-desk representative at ESILV.
        Your goal is to gauge the user's interest level based on the conversation.
        
        RULES:
        1. You do not invent information.
        2. You analyze the Info Specialist's answer and the User's question.
        3. If the user seems interested in applying, programs, or specific details, you add a polite closing asking if they want an advisor to contact them.
        4. Use the EXACT phrase: "Would you like me to have an advisor contact you? Just leave your name and email." if they are interested.
        5. If the user is NOT interested, or the Info Agent said "I don't know", you do NOT add the phrase. You just pass the Info Agent's answer through.""",
        verbose=True,
        allow_delegation=False,
        tools=[], # No tools, just text processing
        llm=LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
    )

    # Agent 3: The Clerk (Silent Background Process)
    clerk_agent = Agent(
        role='The Clerk',
        goal='Silently extract and save contact information if provided.',
        backstory="""You are a silent data entry clerk.
        Your ONLY job is to look at the user's messages.
        
        RULES:
        1. You NEVER speak to the user.
        2. You ONLY trigger the 'Save Lead Contact Info' tool if you see a valid Name AND Email.
        3. You ignore placeholders like "John Doe" or "example.com".
        4. If no contact info is present, you do nothing and return "No lead found".
        5. You are invisible to the user.""",
        verbose=True,
        allow_delegation=False,
        tools=[save_lead_tool],
        llm=LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
    )

    return info_agent, sales_agent, clerk_agent

# 3. Define Tasks & Crew
def create_crew(user_question):
    info_agent, sales_agent, _ = create_agents()

    # Task 1: Answer the question
    answer_task = Task(
        description=f"""Analyze the user's question: '{user_question}'.
        Use the 'ESILV Knowledge Base' tool to find relevant information.
        Synthesize the retrieved information into a clear and helpful response.
        If the information is not found in the knowledge base, honestly state that you don't know.""",
        expected_output="A helpful and accurate answer to the user's question based on the retrieved context.",
        agent=info_agent
    )
    
    # Task 2: Sales Polish (Append Call to Action)
    sales_task = Task(
        description=f"""Review the User's Question ('{user_question}') and the Answer provided by the Info Agent.
        
        1. If the Info Agent's answer indicates they found relevant information AND the user seems interested (e.g., asking about programs, prices, admissions, campus life):
           - Append the following text to the answer: "\n\nWould you like me to have an advisor contact you? Just leave your name and email."
           
        2. If the Info Agent said "I don't know" or the topic is negative/unrelated:
           - Do NOT append anything. Return the Info Agent's answer exactly as is.
           
        3. Ensure the final output is a cohesive response.""",
        expected_output="The final text to be shown to the user.",
        context=[answer_task], # Pass the output of answer_task to this task
        agent=sales_agent
    )

    crew = Crew(
        agents=[info_agent, sales_agent],
        tasks=[answer_task, sales_task],
        verbose=True,
        process=Process.sequential
    )

    return crew

def create_lead_capture_crew(user_question, agent_answer):
    _, _, clerk_agent = create_agents()

    lead_task = Task(
        description=f"""Analyze the interaction below to determine if the user has voluntarily provided their Name and Email address.

        User Question: "{user_question}"
        Agent Answer: "{agent_answer}"
        
        INSTRUCTIONS:
        1. Look for EXPLICIT contact details (Name AND Email) in the User Question.
        2. If BOTH name and email are present, use the 'Save Lead Contact Info' tool.
        3. If NO contact details are present, do NOT use any tool. Just return the text "No lead found".
        4. Do NOT ask for more information. This is a silent background process.
        
        CRITICAL: 
        - IGNORE "John Doe", "Jane Doe", "example.com" or other placeholders.
        - Only save if it looks like a REAL user provided their details.
        """,
        expected_output="Status message indicating if a lead was saved or not.",
        agent=clerk_agent
    )

    crew = Crew(
        agents=[clerk_agent],
        tasks=[lead_task],
        verbose=True,
        process=Process.sequential,
        max_rpm=10,
        cache=False
    )
    
    return crew

def run_crew(question):
    # Phase 1: Get Information & Sales Pitch
    main_crew = create_crew(question)
    final_answer = main_crew.kickoff()
    
    # Phase 2: Check for leads (Silent / Background)
    # We pass the answer too, just in case context is needed, though usually lead is in the question
    try:
        lead_crew = create_lead_capture_crew(question, str(final_answer))
        lead_crew.kickoff()
    except Exception as e:
        print(f"Lead capture background process failed (non-critical): {e}")

    return str(final_answer)
