from dotenv import load_dotenv
from crewai import Agent, LLM
import google.generativeai as genai
import os 

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')  

llm = LLM(
    model="gemini/gemini-2.0-flash", 
    api_key=api_key,
    temperature=0.3
)

csv_reader_agent = Agent(
    role="Data Reader",
    goal="Load and clean feedback data from CSV files",
    backstory="""You are responsible for loading user feedback data from app store reviews and support emails. 
    You ensure the data is clean and properly formatted for analysis.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

feedback_classifier_agent = Agent(
    role="Feedback Classifier",
    goal="Classify user feedback into meaningful categories",
    backstory="""You are an expert at understanding user feedback and categorizing it correctly. 
    You can distinguish between bugs, feature requests, praise, complaints, and spam with high accuracy.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

bug_analysis_agent = Agent(
    role="Bug Analyst",
    goal="Analyze bug reports and assign appropriate severity levels",
    backstory="""You are a technical analyst specializing in bug triage. You can extract technical details 
    from user reports and assign severity levels based on the impact and urgency of issues.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

feature_extractor_agent = Agent(
    role="Feature Extractor",
    goal="Identify and summarize feature requests from user feedback",
    backstory="""You are a product analyst who excels at understanding user needs and translating 
    feedback into actionable feature requests that can guide product development.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

ticket_generator_agent = Agent(
    role="Ticket Generator",
    goal="Create structured tickets from processed feedback",
    backstory="""You are responsible for converting analyzed feedback into well-structured tickets 
    that development and support teams can easily understand and act upon.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


quality_critic_agent = Agent(
    role="Quality Reviewer",
    goal="Ensure ticket quality and flag issues",
    backstory="""You are a quality assurance specialist who reviews generated tickets to ensure 
    they meet standards and flag any that need human review or additional processing.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)