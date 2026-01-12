# Data Camp Python smolagents 

# Task for the agent
task = f"""Analyze my monthly expense data by category. Calculate total spending per category, find my highest expense area, and suggest a realistic budget for next month. Use simple text format in your final answer. Here is my weekly expense data for the past four weeks:

{expense_data}
"""

# Execute the financial analysis
result = agent.run(task)

print("Personal finance analysis:\n")
print(result)

# https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery

# Import the CodeAgent class
from smolagents import CodeAgent

# Create a basic agent without tools
agent = CodeAgent(tools=[], model=model)

task = "I deposit $100 every month into an account that pays 5% annual interest, compounded monthly. Calculate the total balance after 10 years."

# Run the agent
result = agent.run(task)
print(result)

# Allow Tool Agent to Visit the Web
# Import the VisitWebpageTool class
from smolagents import VisitWebpageTool

# Create agent with web search capabilities
agent = CodeAgent(
    tools=[VisitWebpageTool()],
    model=model
)

task = "Find GBP to USD exchange rates and summarize how this rate has changed over the past 7 days. A good source is usually Wise."

# Run the agent with the task
result = agent.run(task)

# Creating an agent with Custom Tools

# Import the tool decorator
from smolagents import tool

# Create a tool with the @tool decorator
@tool
def generate_order_id(table_id: str, drink_name: str) -> str:
    """
    Generates a unique order ID for a café order.
    
    Args:
        table_id: The table's identifier (e.g. "T5")
        drink_name: Name of the drink (e.g. "Latte")
    
    Returns:
        A string in the format "{table_id}_{drink_name}_{YYYYMMDD_HHMM}"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    order_id = f"{table_id}_{drink_name}_{timestamp}"
    
    # Return the order ID
    return order_id
  
# Create a tool that receives the table_id as input
@tool
def lookup_orders(table_id: str) -> list[str]:
    """
    Retrieves the current drink orders for a café table.

    Args:
        table_id (str): The table's identifier (e.g., "T5").

    Returns:
        list[str]: A list of drink orders, each formatted like "Latte (Large)".
    """
    
    # Read the orders.csv file
    df = pd.read_csv('orders.csv')
    orders = df[df['table_id'] == table_id].apply(lambda row: f"{row['drink_name']} ({row['size']})", axis=1).tolist()
    
    # Return the table orders
    return orders
  
# Create a code agent with the lookup_orders and generate_order_id tools
agent = CodeAgent(
    tools=[lookup_orders, generate_order_id],
    model=model,
    # Authorize pandas import
    additional_authorized_imports=['pandas']
)

task = (
    "For table 5, list their current drink orders and generate a unique order ID for each one."
)

# Run the agent passing the task
result = agent.run(task)
print(result)

# Agentic Retrieval Augmented Generation (RAG)

class ApplianceSearchTool(Tool):
    name = "appliance_manual_search"
    description = "Search appliance manuals for maintenance and usage information"
    inputs = {"query": {"type": "string", "description": "Question about appliance operation"}}
    output_type = "string"

    # Pass the vector store into the constructor
    def __init__(self, vector_store, k=3):
        super().__init__()
        self.vector_store = vector_store
        self.k = k

    # Accept the query string as input to the forward method
    def forward(self, query):
        # Use self.k here to specify how many chunks to return
        docs = self.vector_store.similarity_search(query, k=self.k)
        return "\n\n".join(doc.page_content for doc in docs) or "No relevant manual sections found."

# Create appliance search tool
appliance_tool = ApplianceSearchTool(vector_store)

# Create AI assistant for appliance help
assistant = CodeAgent(
    tools=[appliance_tool],
    model=model,
    instructions="Help with appliance questions using manual information. Search multiple times if needed for complete answers.",
    verbosity_level=1,
    max_steps=6
)

result = assistant.run("If the AC isn’t cooling and shows error E1, what should I check and what’s the next step?")
print(result)

# Track Agent Steps, Monitor the Action
# You're building a smolagents assistant for a basketball coach who needs help analyzing PDF reports that contain player stats, scouting insights, and game strategies.

# The coach relies on the agent to answer questions like: "What defensive strategies should we run against their second unit?"

# But the coach doesn't just want answers — they want visibility into what the agent is doing behind the scenes.

# In this exercise, you'll write an action callback that runs every time the agent takes a step, such as calling a tool or using the model. This callback will:

# Show the number of steps,
# And if the agent has finished, display how many tokens were used.
# This will help the coach (and you!) monitor how the agent is progressing and how much work it's doing to reach a conclusion.
# Define an action callback that accepts the agent step and the agent

def action_callback(agent_step, agent):
    step_num = agent_step.step_number
    print(f"Step {step_num}: Analyzing basketball data!")
    
    # Check if the agent step includes token usage
    if agent_step.is_final_answer:
        total_tokens = agent_step.token_usage.total_tokens
        # Print how many tokens were used
        print(f"Analysis complete! Total tokens used: {total_tokens}")
        
# Complete the function signature by adding the agent_step parameter.
# Check whether the current step produced a final answer using the .is_final_answer attribute of agent_step.
# If it's the final answer, get the total number of tokens from total_tokens and print it.

# Game Time: Run the Full Agent
# Your basketball coaching agent is almost ready to hit the court. You've written a callback function to show the step number and token usage.

# Now it's time to assemble everything: you'll configure a CodeAgent that uses that callback.

# You already have access to:

# A basketball_tool that enables the agent to search scouting reports
# A model variable with your language model
# A callback function: action_callback
# Your goal is to finish wiring up the agent.

# Import ActionStep to register a callback for it
from smolagents import ActionStep

coach_agent = CodeAgent(
    tools=[basketball_tool],
    model=model,
    verbosity_level=0,
    # Register a callback that runs when an ActionStep is triggered
    step_callbacks={ActionStep: action_callback}
)

# Run a question through the agent
result = coach_agent.run("What defensive strategy should we use to stop their point guard who averages 25 points per game?")
print(result)

# Chapter 3
# Divide and Conquer: Creating Specialist Agents
# You're helping your younger sister apply to computer science programs. It's a lot to manage: researching schools, comparing programs, and writing strong essays.

# To make the process easier, you decide to build two specialized agents:

# One to research universities and their requirements
# One to help write compelling application essays
# You have access to the CodeAgent and WebSearchTool classes, as well as a pre-configured model.

# School research specialist
school_agent = CodeAgent(
    # Assign a list of tools the agent can use
    tools=[WebSearchTool()],
    model=model,
    # Set the agent's unique name identifier
    name="school_research_agent",
    description="Expert in researching universities, programs, and admission requirements"
)

# Essay writing specialist  
essay_agent = CodeAgent(
    tools=[WebSearchTool()],
    # Provide the model used to generate responses
    model=model,
    name="essay_writing_agent",
    # Write a short description of the agent's area of expertise
    description="Expert in crafting compelling college application essays and personal statements"
)

# Managing Agent Memory

# Don't Forget: Keeping Memory Between Calls
# You're planning a graduation trip and using a smolagents travel assistant to keep track of important dates and information.

# However, the agent forgets unless you explicitly preserve its memory. Your goal is to prevent the agent from resetting between questions so it can remember key travel details across multiple interactions.

# You have access to the travel_agent, already set up with a model and no tools.

# Step 1: Tell the agent your flight date
travel_agent.run("My Tokyo flight confirmation code is ZX9Q2L.")

# Step 2: Confirm the agent remember when passing the correct reset parameter
follow_up = "What’s my Tokyo flight confirmation code?"
response = travel_agent.run(follow_up, reset=False)

print(response)

# Tracing the Agent’s Code Execution
# You've preserved memory in your travel assistant agent, and it successfully remembered your flight confirmation number. Now, you're curious about how it computed or stored that information.

# To find out, you'll inspect the full code the agent executed across both steps using its built-in memory.

# Inspect the code executed by the agent
executed_code = travel_agent.memory.return_full_code()

print("Executed code during session:")
print("=" * 50)
print(executed_code)
