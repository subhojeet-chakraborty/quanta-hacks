import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    template = """
    You are an AI data analyst who manages a Home Inventory. You are interacting with a user who is asking you questions about the Home Inventory's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    When asked 'show items present', display all the items with the quantity present and do not generate a food recipe or anything else.
    When asked for 'shopping list' create a list items from the table whose value is less than 3 , if there
    are none then say 'you don't need anything currently' and do not give me anything except list of items needed.
    Also, generate a food recipe from the food items present in Inventory only when asked 'generate recipe' and do not display food recipe for any other query.
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks. Do not respond with 'based on your SQL query'.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatOpenAI(model="gpt-4-0125-preview")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def run_query_with_commit(db, query):
    conn = db.engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are an AI data analyst who manages Home Inventory. You are interacting with a user who is asking you questions about the Home Inventory's database.
    Based on the table schema below, question, SQL query, and SQL response, write a natural language response.
    
    When asked 'show items present', just display the items with the quantity.
    When asked for 'shopping list' create a list items from the table whose value is less than 3 , if there
    are none then say 'you don't need anything currently' and do not give me anything except list of items needed.
    Also, generate a food recipe from the food items present in HomeInventory only when asked 'generate recipe' and do not display food recipe for any other query..
    
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatOpenAI(model="gpt-4-0125-preview")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })


def generate_recipe_from_inventory(db: SQLDatabase) -> str:
    # Update the table name here if necessary
    available_items_query = "SELECT Items FROM Inventory WHERE Quantity > 0"
    available_items = db.run(available_items_query)
    
    if not available_items:
        return "No items available in the inventory."
    
    # Convert the query result into a list of item names
    available_items = [item[0] for item in available_items]
    
    # Define some example recipes based on the available items
    recipes = {
        ("apple", "banana"): "Fruit Salad",
        ("chicken", "potato", "carrot"): "Chicken Stew",
        ("pasta", "tomato", "cheese"): "Pasta with Tomato Sauce",
        # Add more recipes as needed
    }
    
    # Check if any of the defined recipes match the available items
    for ingredients, recipe_name in recipes.items():
        if all(item in available_items for item in ingredients):
            return f"Recipe: {recipe_name}\nIngredients: {', '.join(ingredients)}"
    
    # If no matching recipe found
    return "No recipe found with the available items."


def handle_special_queries(user_query: str, db: SQLDatabase) -> str:
    greetings = {
        "hi": "Hi I am Yokie, your inventory chatbot. How can I help you?",
        "hello": "Hi I am Yokie, your inventory chatbot. How can I help you?",
        "hey": "Hi I am Yokie, your inventory chatbot. How can I help you?",
        "good morning": "Good morning!",
        "good afternoon": "Good afternoon!",
        "good night": "Good night!"
    }
    farewells = ["bye", "goodbye", "see you", "later", "quit"]
    about_bot = ["who are you", "what are you", "what do you do"]
    thanks = ["thanks", "thank you"]
    generic_queries = ["how are you", "what's up", "how's it going", "do you love me", "do you hate me", "i love you"]

    user_query_lower = user_query.lower()

    if user_query_lower in greetings:
        return greetings[user_query_lower]
    elif user_query_lower in farewells:
        return "Goodbye! Have a great day!"
    elif user_query_lower in about_bot:
        return "I am an AI assistant here to help you manage your home inventory. Ask me anything about your inventory."
    elif user_query_lower in thanks:
        return "You're welcome! If you have any more questions, feel free to ask."
    elif user_query_lower in generic_queries:
        return "As an AI assistant, I don't have feelings, but I'm here to help you!"

    # Special responses for specific queries
    if "your name" in user_query_lower:
        return "I am Yokie - your inventory chatbot."
    elif "what do you do" in user_query_lower:
        return "I am Yokie. I am here to help you with your home inventory."
    elif "guide me" in user_query_lower:
        return '''Click on the chatbox below and ask your query. 
                  For example, "tell me about the items in my inventory".'''
    elif "steps to use you" in user_query_lower:
        return "Click on the chatbox below and ask me about your home inventory."
    elif "who is your creator" in user_query_lower:
        return "I have been created by the team of HomeSync."
    elif "what can you do" in user_query_lower:
        return '''I can update the quantity of any items in your inventory, 
                  check for details of items in your inventory, 
                  and provide you with a summarized shopping list.'''

    # Handling "can you" queries
    if user_query_lower.startswith("can you"):
        if "home inventory" in user_query_lower:
            return "Yes- I can assist you with your home inventory."
        else:
            return "No"

    if user_query_lower == "generate recipe":
        return generate_recipe_from_inventory(db)
    elif user_query_lower == "show items present":
        available_items_query = "SELECT Items, Quantity FROM inventory WHERE Quantity > 0"
        available_items = db.run(available_items_query)
        if not available_items:
            return "No items available in the inventory."
        items_list = "\n".join([f"{item[0]}: {item[1]}" for item in available_items])
        return f"Items present in inventory:\n{items_list}"

    return None


def is_valid_query(user_query: str) -> bool:
    # Add more sophisticated validation as needed
    if len(user_query.split()) < 2:
        return False
    return True


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm Yokie- Your AI assistant. Ask me anything about your Home inventory."),
    ]

load_dotenv()

st.set_page_config(page_title="Home Sync", page_icon=":shark:")

st.title("Sync With Your Home")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is Our chatbot to chat with Your inventory")
    
    st.text_input("Host", value="autorack.proxy.rlwy.net", key="Host")
    st.text_input("Port", value="49449", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="kgbQryjqbVQFojvZRoMrAPMAHvHCAQer", key="Password")
    st.text_input("Database", value="railway", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    special_response = handle_special_queries(user_query, st.session_state.db)
    if special_response:
        response = special_response
    # elif not is_valid_query(user_query):
    #     response = "Sorry, I could not understand, try a different query."
    else:
        try:
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        except Exception:
            response = "Sorry, I could not understand, try a different query."
    
    st.session_state.chat_history.append(AIMessage(content=response))
    
    with st.chat_message("AI"):
        st.markdown(response)
def update_inventory(db, item_name: str, new_quantity: int) -> str:
    try:
        cursor = db.engine.raw_connection().cursor()
        
        # Check if the item exists in the inventory
        cursor.execute("SELECT * FROM Inventory WHERE Items = %s", (item_name,))
        result = cursor.fetchone()
        
        if result:
            # Update the quantity of the item
            update_query = "UPDATE Inventory SET Quantity = %s WHERE Items = %s"
            cursor.execute(update_query, (new_quantity, item_name))
            db.engine.raw_connection().commit()
            return f"Successfully updated {item_name} quantity to {new_quantity}."
        else:
            return f"Item '{item_name}' not found in the inventory."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def update_inventory_form(db):
    st.subheader("Update Inventory Item")

    # Get the item name and new quantity from the user
    item_name = st.text_input("Enter item name to update:")
    new_quantity = st.number_input("Enter new quantity:", min_value=0, step=1)
    
    if st.button("Update Item"):
        if item_name and new_quantity >= 0:
            # Call the update function to update the item
            response = update_inventory(db, item_name, new_quantity)
            st.success(response)
        else:
            st.warning("Please provide valid inputs for item name and quantity.")
st.markdown("""
<style>
  div[data-testid="stHorizontalBlock"] div[style*="flex-direction: column;"] div[data-testid="stVerticalBlock"] {
    border: 1px solid red;
  }
</style>
""",
  unsafe_allow_html=True,
)
