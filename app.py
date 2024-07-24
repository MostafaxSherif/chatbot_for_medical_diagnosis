import streamlit as st
from PIL import Image
from streamlit_chat import message
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import random

# Configure layout
logo = Image.open('6517512.png')
st.set_page_config(
    page_title='Medical bot',
    page_icon=logo, 
    menu_items={
        'About': "# *Bot helps with some medical problems consultations!*"
    },
    layout="centered"
    )
# logo = Image.open('4416431.png')
st.title('⚕️👨‍💻🤖  Medical Bot')

# Add logo and bot name to the web page
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo
st.sidebar.image(add_logo(logo_path="6517512.png", width=240, height=250)) 
st.sidebar.header("Medical Bot")
st.sidebar.markdown("Bot helps to answer some medical questions!")

# Chat Bot functionality
@st.cache_resource()
def model_load(llm_name="llama-2-7b-chat.ggmlv3.q2_K.bin",embed_m_name='sentence-transformers/all-mpnet-base-v2',
               temperature=.2,
               max_tokens=256,
               vdb_loc='VDB/medicalDocs/'):
    """" read llm model, embeding model, and load vector store
        Parameters:
        -------------
        llm_name: str
            location of the LLM model
        embed_m_name: str
            location of the Embedding model
        Temperature: float
            unpredictability of a language model's output, the higher temperature the more creative output
        max_tokens: int
            The maximum number of tokens to generate in the completion 
        vdb_loc: str
            location of the Vector store DB

        Returns:
        -------------
        llm model object and vector store object
            """         
    llm = CTransformers(model=llm_name,model_type="llama",
                config={'max_new_tokens': max_tokens, 'temperature': temperature})

    embeddings_model = HuggingFaceEmbeddings(model_name=embed_m_name,model_kwargs={'device': 'cpu'})
    vectordb = Chroma(
        persist_directory=vdb_loc,
        embedding_function=embeddings_model
    )
    return llm,vectordb
# define prompot for chatbot
def prompt_defination():
    """
    Function helps LLM model to solve a problem by providing steps before giving a final answer

    Returns:
    ---------
    prompt: object
    """
    template = """"Use the following pieces of context to answer the question at the end. 
                    If you do not know the answer, please think rationally and answer from your own knowledge base. 
                    Keep the answer as concise as possible. Always say "thanks for asking! " at the end of the answer.

                    Context: {context} 
                    Question: {question}
                """
    prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])

    return prompt 


# get embedding and llm model to build RetrievalQA
@st.cache_resource()
def create_qa_chain():
    """
    load LLM model and VectorStore and build chain object that will receive the query and go through llm, vector store, and prompt template

    to build the final answer 
    """
    # load the llm, vector store, and the prompt
    models=model_load()
    llm,vdb= models[0],models[1]
    del(models)
    prompt = prompt_defination()

    # create the qa_chain
    retriever = vdb.as_retriever(search_kwargs={'k': 1})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True,
                                       chain_type_kwargs={
                                            "verbose": True,
                                            "prompt": prompt,
                                            "memory": ConversationBufferMemory(
                                                memory_key="history",
                                                input_key="question")
                                        })
    
    return qa_chain

def generate_response(query, qa_chain):
    """
    send query to chain object
    """
    
    return qa_chain({'query':query})


# object from create_qa_chain
qa_chain = create_qa_chain()

assistant_response = random.choice(
    [
        "This is Medical bot! How can I assist you today?",
        "how can I help?",
        "I'm med bot! send me your medical query",
        "Thanks for chosing me to help! share your medical query",
    ]
)
message(assistant_response, is_user=False) 

# create empty lists for user queries and responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# React to user input
if prompt := st.chat_input("Say something![send your medical query]"):
# Display user message in chat message container

    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = generate_response(prompt,qa_chain)
    result=response["result"]
    src_doc=response["source_documents"]
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        print(src_doc)
        st.markdown(result)
        st.markdown(src_doc)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})

  