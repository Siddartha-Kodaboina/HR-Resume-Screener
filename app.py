
import os
from typing import List
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
import chainlit as cl

# Set up OpenAI API key


# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "resume-db-new"
namespace = None

# Initialize the OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize the Ollama model
# llm = Ollama(model="llama3.2:1b")
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

welcome_message = "Welcome to the Machine Engineer Recruiter Bot! Ask anything you would like to know/find regarding this position."
system_prompt = """You role is job recruiter who is hiring for a ML Engineer position with the below Job Description
The job description is- 
Develop, train, and optimize machine learning models
Conduct research on state-of-the-art machine learning techniques
Deploy and monitor machine learning models in production environments
Collaborate with software engineers to integrate ML models into applications
Qualifications:
Strong background in machine learning algorithms and libraries
Experience with Python, TensorFlow, PyTorch, or similar frameworks
Experience with cloud platforms such as AWS, GCP, or Azure
Familiarity with data processing tools such as Hadoop, Spark, etc.
Proficiency in statistics and applied mathematics
Excellent problem-solving skills and attention to detail, answer questions with this in mind."""

@cl.on_chat_start
async def start():
    await cl.Message(content=welcome_message).send()
    
    # Check if the index exists, if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # Dimension for "text-embedding-ada-002"
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    # Get the Pinecone index
    index = pc.Index(index_name)
    
    # Create the docsearch object using LangchainPinecone
    docsearch = LangchainPinecone.from_existing_index(index_name, embeddings, namespace=namespace)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=  message_history,
        return_messages=True,
        
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    
    if chain is None:
        await cl.Message(content="Sorry, there was an error initializing the chat. Please try again.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ainvoke({"question": message.content}, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()