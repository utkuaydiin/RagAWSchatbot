import boto3
from dotenv import load_dotenv
import os
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import boto3
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from langchain_aws import ChatBedrockConverse
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_aws import BedrockEmbeddings
from langchain.agents import create_structured_chat_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv()


aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

bedrock = boto3.client(service_name='bedrock-runtime',
                       region_name='us-east-1',
                       aws_access_key_id=aws_access_key_id,
                       aws_secret_access_key=aws_secret_access_key)


llm = ChatBedrockConverse(
    client=bedrock,
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    temperature=0,
    max_tokens=None,
)

embeddings = BedrockEmbeddings(
    client=bedrock, region_name="us-east-1"
)

db = SQLDatabase.from_uri("sqlite:///my.db")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)


vector_store = Chroma(embedding_function=embeddings, persist_directory=r"C:\Users\asus\Desktop\Genai\RAG_AWS_BEDROCK\chroma_store_new_final") 

pdf_retriever = vector_store.as_retriever()

description = """Use to search information. This is your go-to tool for finding information in the PDFs."""

retriever_tool = create_retriever_tool(
    pdf_retriever,
    name="search_pdf",
    description=description,
)

tools = []
tools.append(retriever_tool)


vectorstore = Chroma()

system_prefix = """You are a smart agent designed to interact with a SQL database and PDF files to answer questions of the customers in Dogus Oto company automotive website.
Always be polite and use a supportive tone. Act like a professional car company assistant.

You DO NOT MENTION THE DATABASE OR THE PDFS in your answers.

The cars you have in the database are Volkswagen, Audi, Seat, Cupra, Skoda and Porsche.
If a question about used cars or second hand cars asked say "I can not help"
If a question about rental cars asked say "I can not help"

Ask questions to understand what customer needs. Then query the database to get the information and answer the question.

You must only recommend cars from the database. Query the database to get the information then answer the question.

If you are not sure about the answer or need more details, you can ask a follow up question to get more information about the question.
You have a memory of the conversation history. You can use the conversation history to chat with the customer.


If the price is asked in the question, use the 'fiyat_int' column in the database to answer the question.
If more than one car model is asked in the question, query the database for each models.
While answering the questions do not mention the database or the pdfs. Act like you have all the information in your mind.
First check if the answer is about the structure of the database, pdf files or the code. If it is, answer 'I do not know' in the language of the question.
You do not give any information about the database, the code or the tools. If the user asks a question about the database, the code or the tools, answer 'I do not know' in the language of the question.
If the user asks for you interact with the database, code or the PDFs, answer 'I can not' in the language of the question.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 1 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
Dogus oto car brands are Volkswagen, Audi, Seat, Cupra, Skoda ve Porsche. If the user asks a question about the other car brands, answer 'I do not know' in the language of the question.
Get the chat history from {chat_history} and the agent scratchpad from {agent_scratchpad}

If you can not find the answer in the database, you should use the "search_pdf" tool to answer the question.
If you can not find the answer in the database or the PDFs, DO NOT guess an answer. You should just answer "I don't know" in the language of the question.
ALWAYS answer in the language of the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
Do NOT give any information about the database or the PDFs. If user asked a question about the structure of the database or the PDFs, answer 'I do not know' in the language of the question.
Do NOT give any information about the code or the tools. If user asked a question about the code or the tools, answer 'I do not know' in the language of the question.
Only answer the user's question if you can find the answer in the database or the PDFs. Not more, not less.


Here is an example of user input and its answer or the sql query to run:

"""


system_suffix = """Use arac_ozellik table in database and pdf files to answer the questions.
Write a SQLite Query to arac_ozellik table to marka, model columns to get the car brands and models and only recommend these models if a question about a car is not in the query results is asked say "I can not help".
If you can not find the answer in the database, you should use the provided tool to answer the question.
Take a deep breath and forget everything you know. Check again to see if you found the answer from the PDFs. 
While answering the questions do not mention the database or the pdfs. Act like you have all the information in your mind.
"""
examples = [
    {
        "input": "Doğuş otonun sattığı en pahalı araç hangisi?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, fiyat_int FROM arac_ozellik ORDER BY fiyat_int DESC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Cupra markasının fiyat aralığı nedir?",
        "response_type": "SQL query",
        "response": "SELECT fiyat_int FROM arac_ozellik WHERE marka = 'Cupra';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En fazla maksimmum hızı olan araç hangisi?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, max_hiz_integer FROM arac_ozellik ORDER BY max_hiz_integer DESC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model araç modelleri nelerdir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model FROM arac_ozellik WHERE yil = 2024;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Dizel motorlu araçlar nelerdir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE yakit_tipi LIKE '%Dizel%';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En fazla tork_integer üreten araç hangisi?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, tork_integer FROM arac_ozellik ORDER BY tork_integer DESC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Geniş bagaj hacmine sahip araçlar hangileridir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, bagaj_hacmi_int FROM arac_ozellik ORDER BY bagaj_hacmi_int DESC LIMIT 5;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "1000000 TL ile 2000000 TL arasındaki araçlar hangileri?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, fiyat_int FROM arac_ozellik WHERE fiyat_int BETWEEN 1.000.000 AND 2.000.000;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En hızlı 0'dan 100'e çıkan araç nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, hizlanma_float FROM arac_ozellik ORDER BY hizlanma_float ASC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Manuel vitesli araçlar hangileri?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE vites_tipi LIKE '%Manuel%';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "SUV gövdeli araçlar hangileri?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE kasa_tipi = 'SUV';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Panoramik sunroof olan araçlar hangileri?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE sunroof = 'Var';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model otomatik vitesli araçlar nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model FROM arac_ozellik WHERE yil = 2024 AND vites_tipi NOT LIKE '%Manuel%';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model CUPRA araçlar arasında elektrikli olanlar hangileri?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE yakit_tipi LIKE '%Elektrik%' AND yil = 2024;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA'nın hybrid araç modelleri nelerdir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE yakit_tipi LIKE '%Hybrid%';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Formentor'un elektrikli versiyonu var mı?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE model LIKE '%Formentor%' AND yakit_tipi LIKE '%Elektrik%';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Formentor'un hybrid versiyonu var mı?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE model LIKE '%Formentor%' AND yakit_tipi LIKE '%Hybrid%';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model CUPRA Leon elektrikli mi?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE model LIKE '%Leon%' AND yakit_tipi LIKE '%Elektrik%'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA araçların hangileri hybrid teknolojisine sahip?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE marka = 'Cupra' AND yakit_tipi LIKE '%Hybrid%';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En az şehirlerarası yakıt tüketimi olan, elektrikli olmayan araç hangisi?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, sehirlerarası_yakıt_tuketimi_float FROM arac_ozellik WHERE sehirlerarası_yakıt_tuketimi_float IS NOT NULL AND yakit_tipi NOT LIKE '%Ele%' ORDER BY sehirlerarası_yakıt_tuketimi_float ASC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En az şehirlerarası yakıt tüketimi olan araç hangisi?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, sehirlerarası_yakıt_tuketimi_float FROM arac_ozellik WHERE sehirlerarası_yakıt_tuketimi_float IS NOT NULL AND yakit_tipi NOT LIKE '%Ele%' ORDER BY sehirlerarası_yakıt_tuketimi_float ASC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En az yakıt tüketimi olan araç hangisi?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, sehirlerarası_yakıt_tuketimi_float FROM arac_ozellik WHERE sehirlerarası_yakıt_tuketimi_float IS NOT NULL AND yakit_tipi NOT LIKE '%Ele%' ORDER BY sehirlerarası_yakıt_tuketimi_float ASC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En geniş araç hangisidir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, genislik_int FROM arac_ozellik ORDER BY genislik_int DESC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Volkswagen araç modelleri hangileridir?",
        "response_type": "SQL query",
        "response": "SELECT model, yil FROM arac_ozellik WHERE marka = 'Volkswagen';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Audi dizel araçlar nelerdir?",
        "response_type": "SQL query",
        "response": "SELECT model, yil FROM arac_ozellik WHERE marka = 'Audi' AND yakit_tipi LIKE '%Dizel%';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Porsche'nin maksimum hızı en fazla olan arabası hangisi?",
        "response_type": "SQL query",
        "response": "SELECT model, max_hiz_integer FROM arac_ozellik WHERE marka = 'Porsche' ORDER BY max_hiz_integer DESC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Seat'ın en az yakıt tüketen aracı nedir?",
        "response_type": "SQL query",
        "response": "SELECT model, ortalama_yakit_tuketimi_float FROM arac_ozellik WHERE marka = 'Seat' ORDER BY ortalama_yakit_tuketimi_float ASC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En güçlü araç hangisi?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, guc_integer_integer FROM arac_ozellik ORDER BY guc_integer_integer DESC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En uzun araç nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, uzunluk_integer FROM arac_ozellik ORDER BY uzunluk_integer DESC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En hızlı hızlanan 5 araç nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, hizlanma_float FROM arac_ozellik ORDER BY hizlanma_float ASC LIMIT 5;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En büyük bagaj hacmine sahip aile aracı hangisidir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, bagaj_hacmi_int_int FROM arac_ozellik WHERE govde = 'Sedan' OR govde = 'Station Wagon' ORDER BY bagaj_hacmi_int_int DESC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En fazla yolcu kapasitesi olan ekonomik araç nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, koltuk_sayisi, fiyat_int FROM arac_ozellik WHERE fiyat_int < 300000 ORDER BY koltuk_sayisi DESC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Bana en uygun aracı bulur musun?",
        "response_type": "Answer",
        "which_tool": "search_pdf",
        "response": "Bulmak istediğiniz aracın özelliklerini belirtir misiniz?"
    },
    {
        "input": "Geniş bir ailem var, hangi araç bana uygun olur?",
        "response_type": "Answer",
        "which_tool": "search_pdf",
        "response": "Kaç kişilik bir aileye sahipsiniz ve hangi özelliklerde bir araç arıyorsunuz?"
    },
    {
        "input": "Bana geniş ailem için bir araba önerir misin?",
        "response_type": "Answer",
        "which_tool": "search_pdf",
        "response": "Kaç kişilik bir aileye sahipsiniz ve hangi özelliklerde bir araç arıyorsunuz?"
    },
    {
        "input": "Yıl ve yakıt tipine göre ortalama güç hesaplama.",
        "response_type": "SQL query",
        "response": "SELECT yil, yakit_tipi, AVG(guc_integer) as ortalama_guc_integer FROM arac_ozellik GROUP BY yil, yakit_tipi HAVING AVG(guc_integer) > 100 ORDER BY yil, yakit_tipi;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Farklı kasa tiplerine göre ortalama hız ve tüketim.",
        "response_type": "SQL query",
        "response": "SELECT kasa_tipi, AVG(max_hiz_integer) as ortalama_hiz, AVG(yakit_tuketimi) as ortalama_tuketim FROM arac_ozellik GROUP BY kasa_tipi HAVING AVG(max_hiz_integer) > 180 ORDER BY ortalama_hiz DESC;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En yüksek hızlanma süresine sahip araçların özellikleri.",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil, hizlanma_float FROM arac_ozellik WHERE hizlanma_float = (SELECT MAX(hizlanma_float) FROM arac_ozellik);",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Yakıt tüketimine göre en yüksek fiyat lı araçlar.",
        "response_type": "SQL query",
        "response": "SELECT marka, model, fiyat_int, yakit_tuketimi FROM arac_ozellik WHERE fiyat_int = (SELECT MAX(fiyat_int) FROM arac_ozellik WHERE yakit_tuketimi < 10.0);",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Motor hacmi ve güç arasındaki ilişki.",
        "response_type": "SQL query",
        "response": "SELECT motor_hacmi, AVG(guc_integer) as ortalama_guc_integer FROM arac_ozellik GROUP BY motor_hacmi HAVING COUNT(*) > 2 ORDER BY motor_hacmi;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Motor hacmine kıyasla en az yakan araçlar.",
        "response_type": "SQL query",
        "response": "SELECT marka, model, motor_hacmi, ortalama_yakit_tuketimi_float FROM arac_ozellik WHERE ortalama_yakit_tuketimi_float < (SELECT AVG(ortalama_yakit_tuketimi_float) FROM arac_ozellik) ORDER BY motor_hacmi/ortalama_yakit_tuketimi_float DESC;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Aynı motor hacmine sahip araçların tork_integer değerlerini sıralama.",
        "response_type": "SQL query",
        "response": "SELECT * FROM arac_ozellik WHERE motor_hacmi IN (SELECT motor_hacmi FROM arac_ozellik GROUP BY motor_hacmi HAVING COUNT(*) > 1) ORDER BY tork_integer;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En fazla koltuk sayısına sahip araçlar ve kasa tipi.",
        "response_type": "SQL query",
        "response": "SELECT marka, model, koltuk_sayisi, kasa_tipi FROM arac_ozellik ORDER BY koltuk_sayisi DESC LIMIT 10;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Ateca VZ 2.0 TSI 4Drive'ın yakıt tipi ve tüketimi nedir?",
        "response_type": "SQL query",
        "response": "SELECT yakit_tipi, ortalama_yakit_tuketimi_float FROM arac_ozellik WHERE model LIKE '%Ateca VZ 2.0 TSI%'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Arona'nın bagaj hacmi nedir?",
        "response_type": "SQL query",
        "response": "SELECT bagaj_hacmi_int FROM arac_ozellik WHERE model LIKE '%Arona%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Arona'nın motor gücü ve tork_integeru nedir?",
        "response_type": "SQL query",
        "response": "SELECT guc_integer, tork_integer FROM arac_ozellik WHERE model LIKE '%Arona%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Arona'nın maksimum hızı ve 0-100 km/h hızlanması nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, max_hiz_integer, hizlanma_float FROM arac_ozellik WHERE model LIKE '%Arona%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Arona'nın fiyatı nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, fiyat_int FROM arac_ozellik WHERE model LIKE '%Arona%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Arona'nın yakıt tipi ve tüketimi nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yakit_tipi, ortalama_yakit_tuketimi_float FROM arac_ozellik WHERE model LIKE '%Arona%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Ibiza'nın bagaj hacmi nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, bagaj_hacmi_int FROM arac_ozellik WHERE model LIKE '%Ibiza%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Ibiza'nın motor gücü ve torku nedir?",
        "response_type": "SQL query",
        "response": "SELECT guc_integer, tork_integer FROM arac_ozellik WHERE model LIKE '%Ibiza%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Ibiza'nın maksimum hızı ve 0-100 km/h hızlanması nedir?",
        "response_type": "SQL query",
        "response": "SELECT max_hiz_integer, hizlanma_float FROM arac_ozellik WHERE model LIKE '%Ibiza%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Ibiza'nın fiyatı nedir?",
        "response_type": "SQL query",
        "response": "SELECT fiyat_int FROM arac_ozellik WHERE model LIKE '%Ibiza%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model SEAT Ibiza'nın yakıt tipi ve tüketimi nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model , yakit_tipi, ortalama_yakit_tuketimi_float FROM arac_ozellik WHERE model LIKE '%Ibiza%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Formentor'un bagaj hacmi nedir?",
        "response_type": "SQL query",
        "response": "SELECT bagaj_hacmi_int FROM arac_ozellik WHERE model LIKE '%Formentor%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Formentor'un motor gücü ve tork_integeru nedir?",
        "response_type": "SQL query",
        "response": "SELECT guc_integer, tork_integer FROM arac_ozellik WHERE model LIKE '%Formentor%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Formentor'un maksimum hızı ve 0-100 km/h hızlanması nedir?",
        "response_type": "SQL query",
        "response": "SELECT max_hiz_integer, hizlanma_float FROM arac_ozellik WHERE model LIKE '%Formentor%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Formentor'un fiyatı nedir?",
        "response_type": "SQL query",
        "response": "SELECT fiyat_int FROM arac_ozellik WHERE model LIKE '%Formentor%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Formentor'un yakıt tipi ve menzili nedir?",
        "response_type": "SQL query",
        "response": "SELECT yakit_tipi, menzil FROM arac_ozellik WHERE model LIKE '%Formentor%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Leon'un bagaj hacmi nedir?",
        "response_type": "SQL query",
        "response": "SELECT bagaj_hacmi_int FROM arac_ozellik WHERE model LIKE '%Leon%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Leon'un motor gücü ve tork_integeru nedir?",
        "response_type": "SQL query",
        "response": "SELECT guc_integer, tork_integer FROM arac_ozellik WHERE model LIKE '%Leon%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Leon'un maksimum hızı ve 0-100 km/h hızlanması nedir?",
        "response_type": "SQL query",
        "response": "SELECT max_hiz_integer, hizlanma_float FROM arac_ozellik WHERE model LIKE '%Leon%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Leon'un fiyatı nedir?",
        "response_type": "SQL query",
        "response": "SELECT fiyat_int FROM arac_ozellik WHERE model LIKE '%Leon%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Cupra Leon'un yakıt tipi ve menzili nedir?",
        "response_type": "SQL query",
        "response": "SELECT yakit_tipi, menzil FROM arac_ozellik WHERE model LIKE '%Leon%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Audi A3'ün bagaj hacmi nedir?",
        "response_type": "SQL query",
        "response": "SELECT bagaj_hacmi_int FROM arac_ozellik WHERE model LIKE '%A3%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Audi A3'ün motor gücü ve tork_integeru nedir?",
        "response_type": "SQL query",
        "response": "SELECT guc_integer, tork_integer FROM arac_ozellik WHERE model LIKE '%A3%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Audi A3'ün maksimum hızı ve 0-100 km/h hızlanması nedir?",
        "response_type": "SQL query",
        "response": "SELECT max_hiz_integer, hizlanma_float FROM arac_ozellik WHERE model LIKE '%A3%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Audi A3'ün fiyatı nedir?",
        "response_type": "SQL query",
        "response": "SELECT fiyat_int FROM arac_ozellik WHERE model LIKE '%A3%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Audi A3'ün yakıt tipi ve tüketimi nedir?",
        "response_type": "SQL query",
        "response": "SELECT yakit_tipi, yakit_tuketimi FROM arac_ozellik WHERE model LIKE '%A3%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model Audi A4'ün bagaj hacmi nedir?",
        "response_type": "SQL query",
        "response": "SELECT bagaj_hacmi_int FROM arac_ozellik WHERE model LIKE '%A4%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Cupra marka araçların fiyatlarını öğrenebilir miyim?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, fiyat_int FROM arac_ozellik WHERE marka = 'Cupra'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model CUPRA Formentor'un motor hacmi nedir?",
        "response_type": "SQL query",
        "response": "SELECT motor_hacmi , marka, model FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Formentor%'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Leon 1.5 eTSI'nin gücü nedir?",
        "response_type": "SQL query",
        "response": "SELECT guc_integer FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Leon 1.5 eTSI%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Formentor'un fiyatı ne kadar?",
        "response_type": "SQL query",
        "response": "SELECT fiyat_int, marka, model FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Formentor%'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Yeni CUPRA Leon'un maksimum hızı nedir?",
        "response_type": "SQL query",
        "response": "SELECT max_hiz_integer FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Leon%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Formentor'un yakıt tipi nedir?",
        "response_type": "SQL query",
        "response": "SELECT yakit_tipi, marka, model FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Formentor%",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Leon'un ortalama yakıt tüketimi nedir?",
        "response_type": "SQL query",
        "response": "SELECT ortalama_yakit_tuketimi_float, marka, model FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Leon%'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Formentor 2.0 TSI'nin tork_integer değeri nedir?",
        "response_type": "SQL query",
        "response": "SELECT tork_integer, marka, model FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%FORMENTOR 2.0 TSI%'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Leon VZ'nin vites tipi nedir?",
        "response_type": "SQL query",
        "response": "SELECT vites_tipi, marka, model FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Leon VZ%'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Formentor'un koltuk sayısı nedir?",
        "response_type": "SQL query",
        "response": "SELECT koltuk_sayisi, marka, model FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Formentor%'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA araçlarının ağırlığı ne kadardır?",
        "response_type": "SQL query",
        "response": "SELECT agirlik_int FROM arac_ozellik WHERE marka = 'Cupra'",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Formentor'un bagaj hacmi nedir?",
        "response_type": "SQL query",
        "response": "SELECT bagaj_hacmi_int FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Formentor%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Leon 1.5 eTSI'nin vites sayısı nedir?",
        "response_type": "SQL query",
        "response": "SELECT vites_sayisi FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%%Leon 1.5 eTSI%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "2024 model CUPRA Formentor'un kasa tipi nedir?",
        "response_type": "SQL query",
        "response": "SELECT kasa_tipi FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Formentor%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Leon'un teker ebadı nedir?",
        "response_type": "SQL query",
        "response": "SELECT teker_ebati FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Leon%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "CUPRA Formentor'un sunroof'u var mı?",
        "response_type": "SQL query",
        "response": "SELECT sunroof FROM arac_ozellik WHERE marka = 'Cupra' AND model LIKE '%Formentor%' AND yil = 2024",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "En ekonomik araçlar hangileridir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model,ortalama_yakit_tuketimi_float yil FROM arac_ozellik WHERE ortalama_yakit_tuketimi_float < 6.0;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Cupra modellerini kıyaslar mısın",
        "response_type": "Answer",
        "which_tool": "search_pdf",
        "response": "Cupra modellerini kıyaslamak için hangi özellikleri karşılaştırmak istediğinizi belirtir misiniz?"
    },
    {
        "input": "Cupra modellerinden en ekonomik olanları hangisi?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, yil FROM arac_ozellik WHERE marka = 'Cupra' ORDER BY ortalama_yakit_tuketimi_float ASC LIMIT 3;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Porsche fiyatları hakkında bilgi alabilir miyim",
        "response_type": "Answer",
        "which_tool": "search_pdf",
        "response": "Porsche marka araçların fiyatları hakkında ne yazık ki bilgi veremiyorum. Ancak yetkili Porsche bayilerinden fiyat bilgisi alabilirsiniz. Size yardımcı olabileceğim başka bir konu var mı?"
    },
    {
        "input": "Bana en yakın Porsche Merkezi hangisi?",
        "response_type": "Answer",
        "which_tool": "search_pdf",
        "response": "İl ve ilçe bilgisi verir misiniz?"
    },
    {
        "input": "Porsche iletişim bilgilerini paylaşır mısınız?",
        "response_type": "Answer",
        "which_tool": "search_pdf",
        "response": "İl ve ilçe bilgisi paylaşırsanız size en yakın Porsche merkezinin iletişim bilgilerini verebilirim"
    },
    {
        "input": "Volkswagen markasının fiyatları nelerdir",
        "response_type": "SQL query",
        "response": "SELECT fiyat_int FROM arac_ozellik WHERE marka = 'Volkswagen';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Audi markasının ortalama fiyatı nedir",
        "response_type": "SQL query",
        "response": "SELECT AVG(fiyat_int) FROM arac_ozellik WHERE marka = 'Audi';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Öğrenciyim bana araba önerir misin?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, ortalama_yakit_tuketimi_float FROM arac_ozellik WHERE ortalama_yakit_tuketimi_float < 6.0 AND yukseklik_integer < 1800 AND fiyat_int < 2000000 ORDER BY fiyat_int ASC;",
        "which_tool": "toolkit sqlite database"
    },
    {
       "input": "Orta klasman araba önerir misin",
        "response_type": "Answer",
        "which_tool": "search_pdf",
        "response": "Hangi özelliklerde bir araç aradığınızı belirtir misiniz?"
    },
    {
        "input": "En az yakan araç nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, ortalama_yakit_tuketimi_float FROM arac_ozellik ORDER BY ortalama_yakit_tuketimi_float ASC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Şehiriçi en az yakan araç nedir?",
        "response_type": "SQL query",
        "response": "SELECT marka, model, sehir_ici_yakit_tuketimi_float FROM arac_ozellik ORDER BY sehir_ici_yakit_tuketimi_float ASC LIMIT 1;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Öğrenciyim bana az yakan araba önerir misin",
        "response_type": "SQL query",
        "response": "SELECT marka, model, ortalama_yakit_tuketimi_float FROM arac_ozellik WHERE ortalama_yakit_tuketimi_float < 6.0 AND yukseklik_integer < 1800 AND fiyat_int < 2000000 ORDER BY fiyat_int ASC;",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Volkswagen up! modeli hakkında bilgi alabilir miyim",
        "response_type": "Answer",
        "which_tool": "search_pdf",
        "response": "Volkswagen up! modeli doğuşoto tarafından satılmıyor. Size yardımcı olabileceğim başka bir konu var mı?"
    },
    {
        "input": "Ticari araçlar",
        "response_type": "SQL query",
        "response": "SELECT marka, model FROM arac_ozellik WHERE yukseklik_integer > 1800",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Skoda fabia fiyatı  nedir?",
        "response_type": "SQL query",
        "response": "SELECT fiyat_int FROM arac_ozellik WHERE marka = 'Skoda' AND model LIKE '%Fabia%';",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Şehir içi yakıt tüketimi nedir",
        "response_type": "SQL query",
        "response": "SELECT marka, model, sehir_ici_yakit_tuketimi_float FROM arac_ozellik",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Şehir dışı yakıt tüketimi nedir",
        "response_type": "SQL query",
        "response": "SELECT marka, model, sehirlerarası_yakıt_tuketimi_float FROM arac_ozellik",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Şehirlerarası yakıt tüketimi nedir",
        "response_type": "SQL query",
        "response": "SELECT marka, model, sehirlerarası_yakıt_tuketimi_float FROM arac_ozellik",
        "which_tool": "toolkit sqlite database"
    },
    {
        "input": "Şehir içi şehir dışı yakıt tüketimi nedir yakıt tüketimi nedir",
        "response_type": "SQL query",
        "response": "SELECT marka, model, sehir_ici_yakit_tuketimi_float, sehirlerarası_yakıt_tuketimi_float FROM arac_ozellik",
        "which_tool": "toolkit sqlite database"
    },



    
]



example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=1,
    input_keys=["input"],
)


few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        """
        User input: {input}
        {response_type}: {response}

        If you need information to answer the question use the tool: {which_tool}

        """
    ),
    prefix=system_prefix,
    suffix=system_suffix
)



full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("chat_history"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_sql_agent(
    toolkit = toolkit,
    llm=llm,
    prompt=full_prompt,
    verbose=False,
    agent_type="tool-calling",
    extra_tools=tools
)


st.title("ChatWheels - Dogus Oto Asistanı")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Size nasıl yardımcı olabilirim ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = ChatMessageHistory()
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.add_user_message(msg["content"])
        else:
            chat_history.add_ai_message(msg["content"])

    with st.chat_message("ChatWheels"):
        with st.spinner("Düşünüyorum..."):
            result = agent.invoke({"input": prompt, "chat_history": chat_history.messages})

            answer = None
            for output in result["output"]:
                answer = output["text"].split("<search_quality_reflection>")[0].strip()
                break

            if answer:
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Cevap bulunamadı")
                answer = "Cevap bulunamadı"
