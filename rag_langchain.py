from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import  HuggingFacePipeline
import torch
from transformers import pipeline ,AutoModelForCausalLM, AutoTokenizer
from langchain.document_loaders import DirectoryLoader
from huggingface_hub import login
login()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)



folder_path = r'vector_stores\documents'
text_loader_kwargs={'autodetect_encoding': True}
mixed_loader = DirectoryLoader(
    path=folder_path,
    glob=r'.\*.txt',
    loader_cls=TextLoader,
    loader_kwargs=text_loader_kwargs
)

doc = mixed_loader.load()

#loader=TextLoader(r'vector_stores\poison_frog.txt')
#loader=TextLoader(r'vector_stores\poison_frog.txt',r'vector_stores\pride_and_prejudice.txt',encoding = 'UTF-8')
#doc=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
docs=text_splitter.split_documents(doc)

print(docs[6])
from sentence_transformers import SentenceTransformer
encoder = HuggingFaceEmbeddings()
#encoder1=HuggingFaceEmbeddings()
text="hello how are you"
#embeddings = encoder.encode(text)
#print(embeddings)

#dimension = embeddings.shape[1]
#index = faiss.IndexFlatL2(dimension)  
#index.add(embeddings)
db=FAISS.from_documents(documents=docs,embedding=encoder)

print(db)
retriever = db.as_retriever(search_kwargs={"k": 10})


# Query document using the retriever
#query = "who is jane?"
#retrieved_documents = retriever.get_relevant_documents(query)
#print(retrieved_documents)

from langchain.prompts import ChatPromptTemplate

#template = """You are an assistant for question-answering tasks. 
#Use the following pieces of retrieved context to answer the question. 
#If you don't know the answer, just say that you don't know. 
#Question: {question} 
#Context: {context} 
#Answer:
#"""
#prompt = ChatPromptTemplate.from_template(template)

#print(prompt)

import bitsandbytes
print(bitsandbytes.__spec__)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,Trainer, TrainingArguments
#from huggingface_hub import login
#login()
import os
torch.cuda.empty_cache()
# Define the model path
#model_path = "google/gemma-1.1-2b-it"


# Define quantization configuration
quantization_config = BitsAndBytesConfig(load_in_4bit=True)


# Load the tokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_path,quantization_config=quantization_config)

# Load the model
#model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config=quantization_config)

model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it",quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it",quantization_config=quantization_config)
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    max_new_tokens=512
)

messages = [
    {"role": "user", "content": "Where is Milan?"},
]
prompt =pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(
	prompt,
	max_new_tokens=256,
	add_special_tokens=True,
	do_sample=True,
	temperature=0.7,
	top_k=50,
	top_p=0.95
)
#print(outputs[0]["generated_text"][len(prompt):])

# Move input tensors to GPU if available
#input_text = "ek kavita likho."
#input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate output
#outputs = model.generate(**input_ids,max_length=200)
#print(tokenizer.decode(outputs[0]))

#from langchain.chat_models import ChatOpenAI
#from langchain.schema.runnable import RunnablePassthrough
#from langchain.schema.output_parser import StrOutputParser
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

gemma_llm = HuggingFacePipeline(
    pipeline=pipeline,
    model_kwargs={"temperature": 0.7},
) 

#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa = RetrievalQA.from_chain_type(
    #{"context": retriever,  "question": RunnablePassthrough()} 
    llm=gemma_llm,
    retriever=retriever,
    chain_type="stuff"
    #StrOutputParser() 
)


1
#query = "write a character summary of darcy"
print("press 1 to start")
query=input()
print("enter query")
while query!=0:
    query=input()
    if query=='exit()':
        exit()
    else:
            
     response=qa.invoke(query)
     print(response)

