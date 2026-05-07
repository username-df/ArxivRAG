Web application that retrieves, and answers questions regarding ArXiv research papers.

ArXiv's API is used to obtain recent research papers based on the topic the user inputs.

LangChain's text splitter and ChromaDB are used to create and semantically search for relevant sections of the paper when answering the user's questions. 

Finally, Groq's LLM API is utilized to answer the questions based on the relevant sections of the paper using RAG.
