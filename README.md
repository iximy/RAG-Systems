# RAG-Systems
RAG Systems: main stages of data processing
The Retrieval-Augmented Generation (RAG) process is a rather complex system consisting of many components. The question of how to identify existing RAG methods and their optimal combinations to identify best practices remains the most relevant at the moment. In this article, I want to share my experience in implementing approaches and practices in the field of RAG systems that implement a systematic approach to solving this problem.


Typical tasks of RAG systems processes
- Classification of queries,
- Fragmentation
- Data Vectorization
- Search,
- Regrouping,
- Generalization of data ,
- Formation of the response .

Each of these steps plays an important role in ensuring the accuracy and efficiency of the system. For example, query classification helps determine if a search needs to be performed at all, which can significantly reduce processing time. Rearranging and repacking documents increases the relevance of the results, while summarizing helps eliminate redundancy and improve the quality of responses.

In my opinion, the key difficulty of RAG is that each stage requires careful configuration and selection of optimal methods. For example, how do I choose the size of the fragments to search for? Which embedding model should I use? These issues require not only theoretical analysis, but also practical experiments.

Let’s try to consider all the stages of development related to the structure and data processing of RAG systems.


As can be seen from the visualization of the necessary development steps, the tasks are divided into three levels: the level of preparation, processing and classification of the request, the level of preparation, separation, vectorization and storage of documents, and the level of rearrangement, generation and generalization of the response. Schematically, all the stages can be represented in the form of a diagram:
Let’s look at all the stages in turn.
Classification of queries
Query classification is the first step that allows you to determine if a search is needed at all. Not all queries require additional search, especially if the LLM (Large Language Model) model already has sufficient knowledge. For example, if the query is based solely on information provided by the user, the search may be unnecessary.
Defining the request type:
- Actual search (for example, “How much does it cost to build a house?”)
- Opinion or assessment (“Which is better brick or frame?”)
- Reasoning or synthesis of information (“Where is the best place to choose land for construction”)
- Personal or informal inquiry (“What do you think is better to build a house or live in an apartment?”)
Choosing a search strategy:
- If the query requires factual information, a knowledge base search or vector search is used.
- If the query is complex, multiple information sources are combined.
- If the answer can be given without searching, the LLM model responds directly.
We perform filtering and moderation
- Checking for prohibited content, toxicity, confidential data.
- Detection of irrelevant or potentially harmful queries.
Query classification methods
- Linguistic analysis (analysis of sentence structure, keywords).
- ML classification models (trained on labeled data, for example, BERT or GPT).
- Rules and heuristics (for example, the presence of question words or keywords).

In my opinion, automating this process using a classifier is the best solution. However, it is important to keep in mind that the classifier must be trained on a variety of data in order to avoid errors in real-world scenarios.

Division into parts
Splitting documents into fragments is a critical step that affects the accuracy of the search. It is based on three levels of fragmentation: at the token level, semantic fragmentation, and sentence-level fragmentation. In my opinion, sentence-level fragmentation is the most balanced approach, as it preserves the semantics of the text while remaining simple enough to implement.

In RAG, chunking documents is one of the key steps that affects the effectiveness of search and response generation. This process determines which parts of the text will be used for vector representation and search in the future.

Why divide it?

To optimize the search, because vector search engines (FAISS, Chroma, Weaviate) work more efficiently with short fragments.
To reduce search noise, because when indexing too large chunks of text, the search may produce irrelevant results.

To ensure contextual relevance, since partitioning helps to extract only the necessary information and not overload the LLM with unnecessary data. How to divide?

Fixed-length chunking is the division of text into fixed-length chunks, for example, 512 or 1024 tokens each. Pros: ✔ Simple implementation. ✔ Works well with long documents. Cons: ✖ May share a related context.

Loss of meaning when there is a gap at the border.
Semantic chunking Uses NLP models (for example, BERT) to split into semantic blocks (paragraphs, headings, semantic units).

Pros: ✔ Maintains logical integrity. ✔ Improves the accuracy of the search. Cons: ✖ More complex implementation. ✖ May lead to uneven fragment sizes.
Structural chunking Uses partitioning by logical units of a document (headings, lists, tables).

Pros: ✔ It is well suited for technical documents, articles and reference books. ✔ Preserves the context. Cons: ✖ Requires pre-processing of documents. ✖ Is not always applicable to unstructured data.

Splitting into fragments has a critical effect on the quality of search and generation. The choice of strategy depends on the type of documents and the tasks of the system.

Fixed sizes are suitable for large texts.
Semantic splitting — provides more meaningful chunks.
Structural partitioning is useful for technical data. The best approach is to combine the methods depending on the context!
Choosing an embedding model
The embedding model performs the main function in searching for relevant documents. Experience shows that it is important to combine high performance with a small model size. It is also important to note that the choice of model may depend on the specific task and available resources.

How does it work?
The model converts text into a multidimensional vector representation. These vectors are then used to search for similar fragments in the database.

When choosing a model, it is important to determine:

The language and specifics of text data (domain)

If the system works in English → OpenAI ada-002, BGE, SBERT
For Russian → DeepPavlov/RuBERT, sbert_large_nlu_ru
For code or medical data → CodeBERT, BioBERT
Optimal vector size and speed

Short vectors (384–512) are faster, but less accurate.
Long vectors (1024+) provide accurate search, but require more memory.
The ability to extract semantic connections

Cosine similarity works better with SBERT, BGE
For long texts, E5 or Cohere are better suited.
Database compatibility

If FAISS/Chroma → SBERT, BGE, ada-002 is used
If Milvus/Weaviate → OpenAI is used, Cohere


Search methods such as query generation significantly improve the relevance of the results. However, I would like to note that these methods can be resource-intensive, which is important to consider in real-world applications.

Search (Retrieval) At this stage, the system performs vector search or hybrid search (a combination of vector and classical text search).
Search methods:

Vector search (ANN — Approximate Nearest Neighbors): comparison of query embeddings and database (FAISS, Weaviate, Pinecone).
Classic Search (BM25): keyword search (Elasticsearch, Weaviate, Qdrant).
Hybrid search (BM25 + ANN): combining Hybrid search (BM25 + ANN): combining The two methods to improve accuracy.
2. Re-ranking After the search, the received documents are ranked by relevance. Simple methods (BM25) can return irrelevant results, so re-ranking using neural networks is used.

Ranking models:

Cross-Encoder (SBERT) is a fully trained model for determining the most relevant fragments.
Cohere Rerank is a cloud–based model for improving search.
OpenAI reranking (GPT) — finishing using GPT.


We implement summarization and response generation
Summarizing the extracted documents is a step that helps eliminate redundancy and improve the quality of responses. It is recommended to combine extractive and generative approaches, as this approach ensures high accuracy and efficiency.

After the system has extracted fragments from our knowledge base and ranked them, text processing follows to highlight key ideas.:

During data processing, irrelevant details are deleted: For example, unnecessary words or redundant information that does not affect the response may be removed. Text pre-formation can also be performed, which may include reformulation or simplification of fragments to make information more accessible to perception.

Next, using the Context Preparation for generation:
After extracting relevant documents and fragments, this data forms a context that is used to generate a response.

The context includes both information from the extracted documents and the user’s request itself. It is important that the system correctly matches the query with the necessary fragments from the documents.

In some cases, this context is further structured or cleaned up to exclude unnecessary or inappropriate data that may interfere with generation.generative models create a response based on the processed fragments.

A generative model can take information from many fragments and “compress” it into a concise and accurate answer. The models that are best suited for such tasks are those that are able to coherently generate text based on the provided context. Models can also use attention mechanisms to “remember” key parts of the context necessary for a correct response.

The last part of the generalization is reconciliation. It is important that all the data in the response is logically combined and does not contradict each other. The implementation must comply with the principles of reference, when models must ensure that references to facts are accurate, as well as consistency, when the answer must be logically structured so that there are no sudden leaps of thought.

Models for summarizing in RAG systems

T5 (Text-to-Text Transfer Transformer) is a model that is trained on a variety of text tasks and can perform text generation tasks, including summarization.
It is well suited for generating a text summary and forming a concisely worded response.

BART (Bidirectional and Auto-Regressive Transformers) is a model for compressive and abstract abstraction. It works as an encoder-decoder, extracting information from the text and generating a logical, concise response.

Advantage should be given to models capable of generating responses using the provided context. For example, GPT, They can not only perform summarization, but also perform complex generative tasks, including dialog interfaces.

When choosing summarizing strategies in RAG systems, one should rely on their features. When using compressive Summarization, the system selects significant fragments from the source text without changing them, and combines them to form an answer. This approach is simpler, but limits flexibility.

In the case of Abstract Summarization, the model reformulates the source data in its own words, which gives more flexibility and the ability to create more coherent and logical answers. It’s more complicated, but the result is often more accurate and useful.

The choice of strategies for building RAG systems lies between maximum performance, includes all modules to achieve maximum accuracy and balanced efficiency, this strategy optimizes performance and efficiency, eliminating some resource-intensive methods.

In my opinion, the choice of strategy depends on the specific requirements of the project. For example, if processing speed is important, you can sacrifice some accuracy for efficiency. However, in tasks where accuracy is critically important, it is better to use all modules.

In this article, I have drawn attention to several important ideas.:

The importance of building a modular design: Optimizing each component individually allows you to create flexible and easily configurable systems. This is especially important in complex projects where requirements can change over time.
A systematic approach to practical implementation: the above experience demonstrates how rigorous testing on generally accepted datasets can ensure the reliability of results. These implementation examples may be useful when selecting and analyzing the implementation of RAG stages.
Existing practical limitations: Despite all the advantages of RAG, there are challenges such as generalization to private data, real-time performance, and integration of multimodal data. These issues require additional study when building such systems.
In my opinion, the proposed methods and strategies can significantly improve the quality and efficiency of RAG systems. However, it is important to remember that each task is unique, and successful implementation of RAB requires not only following best practices, but also adapting to specific conditions.
